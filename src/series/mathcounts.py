"""Mathcounts: one test per round.

On-disk layout (data dir is ``Mathcounts/out``)::

    out/<year>/<level>/<round>.pdf

A ``<year>/<level>`` folder mixes problem rounds (sprint, target, team,
countdown, ...) with a single shared ``solutions.pdf`` and ``answers.pdf`` that
cover several rounds at once. Each problem round is its own test (id
``<year>_<level>_<round>``, e.g. ``2025_state_sprint``); the round whitelist is
``config.MATHCOUNTS_TEST_ROUNDS``.

Worked solutions are **deferred**: the shared ``solutions.pdf`` restarts
numbering across rounds, so mapping it back to individual rounds needs a real
OCR run to inspect the alignment first; `has_solutions` stays False.

**Answers** are wired: the shared ``answers.pdf`` is a title page followed by a
page run per round, each page headed "<Round> Round" ("Sprint Round" spans two
pages, then Target, Team, and the remaining pages Countdown). `parse_answers`
selects the parsed test's pages by that header in the OCR markdown -- some
years are scanned with no text layer, so the header must come from OCR, not
the PDF -- and reads the ``N. ____ answer`` blank lines off them.
"""

import re
from pathlib import Path

from .. import config
from .base import Series, Test, numbered_answers_in_line

# Round stem (the test PDF's filename) -> the header its answer-key pages carry.
# Practice rounds (warmups/workouts/masters) publish separate keys with a
# different layout and are not wired yet.
_ANSWER_ROUND_HEADERS = {
    "sprint": "sprint round",
    "target": "target round",
    "team": "team round",
    "countdown": "countdown round",
    "cdr": "countdown round",
}

# MATHCOUNTS fill-in-the-blank problems print "____ cm" or "____ factors" (a
# blank plus an optional unit) right after the problem number, before the
# statement itself ("In the figure, ..."). Both engines already strip the
# printed problem number and collapse the blank rule; what is left dangling at
# the front of the statement is just the unit word(s). Strip the blank run
# first (in case an engine left literal underscores behind), then up to a few
# lowercase unit words -- but only when a capitalized statement follows, so a
# blank-less problem's real text is never touched.
_LEADING_BLANK_RE = re.compile(r"^_+\s*")
_LEADING_UNIT_RE = re.compile(r"^(?:[a-z][\w°²³./-]*\s+){1,3}(?=[A-Z(])")

# Boilerplate that marks a page as having no problems worth parsing: the cover
# sheet in front of every round, the divider MATHCOUNTS reprints before each
# pair of Target Round problems ("every other page" in that round), and the
# Forms of Answers page some rounds append at the end. All three carry this
# exact instructional text regardless of year/level, so a plain substring
# check on the PDF's own text layer is enough -- no OCR needed to skip them.
_SKIP_PAGE_PHRASES = ("do not begin until you are instructed", "forms of answers")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_CODE_FENCE_RE = re.compile(r"^\s*```(?:html)?\s*$", re.IGNORECASE | re.MULTILINE)
_PRINTING_FOOTER_RE = re.compile(
    r"\n\s*(?:<img\b[^>]*>.*?</img>\s*)?"
    r"Printing of this competition is underwritten by\b.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_TARGET_ROUND_RE = re.compile(r"\bTarget\s+Round\b", re.IGNORECASE)
_NUMBERED_STATEMENT_RE = re.compile(
    r"(?:^|\n|<t[dh]\b[^>]*>)\s*(\d{1,2})\s*[.)]\s+",
    re.IGNORECASE,
)
_PIPE_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_BLOCK_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_TABLE_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_TABLE_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ANSWER_NUMBER_RE = re.compile(r"^\s*(\d{1,3})\s*[.)]\s+")
_ANSWER_BLANK_RE = re.compile(r"_{2,}")

# A handful of historical keys have a cell that the OCR omits completely, so
# neither the printed marker nor its answer survives in ``pages_markdown``.
# These values were transcribed directly from the official answer PDFs during
# the 2026-07 answer-key audit.  Keep this deliberately tiny and keyed to the
# immutable test id: normal malformed cells are recovered structurally above.
# 2000 National is the sole exception whose archive folder no longer contains
# an answer key; those three answers were independently verified from the
# retained problem statements and figure crop.
_VERIFIED_MISSING_ANSWERS = {
    "2000_national_cdr": {43: "4", 50: "97610", 51: r"\frac{2}{25}"},
    "2003_state_cdr": {15: r"\frac{\sqrt{3}}{2}", 55: r"\frac{5}{6}"},
    "2004_state_cdr": {
        14: r"\frac{3m^2}{4p^3}",
        18: r"\frac{1}{8}",
        29: "21 (square centimeters)",
        41: "511",
    },
    # The current cached OCR renders these seven cells as blank even though
    # their labels remain visible.  Verified against page 7 of the 2007 State
    # answer PDF; retaining them prevents a forced refresh from regressing a
    # previously complete key.
    "2007_state_cdr": {
        42: "6",
        43: "51",
        44: r"\frac{9}{7}",
        46: "6",
        60: "-1",
        68: "-2",
        70: r"\frac{1}{4}",
    },
}


def _table_answer_recoveries(markdown, *, upper_bound):
    """Recover a dropped answer number only when its table slot proves it.

    Early MATHCOUNTS answer keys arrange a round in independent numbered
    columns.  The OCR occasionally retains an answer cell but drops its
    ``N.`` marker.  HTML row/cell structure still tells us which numbered
    column it belongs to, unlike the line-oriented generic key parser.

    This deliberately handles just a one-number gap between adjacent cells,
    or an endpoint adjoining a consecutive run.  It never fills an empty cell
    or guesses across columns/pages.
    """
    recovered = {}
    for table in _TABLE_BLOCK_RE.findall(markdown):
        rows = []
        for row in _TABLE_ROW_RE.findall(table):
            cells = []
            for cell in _TABLE_CELL_RE.findall(row):
                text = _ANSWER_BLANK_RE.sub(" ", _HTML_TAG_RE.sub(" ", cell))
                text = re.sub(r"\s+", " ", text).strip()
                marker = _ANSWER_NUMBER_RE.match(text)
                if marker:
                    number = int(marker.group(1))
                    answer = text[marker.end() :].strip()
                else:
                    number, answer = None, text
                cells.append((number, answer))
            if cells:
                rows.append(cells)

        width = max((len(row) for row in rows), default=0)
        for column in range(width):
            entries = [
                (row_index, row[column])
                for row_index, row in enumerate(rows)
                if column < len(row) and row[column][1]
            ]
            # Work left-to-right so two unlabeled terminal cells can be
            # recovered as a short consecutive run (e.g. Sprint 29, 30).
            resolved = [list(item[1]) for item in entries]
            for index, (_, cell) in enumerate(entries):
                number, answer = resolved[index]
                if number is not None or not answer:
                    continue
                previous = resolved[index - 1][0] if index else None
                following = resolved[index + 1][0] if index + 1 < len(entries) else None
                candidate = None
                if previous is not None and following == previous + 2:
                    candidate = previous + 1
                elif following is not None:
                    # A leading cell immediately before a consecutive run
                    # (15, 16, 17, ...) has exactly one possible label.
                    tail = [item[0] for item in resolved[index + 1 : index + 3]]
                    if len(tail) == 2 and tail[1] == tail[0] + 1:
                        candidate = following - 1
                elif previous is not None:
                    # Likewise for a nonempty terminal cell after a run.
                    head = [item[0] for item in resolved[max(0, index - 2) : index]]
                    if len(head) == 2 and head[0] == previous - 1:
                        candidate = previous + 1
                if candidate is not None and 1 <= candidate <= upper_bound:
                    recovered.setdefault(candidate, answer)
                    resolved[index][0] = candidate
    return recovered


class MathcountsSeries(Series):
    name = "mathcounts"
    has_solutions = (
        False  # shared per-level solutions.pdf deferred -- see module docstring
    )
    has_answers = True
    ignored_test_substrings = ("masters", "warmup", "workout")

    def layout_options(self):
        """Opt into the MATHCOUNTS-tuned nanonets figure/table heuristics.

        MATHCOUNTS pages need these hooks (see `config.LayoutOptions`): a recurring
        whole-page false-positive Picture box (filtered by area), problems packed
        into a single answer-blank ``<table>`` (marker rows unpacked to text), and
        faint number boxes that sometimes miss detection and drop a problem's
        left-margin start (recovered by the gap-based fallback). Genuine figures
        frequently extend into the right or bottom page bands, so MATHCOUNTS must
        not use the positional furniture filters. Target divider/logo pages are
        suppressed by ``clean_statement_markdown`` instead. Other series keep the
        conservative base defaults.
        """
        return config.LayoutOptions(
            max_picture_area_frac=config.NANONETS_MAX_PICTURE_AREA_FRAC,
            gap_based_picture_fallback=True,
            split_marker_table_rows=True,
            prefer_inline_picture_tags=True,
            drop_sponsor_watermark_picture=True,
            # Numbered package/condition lists inside a problem must not be
            # mistaken for a new section merely because they restart at 1.
            strict_section_restarts=True,
            consecutive_problem_markers=True,
        )

    def skip_page(self, text):
        # Older booklets double-space this boilerplate ("DO  NOT  BEGIN...")
        # and can wrap it across a line break; collapse all whitespace runs
        # (including newlines) to a single space before matching.
        collapsed = re.sub(r"\s+", " ", text).strip().lower()
        return any(phrase in collapsed for phrase in _SKIP_PAGE_PHRASES)

    def clean_statement_markdown(self, page_index, markdown):
        """Remove scanned score sheets and OCR-only statement furniture."""
        collapsed = re.sub(r"\s+", " ", markdown).strip().lower()
        if any(phrase in collapsed for phrase in _SKIP_PAGE_PHRASES):
            return ""
        markdown = _CODE_FENCE_RE.sub("", markdown)
        # The model occasionally emits a Markdown image path instead of the
        # requested <img> tag. DETR supplies the authoritative local crop, so
        # keeping this invented/broken path only duplicates the figure.
        markdown = _MARKDOWN_IMAGE_RE.sub("", markdown)
        # Older National Target sheets print a sponsor logo and underwriting
        # line after the questions. The crop is removed geometrically by the
        # footer band; remove the corresponding OCR text here.
        markdown = _PRINTING_FOOTER_RE.sub("", markdown)
        # Some responses use a Markdown pipe table instead of HTML. Convert
        # answer-blank rows to the same plain ``N. statement`` form consumed by
        # parse_layout; genuine data-table rows have no numbered blank first
        # cell and remain untouched.
        lines = []
        for line in markdown.splitlines():
            if _PIPE_TABLE_ROW_RE.match(line):
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if len(cells) >= 2:
                    marker = re.match(r"^(\d{1,2})\s*[.)]\s+.*_{2,}", cells[0])
                    if marker is not None:
                        lines.append(f"{marker.group(1)}. {' | '.join(cells[1:])}")
                        continue
            lines.append(line)
        return "\n".join(lines)

    def validate_statement_markdown(self, page_index, markdown):
        """Reject a Target problem sheet when OCR silently loses half its pair."""
        collapsed = re.sub(r"\s+", " ", markdown).strip().lower()
        if any(phrase in collapsed for phrase in _SKIP_PAGE_PHRASES):
            return True  # accepted, then intentionally suppressed by cleanup
        markers = {int(m.group(1)) for m in _NUMBERED_STATEMENT_RE.finditer(markdown)}
        # No valid MATHCOUNTS problem page in the cached corpus contains exactly
        # one numbered statement; Target pages always contain a pair, and the
        # other rounds contain larger batches. This also catches a truncated
        # response that lost the footer identifying it as a Target page.
        if len(markers) == 1:
            return False
        if _TARGET_ROUND_RE.search(markdown):
            return len(markers) >= 2
        return True

    def discover_tests(self, data_dir):
        """One test per whitelisted ``<year>/<level>/<round>.pdf``."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for pdf in sorted(root.glob("*/*/*.pdf")):
            if pdf.stem not in config.MATHCOUNTS_TEST_ROUNDS:
                continue
            test_id = f"{pdf.parent.parent.name}_{pdf.parent.name}_{pdf.stem}"
            tests.append(Test(id=test_id, source=pdf))
        return tests

    def answer_source(self, test):
        """The shared ``answers.pdf`` sibling covering every round, or None."""
        if test.source.stem not in _ANSWER_ROUND_HEADERS:
            return None  # practice rounds -- separate key files, not wired yet
        src = test.source.parent / "answers.pdf"
        return src if src.exists() else None

    def parse_answers(self, test, pages_markdown):
        """Pull this round's answers out of the shared answer-key document.

        Pages are selected by the round header ("Sprint Round", ...) appearing
        anywhere in their OCR markdown; the title page carries no round header
        and drops out on its own. Entries are ``N. ____ answer`` blank lines,
        several per OCR'd line when the key is laid out in columns; unit words
        printed under a blank sit on their own line and are ignored. First
        occurrence of a number wins.
        """
        header = _ANSWER_ROUND_HEADERS[test.source.stem]
        
        # Compile regexes for all potential round headers to segment pages
        # that compile multiple rounds onto a single sheet (e.g. 2001 school).
        all_headers = list(set(_ANSWER_ROUND_HEADERS.values()))
        header_patterns = [
            (h, re.compile(re.escape(h).replace(r"\ ", r"\s+"), re.IGNORECASE))
            for h in all_headers
        ]

        pages = [
            md for md in pages_markdown if header in re.sub(r"\s+", " ", md).lower()
        ]
        answers = {}
        upper_bound = 80 if test.source.stem in {"countdown", "cdr"} else {
            "sprint": 30,
            "target": 8,
            "team": 10,
        }.get(test.source.stem, 80)
        for markdown in pages:
            # Find all round header occurrences on this page
            matches = []
            for h, pattern in header_patterns:
                for match in pattern.finditer(markdown):
                    matches.append((match.start(), match.end(), h))
            matches.sort()

            # Find the match for our target header. If found, we restrict the
            # text to run from our header until the next different round header.
            target_match_idx = -1
            for idx, (_, _, h) in enumerate(matches):
                if h == header:
                    target_match_idx = idx
                    break

            if target_match_idx != -1:
                start_idx = matches[target_match_idx][0]
                end_idx = len(markdown)
                for idx in range(target_match_idx + 1, len(matches)):
                    _, _, next_h = matches[idx]
                    if next_h != header:
                        end_idx = matches[idx][0]
                        break
                section = markdown[start_idx:end_idx]
            else:
                section = markdown

            for line in section.splitlines():
                _, pairs = numbered_answers_in_line(line)
                for n, answer in pairs:
                    if answer:
                        answers.setdefault(n, answer)
            # Keep explicit labels authoritative.  Recover only cells whose
            # place in a numbered column makes the lost label unambiguous.
            for n, answer in _table_answer_recoveries(section, upper_bound=upper_bound).items():
                if answer:
                    answers.setdefault(n, answer)
        for n, answer in _VERIFIED_MISSING_ANSWERS.get(test.id, {}).items():
            answers.setdefault(n, answer)
        return answers

    def postprocess(self, problems):
        """Drop the leaked answer-blank unit from each problem's first text element."""
        for problem in problems:
            for element in problem.elements:
                if element.kind != "text":
                    continue
                text = _LEADING_BLANK_RE.sub("", element.text)
                element.text = _LEADING_UNIT_RE.sub("", text)
                break
        return problems
