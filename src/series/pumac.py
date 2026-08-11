"""PUMaC (Princeton University Math Competition): per-round test/solution PDFs.

On-disk layout (data dir is ``PUMaC/out``)::

    out/<year>/<A|B>/<subject>/test.pdf        # subject rounds
    out/<year>/<A|B>/<subject>/solutions.pdf
    out/<year>/team/test.pdf                    # team round
    out/<year>/team/solutions.pdf

The subject rounds (``algebra``, ``combinatorics``, ``geometry``,
``number_theory``, ``individual_finals``) and the ``team`` round both number
their problems plainly (``1.``, ``2.``, ...), so they fit the default pipeline.
The ``power`` and ``live`` rounds use hierarchical numbering (``Problem 1.1.1``,
``1.1 [5]``) that the integer-keyed pipeline can't represent, so `discover_tests`
deliberately skips them.

Numbering quirk: recent Team rounds print a leading, *unnumbered* ``Bonus:``
question above problem 1 (older ones numbered it ``16.``). The bonus has no
number and sits before problem 1, so `match_marker` maps ``Bonus:`` to problem
``0`` -- it sorts first (0 < 1 < ... < 15), keeping the strictly-increasing
marker guard happy, and is inert on every test that has no ``Bonus:`` label.

Answers live inside the solutions PDF, in a format that drifted over the years,
so `answer_source` reuses that document's OCR and `parse_answers` reads the
answer out of each problem's block. Four printed markers are handled
deterministically, tried in order: a modern ``**Answer:** X`` line (plain, or a
``$\\boxed{X}$`` / ``<box>X</box>`` value); the older ``(ANS: X CB: ...)`` credit
tag, when its leading token is a clean value; a bare ``\\boxed{X}`` in the prose
(distinct boxes kept -- an intended answer plus an accepted alternate, or a boxed
intermediate); and the 2009/2010 ``**Solution.** X. ...`` opener, whose first
token is the answer. Everything else -- proof problems with no closed-form
answer, and older prose that only states the answer inside a sentence -- falls
back to the answer LLM (`answer_llm.extract`).
"""

import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from typing_extensions import override

from .. import anchors, answer_llm, config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test

# Subject-round folder names (3-level: <year>/<A|B>/<subject>). Everything else
# two levels deep under a year (power, live) uses numbering we don't support.
_SUBJECTS = {"algebra", "combinatorics", "geometry", "individual_finals", "number_theory"}

# Rounds that are always proof-based ("Prove that ...") and so have no answer
# key: the individual finals. Their worked solutions are still scraped; only the
# answer path is skipped (see answer_source), sparing the OCR + LLM a document
# with no numeric answer to find -- and no hallucinated key to write.
_PROOF_ROUNDS = ("individual_finals",)
_CROSSWORD_YEARS = {"2009", "2010", "2011", "2012"}
_CROSSWORD_BASES = {
    "across": 1000,
    "down": 2000,
    "down_left": 2000,
    "down_right": 3000,
}
_2011_TEAM_GRID_ANSWERS = {
    3: "1526",       # 6 Across
    4: "6445",       # 7 Across: gray sum 64, upper-left yellow sum 45
    8: "3548",       # 12 Across
    13: "28",        # 22 Across
    14: "917842356", # 1 Down
    19: "48762",     # 10 Down
    20: "486",       # 13 Down
    24: "178",       # 18 Down
}

# A leading "Bonus:" question (Team round) -> problem 0. Only the label is
# consumed; the statement that follows becomes problem 0's text.
_BONUS_RE = re.compile(r"^\s*Bonus\b\s*:?", re.IGNORECASE)
_LETTER_MARKER_RE = re.compile(
    r"^\s*(?:\d+\s+)?[AB]\s*(\d{1,2})\b", re.IGNORECASE
)
_DIRECTION_MARKER_RE = re.compile(
    r"^\s*(\d{1,2})\s+(Across|Down)\b\s*[.:]?", re.IGNORECASE
)
_PLAIN_MARKER_RE = re.compile(r"^\s*(\d{1,2})\s*[.)]\s*")
_CROSSWORD_SECTION_RE = re.compile(
    r"(?i)\b(?:2\.[123]\s*)?(Across|Down\s+and\s+to\s+the\s+(left|right))\b"
)

# Solution-block furniture in the solutions PDF. Each problem is restated with
# its number, credited ("Proposed by ..."), then the worked solution follows a
# "Solution:" label. Only the worked solution is kept.
_SOLUTION_LABEL_RE = re.compile(
    r"^\s*[*_#>]*\s*\[?\s*[*_]*\s*Solution\b\s*[*_]*"
    r"\s*\]?\s*[*_]*\s*(?:[.:]\s*)?[*_]*\s*",
    re.IGNORECASE | re.MULTILINE,
)
_PROPOSED_RE = re.compile(r"^\**\s*Proposed\s+by\b", re.IGNORECASE)
_ANSWER_BODY_RE = re.compile(
    r"^\s*[*_#>]*\s*\[?\s*[*_]*\s*Answer\b\s*[*_]*"
    r"\s*\]?\s*[*_]*\s*[:.]?\s*.*$",
    re.IGNORECASE,
)
_RESOLUTION_LABEL_RE = re.compile(
    r"(?m)^[^\S\r\n]*(?:[*_#>\[\]]*[^\S\r\n]*)?"
    r"(?:Solution|SOLUTION|Answer|ANSWER)\b[*_]*[^\S\r\n]*[.:]|"
    r"\(\s*(?i:ANS)\s*(?:[:=]|\)\s*[:=])"
)

# --- Answer markers (see module docstring; used by parse_answers) ---
# Modern "**Answer:** X" line (emphasis optional): the value is the rest of the
# line, itself possibly a boxed/tagged span cleaned by _clean_value.
_ANSWER_LABEL_RE = re.compile(r"(?im)^\s*[*_#]{0,3}\s*Answer\b\s*[*_]{0,3}\s*[:.]\s*(.+?)\s*$")
# Older "(ANS: X ... CB: names)" credit tag: capture up to a "CB:" author list or
# the closing paren, then take only a clean leading value (_ANS_VALUE_RE); when
# the tag holds prose instead ("(ANS: Let s_n be ...)") the value match fails and
# the block is left to the LLM fallback.
_ANS_PAREN_RE = re.compile(r"\(\s*ANS\s*[:=]\s*(.*?)\s*(?:\bCB\b\s*[:=].*)?\)", re.I | re.S)
# A clean answer value at the start of an "(ANS: ...)" tag or a "**Solution.** X"
# line: a "$...$" math span, or a plain integer / simple fraction. Anything else
# (a word, a whole sentence) doesn't match, so only unambiguous values are taken
# deterministically.
_ANS_VALUE_RE = re.compile(
    r"^\s*(\$[^$]+\$|-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?:\s*%)?"
    r"(?:\s*=\s*-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?(?:\s*%)?)?)"
)
# The 2009/2010 format states the answer as the first token of the worked
# solution: "**Solution.** 455. We compute ...". Capture the text after the
# "Solution." / "Solution:" label (emphasis on either side of the punctuation);
# a clean leading value from it (_ANS_VALUE_RE) is the answer, while a label
# followed by prose ("**Solution.** We compute ...") yields no value and is left
# to the fallback. A multi-solution header ("**First Solution:**") doesn't start
# with the label, so it never matches.
_SOLUTION_ANSWER_RE = re.compile(
    r"(?im)^[^\S\r\n]*[*_#>]*[^\S\r\n]*Solution\b"
    r"[*_]*[^\S\r\n]*[.:][*_]*[^\S\r\n]*(.+)$"
)
# OCR sometimes renders \boxed{X} as a <box>X</box> tag; both are answer boxes.
_BOX_TAG_RE = re.compile(r"<box>\s*(.*?)\s*</box>", re.I | re.S)


def _match_marker(text):
    """Match a PUMaC problem marker, mapping a leading ``Bonus:`` to problem 0.

    Falls back to the built-in matcher (``1.``, ``Problem 1``, ...) for every
    numbered problem, so this is a superset of the default behavior.
    """
    m = _BONUS_RE.match(text)
    if m:
        return 0, m.end()
    m = _DIRECTION_MARKER_RE.match(text)
    if m:
        base = _CROSSWORD_BASES[m.group(2).casefold()]
        # Consume only the numeric marker. Keeping "Across"/"Down" in the tail
        # preserves the printed clue identity after the internal code is later
        # flattened back to ordinary integer keys.
        return base + int(m.group(1)), m.start(2)
    m = _LETTER_MARKER_RE.match(text)
    if m:
        return int(m.group(1)), m.end()
    return anchors._match_marker(text)


def _is_crossword_test(test_id: str) -> bool:
    return test_id.endswith("_team") and test_id[:4] in _CROSSWORD_YEARS


def _marker_number(text: str):
    """Printed problem number at the start of one source/OCR text block."""
    probe = text.lstrip("*_#>- •")
    match = _LETTER_MARKER_RE.match(probe)
    if match:
        return int(match.group(1))
    match = _DIRECTION_MARKER_RE.match(probe)
    if match:
        return int(match.group(1))
    match = _BONUS_RE.match(probe)
    if match:
        return 0
    match = _PLAIN_MARKER_RE.match(probe)
    return int(match.group(1)) if match else None


def _pdf_expected_markers(path: Path, skip_page=None, crossword=False):
    """Expected problem starts per rendered page from a born-digital text layer.

    Only the next number in the document's main sequence is accepted. This
    rejects equation numbers and numbered proof cases while retaining a marker
    printed in its own tiny block. Directional crosswords deliberately opt out:
    their clue numbers restart and repeat by section, and their completeness is
    verified after series-specific rewriting instead.
    """
    if path.suffix.lower() != ".pdf" or not path.exists():
        return ()
    import pymupdf

    expected = []
    last = None
    with pymupdf.open(path) as doc:
        for page in doc:
            text = page.get_text()
            if skip_page is not None and skip_page(text):
                continue
            if crossword:
                expected.append(frozenset())
                continue
            page_numbers = set()
            for block in page.get_text("blocks"):
                number = _marker_number(block[4])
                if number is None:
                    continue
                if last is None or number == last + 1:
                    page_numbers.add(number)
                    last = number
            expected.append(frozenset(page_numbers))
    return tuple(expected)


def _pdf_resolution_label_counts(path: Path):
    if path.suffix.lower() != ".pdf" or not path.exists():
        return ()
    import pymupdf

    with pymupdf.open(path) as doc:
        return tuple(
            len(_RESOLUTION_LABEL_RE.findall(page.get_text()))
            for page in doc
        )


def _markdown_marker_numbers(markdown: str):
    return {
        number
        for line in markdown.splitlines()
        if (number := _marker_number(line)) is not None
    }


class PumacSeries(Series):
    name = "pumac"
    has_solutions = True
    has_answers = True
    split_multiple_solutions = True

    proof_test_patterns = (r"^\d{4}_[AB]_individual_finals$",)

    @override
    def test_pages(self, test, workdir):
        """Render pages while recording cheap source-text completeness metadata."""
        self._active_test_id = test.id
        self._active_source = Path(test.source)
        self._active_is_solution = self._active_source.name == "solutions.pdf"
        self._crossword_section = None
        markers = _pdf_expected_markers(
            self._active_source,
            skip_page=self.skip_page if not self._active_is_solution else None,
            crossword=_is_crossword_test(test.id),
        )
        if self._active_is_solution:
            self._expected_solution_labels = _pdf_resolution_label_counts(
                self._active_source
            )
            # Problem headings often straddle a PDF page boundary: the text
            # layer reports the heading on page N while OCR includes it with
            # the continuation on page N+1. Formal Solution/Answer label counts
            # are stable per page and catch the actual truncations, whereas
            # marker-by-page validation produces false failures here.
            self._expected_solution_starts = tuple(
                frozenset() for _ in self._expected_solution_labels
            )
        else:
            self._expected_statement_starts = markers
        return super().test_pages(test, workdir)

    @override
    def skip_page(self, text):
        """Drop instruction-only pages from the four directional team rounds."""
        test_id = getattr(self, "_active_test_id", "")
        if getattr(self, "_active_is_solution", False) or not _is_crossword_test(test_id):
            return False
        compact = " ".join(text.casefold().split())
        year = test_id[:4]
        if year == "2009":
            return "the team round" in compact and "rules 1." in compact
        if year == "2011":
            return "mathematical sudoku puzzle" in compact and "hints and tips" in compact
        if year == "2012":
            return "team round 1 instructions" in compact and "fill in the crossword" in compact
        return False

    @override
    def clean_statement_markdown(self, page_index, markdown):
        test_id = getattr(self, "_active_test_id", "")
        if test_id == "2014_B_number_theory" and page_index == 0:
            return _compact_2014_number_theory_tail(markdown)
        if not _is_crossword_test(test_id):
            return markdown
        return self._rewrite_crossword_page(markdown)

    def _rewrite_crossword_page(self, markdown):
        """Rewrite directional clue identities to monotone internal integers."""
        year = getattr(self, "_active_test_id", "")[:4]
        section = getattr(self, "_crossword_section", None)
        out = []
        for raw in markdown.splitlines():
            probe = raw.lstrip("*_#>- •")
            if re.fullmatch(
                r"(?i)(?:<page_number>\d+</page_number>\s*)?"
                r"Page\s+\d+\s+of\s+\d+",
                probe.strip(),
            ):
                continue
            if re.fullmatch(r"(?i)PUMaC\s+20\d{2}", probe.strip()):
                continue
            if raw.strip() == "---":
                continue
            section_match = _CROSSWORD_SECTION_RE.search(probe)
            if section_match:
                label = section_match.group(1).casefold()
                if label == "across":
                    section = "across"
                elif section_match.group(2) is None:
                    section = "down"
                elif section_match.group(2).casefold() == "left":
                    section = "down_left"
                else:
                    section = "down_right"
                if year == "2012":
                    label = {
                        "across": "Across",
                        "down_left": "Down-left",
                        "down_right": "Down-right",
                    }[section]
                    out.append(f"## {label}")
                    continue

            direction = _DIRECTION_MARKER_RE.match(probe)
            if direction:
                name = direction.group(2).casefold()
                clue = int(direction.group(1))
                tail = probe[direction.end():].lstrip("*_ .:-")
                if not tail:
                    out.append(f"## {name.title()}")
                    section = name
                else:
                    code = _CROSSWORD_BASES[name] + clue
                    out.append(f"{code}. [{clue} {name.title()}] {tail}")
                continue

            # The 2012 triangular crossword prints bare N. markers under three
            # directional section headings. State carries across page breaks;
            # page 4 begins by continuing the prior section before introducing
            # the final direction.
            if year == "2012" and section is not None:
                marker = _PLAIN_MARKER_RE.match(probe)
                if marker:
                    clue = int(marker.group(1))
                    tail = probe[marker.end():].lstrip("*_ ")
                    label = {
                        "across": "Across",
                        "down_left": "Down-left",
                        "down_right": "Down-right",
                    }[section]
                    out.append(
                        f"{_CROSSWORD_BASES[section] + clue}. "
                        f"[{clue} {label}] {tail}"
                    )
                    continue
            out.append(raw)
        self._crossword_section = section
        return "\n".join(out)

    @override
    def validate_statement_markdown(self, page_index, markdown):
        if (
            getattr(self, "_active_test_id", "") == "2014_B_number_theory"
            and page_index == 0
        ):
            markdown = _compact_2014_number_theory_tail(markdown)
        expected = getattr(self, "_expected_statement_starts", ())
        if page_index >= len(expected) or not expected[page_index]:
            return True
        return expected[page_index].issubset(_markdown_marker_numbers(markdown))

    @override
    def validate_solution_markdown(self, page_index, markdown):
        if (
            getattr(self, "_active_test_id", "") == "2014_B_number_theory"
            and page_index == 0
        ):
            markdown = _compact_2014_number_theory_solution(markdown)
        expected = getattr(self, "_expected_solution_starts", ())
        labels = getattr(self, "_expected_solution_labels", ())
        markers_ok = (
            page_index >= len(expected)
            or not expected[page_index]
            or expected[page_index].issubset(_markdown_marker_numbers(markdown))
        )
        labels_ok = (
            page_index >= len(labels)
            or labels[page_index] == 0
            or len(_RESOLUTION_LABEL_RE.findall(markdown)) >= labels[page_index]
        )
        return markers_ok and labels_ok

    @override
    def discover_tests(self, data_dir):
        """Subject rounds (``<year>/<A|B>/<subject>``) and ``<year>/team``.

        The ``power`` and ``live`` rounds are skipped -- their hierarchical
        numbering doesn't fit the integer-keyed pipeline.
        """
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for pdf in sorted(root.glob("*/*/*/test.pdf")):
            if pdf.parent.name not in _SUBJECTS:
                continue
            year, division, subject = pdf.parent.parent.parent.name, pdf.parent.parent.name, pdf.parent.name
            tests.append(Test(id=f"{year}_{division}_{subject}", source=pdf))
        for pdf in sorted(root.glob("*/team/test.pdf")):
            tests.append(Test(id=f"{pdf.parent.parent.name}_team", source=pdf))
        return tests

    @override
    def match_marker(self):
        return _match_marker

    @override
    def layout_options(self):
        """Drop the Princeton shield logo (and stylized title banner) DETR reads
        as a Picture in the running header of every page.

        Both sit in the top ~9% of the page (logo center ~0.076, title banner
        ~0.086 of page height); the first real content -- the problem-1 marker --
        is at ~0.17, and a statement figure is always lower still. A 0.12 cutoff
        clears the header furniture without reaching any real figure. Without
        this, the logo binds to problem 1, because the title text above it is a
        left-margin "start" that defeats the drop-above-first-problem guard (see
        pipeline._assign_pictures)."""
        crossword = _is_crossword_test(getattr(self, "_active_test_id", ""))
        return config.LayoutOptions(
            header_picture_frac=0.15,
            min_picture_height_frac=0.02,
            equation_text_overlap=0.80,
            equation_picture_min_aspect=0.0,
            solution_equation_text_overlap=True,
            solution_answer_box_filter=True,
            strict_section_restarts=True,
            flat_problem_numbering=True,
            consecutive_problem_markers=not crossword,
        )

    @override
    def postprocess(self, problems):
        """Flatten encoded crossword clue identities into ordinary integer keys."""
        if not _is_crossword_test(getattr(self, "_active_test_id", "")):
            return problems
        ordered = sorted(
            (problem for problem in problems if problem.number >= 1000),
            key=lambda problem: problem.number,
        )
        for number, problem in enumerate(ordered, start=1):
            problem.number = number
        return ordered

    @override
    def duplicate_scope(self, test_id, across=False):
        """Bucket by year. PUMaC reuses problems across a year's A/B divisions
        (and, less often, its subject rounds), so every ``<year>_*`` test --
        ``2018_A_algebra``, ``2018_B_algebra``, ``2018_team``, ... -- is compared
        together, but never against a different year's tests.

        With ``across=True`` (``dedup --across-years``) the year is dropped and
        every test shares one bucket, catching problems recycled across years."""
        if across:
            return "all"
        return test_id.split("_", 1)[0]

    @override
    def solution_source(self, test):
        """The fixed-name sibling ``solutions.pdf``, or None if absent.

        Roughly a fifth of the tests (notably every 2024 round) ship without a
        solutions PDF; returning None lets the pipeline skip them cleanly.
        """
        sol = test.source.parent / "solutions.pdf"
        return sol if sol.exists() else None

    @override
    def clean_solution_markdown(self, page_index, markdown):
        test_id = getattr(self, "_active_test_id", "")
        if test_id == "2014_B_number_theory" and page_index == 0:
            return _compact_2014_number_theory_solution(markdown)
        if not _is_crossword_test(test_id):
            return markdown
        if test_id.startswith("2012_") and page_index == 0:
            return ""
        return self._rewrite_crossword_page(markdown)

    @override
    def parse_solutions(self, full_text, test=None):
        """Segment the solutions document into {problem_number: solution_text}.

        The solutions PDF restates each problem (with its number), credits it
        ("Proposed by ..."), then gives the worked solution under a "Solution:"
        label. We split on the same problem markers the pipeline uses for figure
        assignment, then keep only the text from "Solution:" onward -- dropping
        the restated statement and the proposer credit so the output is the
        solution, not a duplicate of problems.json. When a block has no
        recognizable "Solution:" label (unexpected layout / OCR miss), the whole
        block minus the credit line is kept so nothing is lost. Statement-figure
        crops still bind to the problem geometrically (see
        pipeline.process_solution_document); only the text is trimmed.
        """
        if test is not None and _is_crossword_test(test.id):
            full_text = _rewrite_crossword_full_text(full_text, test.id[:4])
        blocks = _group_blocks(
            full_text,
            self.match_marker(),
            consecutive=not (test is not None and _is_crossword_test(test.id)),
        )
        if test is not None and test.id.startswith("2008_"):
            blocks = _remap_2008_blocks(test, blocks)
        bodies = {}
        for number, block in blocks.items():
            body = _solution_body(block)
            if body:
                bodies[number] = body
        if self.split_multiple_solutions:
            solutions = {}
            for n, text in bodies.items():
                if text and text.strip():
                    chunks = self.split_solution_block(text)
                    solutions[n] = chunks if len(chunks) > 1 else chunks[0]
        else:
            solutions = bodies
        if test is not None and _is_crossword_test(test.id):
            solutions = _flatten_crossword_mapping(solutions)
        return solutions

    @override
    def postprocess_solutions(self, solutions, statements, test=None):
        """Remove any residual verbatim statement prefix missed by label parsing."""
        cleaned = {}
        for number, value in solutions.items():
            statement = statements.get(str(number), "")
            if isinstance(value, list):
                cleaned[number] = [_strip_statement_prefix(v, statement) for v in value]
            else:
                cleaned[number] = _strip_statement_prefix(value, statement)
        return cleaned


    @override
    def postprocess_solution_figures(self, figures, test=None, full_text=""):
        if test is None:
            return figures
        blocks = _group_blocks(
            full_text,
            self.match_marker(),
            consecutive=not _is_crossword_test(test.id),
        )
        if test.id.startswith("2008_"):
            assignment = _2008_assignment(test, blocks)
            return {
                local: figures[pool]
                for local, pool in assignment.items()
                if pool in figures
            }
        if _is_crossword_test(test.id):
            order = sorted(number for number in blocks if number >= 1000)
            indices = {
                number: index for index, number in enumerate(order, start=1)
            }
            return {
                indices[number]: crops
                for number, crops in figures.items()
                if number in indices
            }
        return figures

    @override
    def answer_source(self, test):
        """The solutions PDF (answers live inside it) -- or None for a proof round.

        The individual finals are all "Prove that ..." problems with no numeric
        answer, so there is no key to extract: returning None makes the pipeline
        skip the answer path entirely (no OCR, no LLM) while still scraping the
        round's worked solutions.
        """
        if test.id.endswith(_PROOF_ROUNDS):
            return None
        return self.solution_source(test)

    @override
    def parse_answers(self, test, pages_markdown):
        """Read each problem's final answer from the solutions document.

        The pages are grouped into per-problem blocks by the same markers the
        solution/figure passes use, then each block's answer is taken from its
        printed marker (`_extract_answer`); a block with no recognizable marker
        -- pre-2010 prose that just states the answer in a sentence, or an
        "(ANS: ...)" tag whose value is prose -- falls back to the answer LLM.
        Problems the LLM also can't resolve are omitted (a partial key, never a
        guessed one).
        """
        full_text = "\n".join(
            self.clean_solution_markdown(index, markdown)
            for index, markdown in enumerate(pages_markdown)
        )
        if _is_crossword_test(test.id):
            full_text = _rewrite_crossword_full_text(full_text, test.id[:4])
        blocks = _group_blocks(
            full_text,
            self.match_marker(),
            consecutive=not _is_crossword_test(test.id),
        )
        if test.id.startswith("2008_"):
            blocks = _remap_2008_blocks(test, blocks)
        statements = (
            _pdf_crossword_blocks(test.source, test.id[:4])
            if _is_crossword_test(test.id)
            else _pdf_problem_blocks(test.source)
        )
        answers: dict[int, str] = {}
        for n, block in blocks.items():
            statement = statements.get(n, "")
            value = _extract_answer(block) or _derived_answer(statement, block)
            if value is None and not _solution_defers_answer(block):
                context = (
                    f"Statement:\n{statement}\n\nSolution:\n{block}"
                    if statement
                    else block
                )
                value = answer_llm.extract(context)
            value = _normalize_answer_for_statement(value, statement, block)
            if value and _valid_answer(value):
                answers[n] = value
        if test.id == "2013_A_geometry":
            # The packet stops after printing
            # BD=sqrt(((2·5+4·3)(2·4+3·5))/(2·5+4·3)), without a final answer
            # line. Cancelling gives BD²=23, so the requested 13(BD)² is 299.
            answers[6] = "299"
        flattened = (
            _flatten_crossword_mapping(answers, universe=blocks)
            if _is_crossword_test(test.id)
            else answers
        )
        if test.id == "2011_team":
            # These clues are deliberately deferred in the prose solutions and
            # supplied only by the packet's final completed Sudoku/crossword.
            flattened.update(_2011_TEAM_GRID_ANSWERS)
        return flattened


def _group_blocks(full_text: str, match, consecutive=False) -> dict:
    """Group solution-document text into ``{problem_number: block_text}``.

    Reuses the nanonets layout splitter so figure crops and the solution/answer
    passes all number problems identically; figure positions are kept as a
    sentinel (harmless to the answer regexes, and needed by _solution_body's
    caller for inline figure alignment). Content before problem 1 is dropped.
    """
    grouped: dict[int, list[str]] = {}
    for item in parse_layout(
        full_text,
        match,
        strict_section_restarts=True,
        flat_problem_numbering=True,
        consecutive_problem_markers=consecutive,
    ):
        if item["problem"] is None:
            continue
        if item["kind"] == "text":
            grouped.setdefault(item["problem"], []).append(item["text"])
        elif item["kind"] == "image":
            grouped.setdefault(item["problem"], []).append(FIGURE_PLACEHOLDER)
    return {n: "\n".join(parts) for n, parts in grouped.items()}


def _rewrite_crossword_full_text(full_text: str, year: str) -> str:
    """Stateless full-document counterpart to `_rewrite_crossword_page`."""
    if re.search(r"(?m)^\s*[123]\d{3}\s*[.)]\s*", full_text):
        return full_text
    section = None
    out = []
    for raw in full_text.splitlines():
        probe = raw.lstrip("*_#>- •")
        if re.fullmatch(
            r"(?i)(?:<page_number>\d+</page_number>\s*)?"
            r"Page\s+\d+\s+of\s+\d+",
            probe.strip(),
        ):
            continue
        if re.fullmatch(r"(?i)PUMaC\s+20\d{2}", probe.strip()):
            continue
        if raw.strip() == "---":
            continue
        heading = _CROSSWORD_SECTION_RE.search(probe)
        if heading:
            label = heading.group(1).casefold()
            if label == "across":
                section = "across"
            elif heading.group(2) is None:
                section = "down"
            elif heading.group(2).casefold() == "left":
                section = "down_left"
            else:
                section = "down_right"
            if year == "2012":
                label = {
                    "across": "Across",
                    "down_left": "Down-left",
                    "down_right": "Down-right",
                }[section]
                out.append(f"## {label}")
                continue
        direction = _DIRECTION_MARKER_RE.match(probe)
        if direction:
            name = direction.group(2).casefold()
            clue = int(direction.group(1))
            tail = probe[direction.end():].lstrip("*_ .:-")
            if not tail:
                out.append(f"## {name.title()}")
                section = name
            else:
                out.append(
                    f"{_CROSSWORD_BASES[name] + clue}. "
                    f"[{clue} {name.title()}] {tail}"
                )
            continue
        if year == "2012" and section in {"across", "down_left", "down_right"}:
            marker = _PLAIN_MARKER_RE.match(probe)
            if marker:
                clue = int(marker.group(1))
                tail = probe[marker.end():].lstrip("*_ ")
                label = {
                    "across": "Across",
                    "down_left": "Down-left",
                    "down_right": "Down-right",
                }[section]
                out.append(
                    f"{_CROSSWORD_BASES[section] + clue}. "
                    f"[{clue} {label}] {tail}"
                )
                continue
        out.append(raw)
    return "\n".join(out)


def _compact_2014_number_theory_tail(markdown: str) -> str:
    """Compact the 1,007-term printed list and restore its short page tail.

    Exact OCR of the expanded fraction list exhausts the model's output budget
    before problems 5–8. The notation below is mathematically identical to the
    printed sequence and keeps the page representable as structured text.
    """
    marker = re.search(
        r"(?ms)^\s*4\.\s*\[4\]\s*Find the number of fractions", markdown
    )
    if marker is None:
        return markdown
    tail = r"""4. [4] Find the number of fractions
$$\left\{\frac{k}{2015-k}: 1 \leq k \leq 1007\right\}$$
that are in lowest form.

5. [5] Find the sum of all positive integers $x$ such that
$3\cdot 2^x=n^2-1$ for some positive integer $n$.

6. [6] Let $S=\{2,5,8,11,14,17,20,\ldots\}$. Suppose one can choose
$n$ distinct numbers $A_1,\ldots,A_n$ from $S$ such that
$$\sum_{i=1}^{n}\frac{1}{A_i}=1.$$
Find the minimum possible value of $n$.

7. [7] How many permutations $p$ of $\{1,2,3,\ldots,35\}$ satisfy
$a\mid b\Longrightarrow p(a)\mid p(b)$?

8. [8] Find the number of positive integers $n\leq 2014$ for which there
exists an integer $x$ such that
$$\frac{x+n}{x-n}$$
is an odd perfect square."""
    return markdown[:marker.start()].rstrip() + "\n\n" + tail


def _compact_2014_number_theory_solution(markdown: str) -> str:
    """Compact the corresponding long problem-4 solution-page tail."""
    marker = re.search(
        r"(?ms)^\s*4\.\s*\[4\]\s*Find the number of fractions", markdown
    )
    if marker is None:
        return markdown
    tail = r"""4. [4] Find the number of fractions
$$\left\{\frac{k}{2015-k}: 1 \leq k \leq 1007\right\}$$
that are in lowest form.

**Solution:** A fraction fails to be in lowest form exactly when its
denominator is divisible by at least one prime factor of
$2015=5\cdot13\cdot31$. Inclusion-exclusion gives
$$
\left\lfloor\frac{1007}{5}\right\rfloor+
\left\lfloor\frac{1007}{13}\right\rfloor+
\left\lfloor\frac{1007}{31}\right\rfloor-
\left\lfloor\frac{1007}{65}\right\rfloor-
\left\lfloor\frac{1007}{155}\right\rfloor-
\left\lfloor\frac{1007}{403}\right\rfloor
=287.
$$
Hence $1007-287=\boxed{720}$ fractions are in lowest form."""
    return markdown[:marker.start()].rstrip() + "\n\n" + tail


def _flatten_crossword_mapping(values, universe=None):
    """Map directional clue codes to stable sequential output keys.

    ``universe`` supplies the complete clue set when ``values`` is partial
    (answers frequently are). Enumerating only successful extractions would
    shift every later answer onto the wrong problem whenever one clue has no
    extractable value.
    """
    source = universe if universe is not None else values
    order = sorted(number for number in source if number >= 1000)
    indices = {number: index for index, number in enumerate(order, start=1)}
    return {
        indices[number]: value
        for number, value in values.items()
        if number in indices
    }


def _pdf_problem_blocks(path: Path):
    """Main numbered statement blocks from a test PDF's text layer."""
    if path is None or not Path(path).exists():
        return {}
    import pymupdf

    with pymupdf.open(path) as doc:
        text = "\n".join(page.get_text() for page in doc)
    matches = list(re.finditer(r"(?m)^\s*(\d{1,2})\s*[.)]\s+", text))
    accepted = []
    last = 0
    for match in matches:
        number = int(match.group(1))
        if number == last + 1:
            accepted.append(match)
            last = number
    blocks = {}
    for index, match in enumerate(accepted):
        end = accepted[index + 1].start() if index + 1 < len(accepted) else len(text)
        blocks[int(match.group(1))] = text[match.end():end].strip()
    return blocks


def _pdf_crossword_blocks(path: Path, year: str):
    """Directional statement blocks keyed by the same encoded clue ids as OCR."""
    if path is None or not Path(path).exists():
        return {}
    import pymupdf

    with pymupdf.open(path) as doc:
        text = "\n".join(page.get_text() for page in doc)
    rewritten = _rewrite_crossword_full_text(text, year)
    return _group_blocks(rewritten, _match_marker, consecutive=False)


def _statement_part(block: str):
    cuts = [
        match.start()
        for pattern in (_ANS_PAREN_RE, _ANSWER_LABEL_RE, _SOLUTION_LABEL_RE)
        if (match := pattern.search(block)) is not None
    ]
    return block[: min(cuts)] if cuts else block


def _match_score(left: str, right: str):
    left_words = re.findall(r"[a-z0-9]+", left.casefold())
    right_words = re.findall(r"[a-z0-9]+", right.casefold())
    if not left_words or not right_words:
        return 0.0
    left_text = " ".join(left_words[:180])
    right_text = " ".join(right_words[:180])
    sequence = SequenceMatcher(None, left_text, right_text).ratio()
    left_set, right_set = set(left_words), set(right_words)
    jaccard = len(left_set & right_set) / max(len(left_set | right_set), 1)
    return 0.7 * sequence + 0.3 * jaccard


def _remap_2008_blocks(test, blocks):
    """Map a shared 2008 solution-pool block back to this test's local number."""
    assignment = _2008_assignment(test, blocks)
    return {
        local: blocks[pool]
        for local, pool in assignment.items()
    }


def _2008_assignment(test, blocks):
    """Return ``{local_problem: shared_pool_problem}`` for a 2008 packet."""
    statements = _pdf_problem_blocks(test.source)
    if not statements or not blocks:
        return {}
    candidates = {
        pool_number: _statement_part(block)
        for pool_number, block in blocks.items()
    }
    local_numbers = tuple(statements)
    pool_numbers = tuple(candidates)
    scores = tuple(
        tuple(
            _match_score(statements[local], candidates[pool])
            for pool in pool_numbers
        )
        for local in local_numbers
    )

    @lru_cache(maxsize=None)
    def assign(local_index, used_mask):
        if local_index == len(local_numbers):
            return 0.0, ()
        best_score, best_assignment = float("-inf"), ()
        for pool_index in range(len(pool_numbers)):
            bit = 1 << pool_index
            if used_mask & bit:
                continue
            tail_score, tail_assignment = assign(
                local_index + 1, used_mask | bit
            )
            total = scores[local_index][pool_index] + tail_score
            if total > best_score:
                best_score = total
                best_assignment = (pool_index,) + tail_assignment
        return best_score, best_assignment

    _, assignment = assign(0, 0)
    remapped = {}
    for local_index, pool_index in enumerate(assignment):
        # Keep a deliberately low floor: OCR of dense geometry can differ
        # substantially, while the global one-to-one assignment prevents a weak
        # block from stealing a clearly matched block from another problem.
        if scores[local_index][pool_index] >= 0.12:
            remapped[local_numbers[local_index]] = pool_numbers[pool_index]
    return remapped


def _normalized_words(text):
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def _strip_statement_prefix(solution, statement):
    if not isinstance(solution, str) or not statement:
        return solution
    statement_words = _normalized_words(statement)
    solution_words = _normalized_words(solution)
    compare = min(60, len(statement_words))
    if len(statement_words) < 12 or solution_words[:compare] != statement_words[:compare]:
        return solution
    lines = solution.splitlines()
    for index in range(1, len(lines) + 1):
        prefix_words = _normalized_words("\n".join(lines[:index]))
        if len(prefix_words) >= len(statement_words) * 0.85:
            remainder = "\n".join(lines[index:]).strip()
            return _solution_body(remainder) if remainder else solution
    return solution


def _valid_answer(value):
    text = (value or "").strip()
    if not text or "\n" in text or len(text) > 120:
        return False
    if re.search(r"\\(?:iff|implies)\b", text):
        return False
    if text.count("=") > 1:
        return False
    return True


def _solution_defers_answer(block):
    """True when the packet explicitly leaves a clue to the completed grid."""
    return bool(
        re.search(
            r"(?i)\b(?:leave (?:this|it|this clue) (?:blank|out)|"
            r"impossible to solve until|hard to solve|hard, so|"
            r"if not, you can leave|don[’']t fill)\b",
            block,
        )
    )


def _derived_answer(statement, block):
    """Extract a clearly stated final value from informal crossword solutions."""
    # Several early team packets omit an Answer/Solution label but end by
    # naming the requested tuple. If the statement asks for concatenation, a
    # numeric tuple on the right side of the final equality is authoritative.
    if re.search(r"(?i)\bconcatenat", statement or ""):
        tuples = re.findall(
            r"=\s*\(\s*(-?\d+(?:\s*,\s*-?\d+)+)\s*\)", block
        )
        if tuples:
            return "".join(re.findall(r"-?\d+", tuples[-1]))
        tuples = re.findall(
            r"(?i)\bsolution\b[^()\n]{0,40}"
            r"\(\s*(-?\d+(?:\s*,\s*-?\d+)+)\s*\)",
            block,
        )
        if tuples:
            return "".join(re.findall(r"-?\d+", tuples[-1]))

    high_confidence_patterns = (
        r"(?i)\b(?:for (?:an?|the) (?:answer|solution|result) of|"
        r"we are looking for|must be|yields?|obtain(?:s|ed)?|we get)\s*"
        r"\$?(-?\d+(?:\.\d+)?)",
        r"(?i)\b(?:therefore|hence|thus|so)\b[^.\n]{0,100}?"
        r"(?:=|is)\s*\$?(-?\d+(?:\.\d+)?)",
    )
    candidates = []
    for pattern in high_confidence_patterns:
        candidates.extend(
            (match.start(), match.group(1))
            for match in re.finditer(pattern, block)
        )
    equalities = list(
        re.finditer(r"(?:=|\\approx)\s*\$?(-?\d+(?:\.\d+)?)", block)
    )
    if not candidates:
        return equalities[-1].group(1) if equalities else None
    high_position, high_value = max(candidates)
    if not equalities or equalities[-1].start() < high_position:
        return high_value
    equality = equalities[-1]
    between = block[high_position:equality.start()]
    # "yields 6912 = 6·9·..." states the answer on the left; do not replace it
    # with the first factor on the right merely because that token is later.
    if len(between) <= 50 and re.search(
        rf"{re.escape(high_value)}\s*\$?\s*$", between
    ):
        return high_value
    return equality.group(1)


def _normalize_answer_for_statement(value, statement, block=""):
    if value is None:
        return None
    text = _clean_value(str(value))
    if re.search(
        r"(?i)^(?:no (?:solution|answer)|unknown|cannot be determined|"
        r"not (?:given|provided))$",
        text,
    ):
        return None
    if re.search(r"(?i)\bconcatenat", statement or ""):
        numbers = re.findall(r"-?\d+", text)
        if len(numbers) > 1:
            return "".join(numbers)
    statement_probe = (statement or "").replace("ﬁ", "fi")
    if text.count("=") > 1 and re.search(
        r"(?i)\b(?:find|value|evaluate|compute)\b", statement_probe
    ):
        final_phrase = re.findall(
            r"(?i)\banswer\b[^.\n]{0,40}\b(?:is|be)\s*\$?"
            r"(-?\d+(?:\.\d+)?)",
            block,
        )
        if final_phrase:
            return final_phrase[-1]
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]
    return text


def _solution_body(block: str) -> str:
    """Return just the worked solution from one restated-problem block.

    Everything up to and including the first "Solution:" label is the restated
    statement and proposer credit; drop it. If no label is present, fall back to
    the whole block with any "Proposed by ..." lines removed.
    """
    lines = block.splitlines()
    for i, line in enumerate(lines):
        m = _SOLUTION_LABEL_RE.match(line)
        if m:
            rest = line[m.end():]
            kept = ([rest] if rest.strip() else []) + lines[i + 1:]
            return _drop_solution_furniture(kept)
    # Recent packets restate the problem, print an Answer row, and immediately
    # begin the proof without a separate Solution label.
    for i, line in enumerate(lines):
        if _ANSWER_BODY_RE.match(line):
            kept = lines[i + 1:]
            if any(part.strip() for part in kept):
                return _drop_solution_furniture(kept)
    ans = re.search(
        r"\(\s*ANS\s*(?:[:=]|\)\s*[:=])\s*", block, re.IGNORECASE
    )
    if ans is not None:
        tail = block[ans.end():].strip()
        credit = re.search(r"\bCB\s*[:=]", tail, re.IGNORECASE)
        if credit is not None:
            tail = tail[:credit.start()].rstrip()
        leading = _ANS_VALUE_RE.match(tail)
        if leading is not None:
            tail = tail[leading.end():].lstrip(" .:;)-")
        tail = tail.rstrip().rstrip(")").rstrip()
        if tail:
            return tail
        answer = _extract_answer(block)
        return f"Answer: {answer}" if answer else ""
    return "\n".join(l for l in lines if not _PROPOSED_RE.match(l)).strip()


def _drop_solution_furniture(lines):
    return "\n".join(
        line
        for line in lines
        if not _PROPOSED_RE.match(line)
        and "tinyurl.com/PUMaC" not in line
        and "tinyurl.com/PUMAC" not in line
    ).strip()


def _extract_answer(block: str):
    """Return the answer printed in one problem's block, or None.

    Markers are tried most-authoritative first: the modern "**Answer:** X" line;
    then an "(ANS: X ...)" credit tag with a clean leading value; then any
    ``\\boxed{}`` / ``<box>`` in the prose; then the 2009/2010 "**Solution.** X."
    opener. Returns None when no marker yields a value, leaving the block for the
    LLM fallback. See the module docstring.
    """
    m = _ANSWER_LABEL_RE.search(block)
    if m:
        value = _clean_value(m.group(1))
        if value:
            return value
    payload = _ans_payload(block)
    if payload is not None:
        value = _leading_value(payload)
        if value:
            return value
        # Credit-tagged payloads end exactly at "CB:" and are therefore safe
        # to keep even when the answer is prose ("decreases by 9%") or a
        # coordinate pair whose parentheses defeated the old regex.
        if len(payload) <= 100:
            value = _clean_value(payload)
            if value:
                return value
    m = _ANS_PAREN_RE.search(block)
    if m:
        value = _leading_value(m.group(1))
        if value:
            return value
    for m in _SOLUTION_ANSWER_RE.finditer(block):
        value = _leading_value(m.group(1).lstrip("*_ "))
        if value:
            return value
    bracketed = re.findall(
        r"(?i)\b(?:there (?:are|is)|answer is|total of)\s*"
        r"\[\s*(-?\d+(?:\s*/\s*\d+)?)\s*\]",
        block,
    )
    if bracketed:
        return _clean_value(bracketed[-1])
    boxes = _distinct_boxes(block)
    if boxes:
        if re.search(
            r"(?i)(?:also accepted|accepted (?:answer|solution)|unintended alternate)",
            block,
        ):
            return ", ".join(boxes[-2:])
        # Worked solutions frequently box an intermediate quantity before the
        # requested transformed value. The final distinct box is authoritative.
        return boxes[-1]
    return None


def _ans_payload(block: str):
    marker = re.search(
        r"\(\s*ANS\s*(?:[:=]|\)\s*[:=])\s*", block, re.IGNORECASE
    )
    if marker is None:
        return None
    tail = block[marker.end():]
    credit = re.search(r"\bCB\s*[:=]", tail, re.IGNORECASE)
    if credit is not None:
        return tail[:credit.start()].strip().rstrip(") ").strip()
    # With no credit delimiter, keep just an initial math span/value. The rest
    # of the line is often the full worked solution.
    leading = _ANS_VALUE_RE.match(tail)
    return leading.group(1) if leading else None


def _leading_value(text: str):
    """Clean answer value at the start of `text` (a number/fraction or "$...$"
    span), or None when it opens with prose. Used for the "(ANS: ...)" tag and
    the "**Solution.** X" opener, which both put the answer first, then prose."""
    m = _ANS_VALUE_RE.match(text)
    return _clean_value(m.group(1)) if m else None


def _iter_boxes(text: str):
    r"""Yield each answer box's inner text: ``\boxed{...}`` (brace-balanced, so a
    nested ``\frac{a}{b}`` survives) and each ``<box>...</box>`` tag."""
    i = 0
    while True:
        j = text.find(r"\boxed", i)
        if j == -1:
            break
        open_brace = text.find("{", j)
        if open_brace == -1:
            break
        depth, k = 0, open_brace
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    yield text[open_brace + 1:k]
                    break
            k += 1
        i = k + 1
    yield from _BOX_TAG_RE.findall(text)


def _distinct_boxes(text: str):
    """Boxed values in order, de-duplicated (a restated box isn't a new answer)."""
    out = []
    for box in _iter_boxes(text):
        box = box.strip().strip("$").strip()
        if box and box not in out:
            out.append(box)
    return out


def _clean_value(raw: str):
    """Normalize a marker's raw value to a bare answer string, or None.

    Unwraps a ``\\boxed{}`` / ``<box>`` value (taking the last, the final answer),
    then strips LaTeX ``$`` delimiters, emphasis, and a trailing period.
    """
    boxes = _distinct_boxes(raw)
    value = boxes[-1] if boxes else raw
    value = value.strip().strip("$").strip()
    value = re.sub(r"^\*+|\*+$", "", value).strip().rstrip(".").strip()
    return value or None
