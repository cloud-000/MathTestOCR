"""Carnegie Mellon Informatics and Mathematics Competition.

On-disk layout (data dir may be ``CMIMC`` or ``CMIMC/out``)::

    out/<year>/<division>/<subject>/test.pdf
    out/<year>/<division>/<subject>/solutions.pdf

Test IDs mirror the path below ``out``, joined by underscores, such as
``2025_individual_algebra`` and ``2022_team_team``.  The older power round is
omitted because its hierarchical proof packet does not have a stable flat
problem numbering scheme; the newer three-problem computer-science proof round
is flat and remains discoverable.

Layout quirks handled here:

* Every page carries the CMIMC wordmark in its running header.  DETR reads it
  as a Picture and, sitting above the page's first problem, it would be filed
  as a figure of whichever problem continued from the previous page --
  ``header_picture_frac`` fences it off.
* The theoretical computer-science round *titles* its proof problems
  ("# Balance the Board") and prints no number at all
  (``heading_problem_markers``); the 2018 team round numbers a relay
  ``1-1.``/``6-2.`` (see ``_RELAY_RE``).
* Display equations -- a stacked fraction after "of", a boxed final answer, a
  summation ending a sentence -- are read as Pictures and would be written out
  as figure crops.  They are roughly square and no shorter than the genuine wide
  strip figures, so neither the aspect nor the height guard separates them;
  ``text_layer_equation_coverage`` does, from the born-digital glyphs.
* Instruction pages, and the solutions packet that 2017's number-theory
  ``test.pdf`` bundles after the round itself, are dropped by ``skip_page``
  before any OCR.
* Whole-page OCR occasionally returns an HTML document (```` ```html ````,
  ``<p>6. ...</p>``) instead of markdown, hiding every marker on the page --
  ``_clean_cmimc_markdown`` unwraps it.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path

from typing_extensions import override

from .. import anchors, answer_llm, config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import CoverageException, Series, Test
from .smt import _boxed_answer

_SOLUTION_LINE_RE = re.compile(
    r"^\s*(?:[*_#]{0,3}\s*)?Solutions?(?:\s+\d+)?\s*[*_]{0,2}\s*[.:]?\s*(.*)$",
    re.I,
)
_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:[*_#]{0,3}\s*)?Answer\s*[*_]{0,2}\s*[.:]?\s*(.*)$", re.I
)
_PROPOSER_LINE_RE = re.compile(r"^\s*[*_]{0,2}\s*Proposed by\b", re.I)
_THE_ANSWER_IS_RE = re.compile(r"\b(?:the\s+)?answer\s+is\s+(.+?)(?:[.!]\s|$)", re.I)

# CMIMC puts its numbered instructions on the first problem page.  They are
# intentionally filtered here rather than with skip_page, which would also
# discard the first several real problems.
_RULE_START_RE = re.compile(
    r"^(?:"
    r"do not look|"
    r"this test consists|"
    r"write (?:your|answers|legibly)|"
    r"no computational aids|"
    r"all (?:the )?answers (?:are|must|should)|"
    r"answers (?:are|must|should)|"
    r"if you believe|"
    r"in your solution|"
    r"problems are not ordered|"
    r"you have \d+ minutes|"
    r"during the test|"
    r"submit your answers|"
    r"we use the following notation|"
    r"the following notation may be useful|"
    r"in the event of a tie"
    r")\b",
    re.I,
)

# The 2018 team round is a relay, numbered "<pair>-<leg>." over two sets of ten
# problems. Each set prints one leg of every pair: set one carries pairs 1-5 leg
# 1 and pairs 6-10 leg 2, set two the complementary halves. So the pair alone
# does not identify a problem -- both components do, and _relay_number folds
# them into the flat 1-20 the rest of the pipeline works in.
_RELAY_RE = re.compile(r"^\s*(\d{1,2})-([12])\s*\.")
_RELAY_MARKER_RE = re.compile(
    r"(?m)^[ \t]*(?:[*_#]{0,3}[ \t]*)?(\d{1,2})-([12])[ \t]*\."
)
_RELAY_PAIRS_PER_SET = 5
_RELAY_SET_SIZE = 10

# Shape of an in-statement numbered list, for _demote_list_markers. A worked
# solution is never three consecutive one-short-line blocks; an algorithm's
# steps always are.
_MIN_LIST_RUN = 3
_MAX_LIST_ITEM_LEN = 160
# Word-level similarity at which a "solution" is really the statement reprinted.
# High enough that a solution quoting its problem back before working it is kept.
_RESTATEMENT_SIMILARITY = 0.85


def _relay_number(pair: int, leg: int) -> int:
    """Flatten a relay "<pair>-<leg>" label to a 1-20 problem number."""
    in_first_set = (pair <= _RELAY_PAIRS_PER_SET) == (leg == 1)
    return pair + (0 if in_first_set else _RELAY_SET_SIZE)


# The whole-page OCR sometimes returns an HTML document rather than markdown.
# Unwrapping it matters more than cosmetics: a marker inside "<p>6. Let ...</p>"
# is invisible to parse_layout, so every problem below it folds into the one
# above (2019 individual algebra lost solutions 6 and 7 this way).
_FENCE_RE = re.compile(r"^\s*```(?:html)?\s*$", re.IGNORECASE | re.MULTILINE)
_BLOCK_TAG_RE = re.compile(
    r"</?(?:html|body|head|main|section|article|p|div|h[1-6])\b[^>]*>",
    re.IGNORECASE,
)
_INLINE_TAG_RE = re.compile(
    r"</?(?:strong|b|em|i|span|center|font)\b[^>]*>", re.IGNORECASE
)
# The running-header wordmark, when OCR transcribes it as a heading instead of
# an <img>. Nanonets reads the stylized logo inconsistently (CMIMC, CMMO, CIMD,
# CMIMD, CIMMO, CIM, ...), so match the shape rather than a fixed spelling: a
# line that is nothing but a short all-caps C-word and a year.
_BANNER_RE = re.compile(
    r"(?im)^[ \t]*(?:[#*_]+[ \t]*)?C[A-Z]{1,5}[ \t]+(?:20\d\d)[ \t]*(?:[#*_]+)?[ \t]*$"
)
_BLANK_LINES_RE = re.compile(r"\n{3,}")
# A row of small figures is sometimes transcribed as one <img> wrapping several
# others. parse_layout pairs tags non-greedily, so the wrapper's own closer is
# left dangling as literal "</img>" text in the statement. Drop the wrapper and
# any closer it orphaned, keeping the inner tags as the figures they describe.
_IMG_WRAPPER_RE = re.compile(r"<img>\s*(?=<img>)", re.IGNORECASE)
_IMG_TAG_RE = re.compile(r"(</?img>)", re.IGNORECASE)

# skip_page: instruction pages, and the worked-solution pages that 2017's
# number-theory test.pdf appends after the round itself.
_INSTRUCTION_PAGE_RE = re.compile(r"do not look at the test before the proctor", re.I)
_SOLUTION_PAGE_RE = re.compile(r"Proposed by|Solutions?\s+Packet", re.I)

# Rounds whose problems are titled rather than numbered (see
# LayoutOptions.heading_problem_markers).
_TITLED_PROOF_ROUND = "team_computer-science"


def _match_marker(text: str):
    relay = _RELAY_RE.match(text)
    if relay is not None:
        return _relay_number(int(relay.group(1)), int(relay.group(2))), relay.end()
    result = anchors._match_marker(text)
    if result is None:
        return None
    if result[0] > 100:
        return None
    if _RULE_START_RE.match(text[result[1] :].strip()):
        return None
    return result


def _clean_cmimc_markdown(markdown: str) -> str:
    """Unwrap OCR HTML scaffolding and drop the running-header wordmark."""
    text = _FENCE_RE.sub("", markdown)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _INLINE_TAG_RE.sub("", text)
    text = _BANNER_RE.sub("", text)
    text = _unwrap_image_groups(text)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def _unwrap_image_groups(text: str) -> str:
    """Drop an <img> that only wraps other <img> tags, and orphaned closers."""
    text = _IMG_WRAPPER_RE.sub("", text)
    parts = []
    depth = 0
    for token in _IMG_TAG_RE.split(text):
        lowered = token.lower()
        if lowered == "<img>":
            depth += 1
        elif lowered == "</img>":
            if depth == 0:
                continue
            depth -= 1
        parts.append(token)
    return "".join(parts)


class CmimcSeries(Series):
    name = "cmimc"
    has_solutions = True
    split_multiple_solutions = True

    has_answers = True
    proof_test_patterns = (r"^\d+_team_computer-science$",)
    ignored_test_substrings = ("mathdash",)

    @override
    def coverage_exceptions(self, test_id: str) -> dict[int, CoverageException]:
        """Record source-verified non-standard questions without dropping them.

        The 2017 packet appends a three-question tiebreaker after the ten
        Number Theory round problems; its official solutions packet covers the
        main round only.  The 2020 Team packet's last item is an interactive
        estimation activity, so it has no canonical answer at all.
        """
        if test_id == "2017_individual_number-theory":
            return {
                number: CoverageException(
                    answer_status="source_missing",
                    solution_status="source_missing",
                    reason="Number Theory tiebreaker omitted from the official packet",
                )
                for number in (11, 12, 13)
            }
        if test_id == "2020_team_team":
            return {
                16: CoverageException(
                    answer_status="not_applicable",
                    solution_status="not_applicable",
                    reason="Interactive estimation activity has no canonical answer",
                )
            }
        return {}

    @override
    def discover_tests(self, data_dir):
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        search_root = root / "out" if (root / "out").is_dir() else root
        return [
            Test(
                id="_".join(pdf.relative_to(search_root).parts[:-1]),
                source=pdf,
            )
            for pdf in sorted(search_root.glob("**/test.pdf"))
            if pdf.parent.name != "power"
        ]

    @override
    def test_pages(self, test: Test, workdir):
        """Render pages, recording which document and test they came from.

        ``skip_page`` and ``layout_options`` both need to know whether the
        active source is a statement or a solution document (only a statement
        PDF may carry a bundled solutions packet to drop) and which round is
        being parsed (only the proof round is titled rather than numbered).
        Also record each retained page's born-digital marker count, which
        ``validate_statement_markdown`` uses to reject an OCR response that
        silently dropped a problem.
        """
        self._active_test_id = test.id
        self._active_source = Path(test.source)
        self._expected_statement_starts = []
        pages = super().test_pages(test, workdir)
        if self._active_source.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(self._active_source) as doc:
                for page in pages:
                    match = re.search(r"(\d+)$", Path(page).stem)
                    index = int(match.group(1)) - 1 if match is not None else None
                    text = doc[index].get_text() if index is not None else ""
                    self._expected_statement_starts.append(_source_marker_count(text))
        return pages

    @override
    def skip_page(self, text: str) -> bool:
        """Drop instruction pages, and a solutions packet bundled into a test.

        2017's number-theory ``test.pdf`` appends the whole worked-solution
        packet after the round; parsed as statements those pages restate every
        problem and their restarting numbers inflate the sequence (26 problems
        with five gaps). Only a statement source is filtered that way -- a
        solution document is *all* solution pages.
        """
        if not text.strip():
            return False
        if _INSTRUCTION_PAGE_RE.search(text):
            return True
        source = getattr(self, "_active_source", None)
        is_statement = source is not None and source.name.lower() == "test.pdf"
        return bool(is_statement and _SOLUTION_PAGE_RE.search(text))

    @override
    def match_marker(self):
        return _match_marker

    @override
    def layout_options(self):
        """Fence off the header wordmark; number the proof round by its titles.

        ``header_picture_frac`` drops the CMIMC wordmark printed at the top of
        every page. Across all 66 source PDFs no genuine figure has its vertical
        centre between 0.08 and 0.12 of the page height (the wordmark sits at
        0.05-0.08, the highest real diagram at 0.13), so 0.10 separates them
        with room to spare.

        ``strict_section_restarts`` keeps a statement's own numbered procedure
        ("1. Write down the number ... 4. Go back to step 1") from being read as
        a section restart, which would renumber every problem below it.

        ``heading_problem_markers`` is scoped to the theoretical computer-science
        round, the only one that titles its proof problems instead of numbering
        them; enabling it elsewhere would turn each round's title heading into a
        problem.

        ``text_layer_equation_coverage`` drops the display equations DETR reads
        as figures -- a stacked fraction after "of", a boxed final answer, a
        summation ending a sentence. They are roughly square, so the aspect guard
        behind ``equation_text_overlap`` cannot see them, and no shorter than the
        round's genuine wide strip figures (a tetromino row is 0.025 of the page
        height against the equations' 0.027-0.038), so height cannot either.
        Their source glyphs can: measured over every CMIMC figure box, the six
        equations cover 0.30-0.52 of their box with text-layer glyphs while no
        genuine figure exceeds 0.12 and nothing at all falls in between.
        """
        titled = getattr(self, "_active_test_id", "").endswith(_TITLED_PROOF_ROUND)
        return config.LayoutOptions(
            inline_figures=True,
            header_picture_frac=0.10,
            text_layer_equation_coverage=0.20,
            strict_section_restarts=True,
            heading_problem_markers=titled,
        )

    @override
    def clean_statement_markdown(self, page_index: int, markdown: str) -> str:
        return _clean_cmimc_markdown(markdown)

    @override
    def clean_solution_markdown(self, page_index: int, markdown: str) -> str:
        """Clean, then defuse a statement's own numbered list.

        The demotion also runs inside ``_group_blocks``; doing it here as well
        (it is idempotent -- a defused list is no longer a run of markers) keeps
        the figure-tagging pass in ``pipeline.process_solution_document``, which
        parses this cleaned markdown directly, on the same numbering as the
        solution text.
        """
        text = _clean_cmimc_markdown(markdown)
        return text if _is_answer_key(text) else _demote_list_markers(text)

    @override
    def validate_statement_markdown(self, page_index: int, markdown: str) -> bool:
        """Reject a clean-looking page OCR that dropped a printed problem.

        CMIMC is born-digital from 2016 on except for 2021 (scanned, no text
        layer, ``expected`` 0 -- accepted unconditionally). Comparing against
        the source's own marker count is what caught 2025 combinatorics losing
        problem 10 to a silently omitted "10." marker.
        """
        expected = getattr(self, "_expected_statement_starts", ())
        if page_index >= len(expected) or expected[page_index] == 0:
            return True
        return _marker_count(_clean_cmimc_markdown(markdown)) >= expected[page_index]

    @override
    def solution_source(self, test: Test):
        solution = test.source.parent / "solutions.pdf"
        return solution if solution.exists() else None

    @override
    def answer_source(self, test: Test):
        """The key lives inside the solution document -- except for proof rounds.

        The theoretical computer-science round is graded on written proofs and
        has no short answer to key; returning None skips the answer path (its
        solutions are still scraped) rather than letting the answer-LLM fallback
        invent one from the prose.
        """
        if test.id.endswith(_TITLED_PROOF_ROUND):
            return None
        return self.solution_source(test)

    @override
    def parse_solutions(self, full_text: str, test: Test = None) -> dict:
        if _is_answer_key(full_text):
            return {}
        bodies = {
            number: _solution_body(block)
            for number, block in _group_blocks(full_text, _is_titled(test)).items()
        }
        if self.split_multiple_solutions:
            res = {}
            for n, text in bodies.items():
                if text and text.strip():
                    chunks = self.split_solution_block(text)
                    res[n] = chunks if len(chunks) > 1 else chunks[0]
            return res
        return {number: body for number, body in bodies.items() if body}


    @override
    def postprocess_solutions(
        self, solutions: dict, statements: dict, test=None
    ) -> dict:
        """Drop a "solution" that is only the problem restated.

        ``_solution_body`` keeps a whole block when it carries none of the
        Solution / Answer / Proposed-by labels, so an unusual layout is never
        silently lost. An estimation or tiebreaker question, though, is printed
        in the packet as its statement and nothing else -- there is no worked
        solution to keep, and storing the statement again is worse than an
        absent entry. Compared against the parsed statement, so no prose
        heuristic is needed.
        """
        kept = {}
        for number, value in solutions.items():
            statement = statements.get(str(number), "")
            bodies = value if isinstance(value, list) else [value]
            bodies = [b for b in bodies if not _is_restatement(b, statement)]
            if bodies:
                kept[number] = bodies if isinstance(value, list) else bodies[0]
        return kept

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        full_text = "\n\n".join(
            self.clean_solution_markdown(index, page)
            for index, page in enumerate(pages_markdown)
        )
        answer_key = _is_answer_key(full_text)
        answers = {}
        for number, block in _group_blocks(full_text, _is_titled(test)).items():
            value = _answer_value(block)
            if not value and answer_key:
                value = _clean_value(block)
            if not value:
                # 2016-2017 solutions state the result only in prose ("... and so
                # the sum of the possible n is 5.") with no Answer line and no
                # \boxed{}. Let the text LLM read it out; it fails soft, so an
                # unreadable block is simply left out of the key.
                value = _clean_value(answer_llm.extract(block) or "")
            if value:
                answers[number] = value
        return answers


def _is_titled(test: Test = None) -> bool:
    """Whether `test` is the proof round whose problems are titled, not numbered."""
    return test is not None and str(test.id).endswith(_TITLED_PROOF_ROUND)


def _group_blocks(full_text: str, titled: bool = False) -> dict[int, str]:
    if _RELAY_MARKER_RE.search(full_text):
        return _group_relay_blocks(full_text)
    if not _is_answer_key(full_text):
        full_text = _demote_list_markers(full_text)
    grouped: dict[int, list[str]] = {}
    for item in parse_layout(
        full_text,
        _match_marker,
        strict_section_restarts=True,
        heading_problem_markers=titled,
    ):
        if item["problem"] is None:
            continue
        value = item["text"] if item["kind"] == "text" else FIGURE_PLACEHOLDER
        grouped.setdefault(item["problem"], []).append(value)
    return {number: "\n".join(parts).strip() for number, parts in grouped.items()}


def _demote_list_markers(full_text: str) -> str:
    """Join a statement's own numbered list back into the prose around it.

    The computer-science rounds print an algorithm as a numbered list inside the
    problem ("1. FUNCTION f(A)", "2. FOR i = ...", ...). Its items look exactly
    like problem markers and, being *higher* than the problem they sit in, sail
    past the strictly-increasing guard and become problems -- taking the rest of
    the document's numbering with them.

    A numbered list is told apart structurally, not by its wording: three or more
    consecutively numbered markers in a row, each owning nothing but one short
    line, and each (after the first) printed directly under its predecessor with
    no blank line between. That last condition is what separates a list from a
    run of real problems -- a problem block always opens after a blank line. The
    run is extended through the item that closes the list, whose "span" runs on
    into the following prose and so is not short itself.

    Each such marker is folded onto the preceding non-blank line, which keeps the
    printed number visible while taking it out of line-start position, where it
    would otherwise read as a problem start.
    """
    lines = full_text.splitlines()
    marks = []
    for index, line in enumerate(lines):
        match = _match_marker(line.lstrip("*_# "))
        if match is not None:
            marks.append((index, match[0]))
    demote = set()
    position = 0
    while position < len(marks):
        run = [position]
        while (
            run[-1] + 1 < len(marks)
            and marks[run[-1] + 1][1] == marks[run[-1]][1] + 1
            and _is_list_item(lines, marks, run[-1])
            and _is_attached(lines, marks[run[-1] + 1][0])
        ):
            run.append(run[-1] + 1)
        if len(run) >= _MIN_LIST_RUN and all(
            _is_list_item(lines, marks, i) for i in run[:-1]
        ):
            demote.update(marks[i][0] for i in run)
            position = run[-1] + 1
        else:
            position += 1
    if not demote:
        return full_text
    out = []
    for index, line in enumerate(lines):
        target = next((i for i in range(len(out) - 1, -1, -1) if out[i].strip()), None)
        if index in demote and target is not None:
            out[target] = f"{out[target]} {line.strip()}".rstrip()
        else:
            out.append(line)
    return "\n".join(out)


def _is_list_item(lines, marks, position) -> bool:
    """Whether the marker at `marks[position]` owns just one short line."""
    start = marks[position][0]
    end = marks[position + 1][0] if position + 1 < len(marks) else len(lines)
    body = [line for line in lines[start:end] if line.strip()]
    return len(body) == 1 and len(body[0]) <= _MAX_LIST_ITEM_LEN


def _is_attached(lines, index) -> bool:
    """Whether the line at `index` follows text directly, with no blank line."""
    return index > 0 and bool(lines[index - 1].strip())


def _group_relay_blocks(full_text: str) -> dict[int, str]:
    """Split a relay document on its "<pair>-<leg>." labels.

    parse_layout cannot be used here: the relay solutions packet walks the pairs
    in order and prints both legs of each ("1-1.", "1-2.", "2-1.", ...), so the
    flattened problem numbers arrive as 1, 11, 2, 12, ... and the
    strictly-increasing marker guard would reject every second block. The labels
    themselves are unambiguous, so scan them directly.
    """
    matches = list(_RELAY_MARKER_RE.finditer(full_text))
    blocks: dict[int, list[str]] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
        number = _relay_number(int(match.group(1)), int(match.group(2)))
        blocks.setdefault(number, []).append(full_text[match.end() : end].strip())
    return {number: "\n".join(parts).strip() for number, parts in blocks.items()}


def _marker_count(text: str) -> int:
    """Count distinct problem starts in one page's OCR."""
    return len(
        {
            item["problem"]
            for item in parse_layout(text, _match_marker, strict_section_restarts=True)
            if item["problem"] is not None
        }
    )


def _source_marker_count(text: str) -> int:
    """Count printed problem starts in a page's born-digital text layer.

    Only markers forming a consecutive run count. A PDF text layer breaks
    display math across lines, so a stray line can look like a marker ("5" from
    a fraction denominator followed by the next problem's "10."); requiring
    ``n, n+1, n+2, ...`` keeps the count a genuine lower bound rather than an
    over-count that would send every page around the retry ladder.
    """
    if _INSTRUCTION_PAGE_RE.search(text):
        return 0
    run = []
    for match in re.finditer(r"(?m)^[ \t]*(\d{1,2})[.)][ \t]+\S", text):
        number = int(match.group(1))
        tail = text[match.end() - 1 :].lstrip()
        if _RULE_START_RE.match(tail):
            continue
        if not run or number == run[-1] + 1:
            run.append(number)
    return len(run)


def _is_answer_key(full_text: str) -> bool:
    head = "\n".join(full_text.splitlines()[:12])
    return bool(
        re.search(r"\bIntegration\s+Bee\s+Answers\b", head, re.I)
        and not re.search(r"(?im)^\s*Solution\b", full_text)
    )


def _solution_body(block: str) -> str:
    """Strip a block's restated statement, keeping only the worked solution.

    A block runs statement -> "Proposed by" -> "Answer" -> worked prose, and any
    of the last three may be missing. Cut at the first label found, preferring
    the explicit "Solution" one. When nothing follows the answer the problem has
    no worked solution at all (estimation and tiebreaker questions): return
    empty rather than storing the restated statement as its own solution.
    """
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _SOLUTION_LINE_RE.match(line)
        if match is not None:
            first = match.group(1).lstrip("*_ ").strip()
            return "\n".join(([first] if first else []) + lines[index + 1 :]).strip()
    for index, line in enumerate(lines):
        match = _ANSWER_LINE_RE.match(line)
        if match is not None:
            return "\n".join(lines[index + 1 :]).strip()
    for index, line in enumerate(lines):
        if _PROPOSER_LINE_RE.match(line):
            return "\n".join(lines[index + 1 :]).strip()
    return block.strip()


def _is_restatement(body: str, statement: str) -> bool:
    """Whether `body` is just `statement` printed again (OCR drift allowed)."""
    if not statement.strip() or not body.strip():
        return False
    body_words, statement_words = _words(body), _words(statement)
    if not body_words or len(body_words) > len(statement_words) * 1.3:
        return False
    return (
        SequenceMatcher(None, body_words, statement_words).ratio()
        >= _RESTATEMENT_SIMILARITY
    )


def _words(text: str) -> list:
    return re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text).lower().split()


def _answer_value(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _ANSWER_LINE_RE.match(line)
        if match is None:
            continue
        inline = _clean_value(match.group(1))
        if inline:
            return inline
        for following in lines[index + 1 :]:
            if _SOLUTION_LINE_RE.match(following):
                break
            value = _clean_value(following)
            if value:
                return value
        return ""
    prose = _THE_ANSWER_IS_RE.search(block)
    if prose is not None:
        return _clean_value(prose.group(1))
    return _boxed_answer(block)


def _clean_value(value: str) -> str:
    value = value.strip().strip("*_").strip()
    if value == FIGURE_PLACEHOLDER or value.lower().startswith("proposed by"):
        return ""
    # An "Answer." line often reprints the closing step rather than the bare
    # result ("$11175 - 3 \\cdot 75 = \\boxed{10950}$"), and even a bare one
    # usually keeps its box and math delimiters. The box is the answer.
    boxed = _boxed_answer(value)
    if boxed:
        value = boxed
    # Peel the wrappers in either order: "$4825$." needs the period off before
    # the closing delimiter is reachable.
    previous = None
    while value != previous:
        previous = value
        value = value.strip().strip("$").strip("*_").rstrip(".").strip()
    return value
