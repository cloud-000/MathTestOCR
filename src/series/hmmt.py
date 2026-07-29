"""Harvard-MIT Mathematics Tournament tests and solutions.

On-disk layout (data dir is ``HMMT/out``)::

    out/<year>/<feb|nov>/<round>/test.pdf
    out/<year>/<feb|nov>/<round>/solutions.pdf
    out/<year>/hmic/test.pdf
    out/<year>/hmic/solutions.pdf

Test IDs mirror their path, joined by underscores: ``2017_feb_algnt`` and
``2013_hmic``. Most problems use ordinary integer markers (``1. [5] ...``);
older per-subject documents use a round-lettered marker instead --
``Problem A1`` (Algebra), ``C8`` (Calculus), ``G3`` (Geometry),
``T4``/``AT10`` (Team, Advanced Topics), ``Gu1`` (Guts), ``O1`` (Oral).
"""

import re
import hashlib
from difflib import SequenceMatcher
from pathlib import Path

from typing_extensions import override

from .. import anchors, answer_llm, config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test


_ANSWER_RE = re.compile(
    r"^\s*(?:\*{1,2}|_+)?\s*Answer\s*:\s*(?:\*{1,2}|_+)?\s*(.*)$",
    re.IGNORECASE,
)
_PROPOSED_RE = re.compile(
    r"^\s*(?:\*{1,2}|_+)?\s*Proposed\s+by\s*:", re.IGNORECASE
)
_SOLUTION_RE = re.compile(
    r"^\s*(?:\*{1,2}|_+)?\s*Solution(?:\s+\d+)?\s*:?\s*"
    r"(?:\*{1,2}|_+)?\s*(.*)$",
    re.IGNORECASE,
)
# Older rounds print a round-lettered marker instead of a bare integer:
# "Problem A1" (Algebra), "C8" (Calculus), "G3" (Geometry), "T4"/"AT10" (Team,
# Advanced Topics), plus "Gu1" (Guts) and "O1" (Oral). Any 1-3 letter prefix
# glued (or spaced) to the problem number counts.
_PREFIXED_MARKER_RE = re.compile(r"^\s*Problem\s+[A-Za-z]{1,3}\s*(\d+)\b", re.IGNORECASE)
# OCR sometimes renders \boxed{X} as a <box>X</box> tag; both are answer boxes.
_BOX_TAG_RE = re.compile(r"<box>\s*(.*?)\s*</box>", re.I | re.S)
_INLINE_MATH_RE = re.compile(r"\${1,2}.*?\${1,2}")
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_PROSE_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_LEADING_NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?"
)
_ANSWER_CHECKBOXES = ("☐", "☑")
_LEADING_TAG_RE = re.compile(
    r"^\s*(?:<(?:p|strong|b|em|span)\b[^>]*>\s*)+", re.IGNORECASE
)
_WORD_MARKER_RE = re.compile(
    r"^\s*Question\s+"
    r"(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\s*[.:]?",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    word.casefold(): number
    for number, word in enumerate(
        ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"),
        start=1,
    )
}
_POWER_QUESTION_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:1998\s+)?Power\s+Question(?:\s+Solutions)?\b\s*[-:]?",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^\s*```(?:html)?\s*$", re.IGNORECASE | re.MULTILINE)
_BLOCK_TAG_RE = re.compile(
    r"</?(?:html|body|head|main|section|article|p|div)\b[^>]*>",
    re.IGNORECASE,
)
_INLINE_TAG_RE = re.compile(
    r"</?(?:strong|b|em|span|center|font)\b[^>]*>",
    re.IGNORECASE,
)
_LIST_TAG_RE = re.compile(r"</?(?:ol|ul|li)\b[^>]*>", re.IGNORECASE)
_BANNER_RE = re.compile(
    r"(?:\*{0,2}|#{1,6}\s*)?"
    r"(?:\d+(?:st|nd|rd|th|\^\{?th\}?|<sup>th</sup>)?\s+(?:annual\s+)?)?"
    r"Harvard\s*[-/]\s*MIT\b[^\n]*(?:Tournament|Guts\s+Round|Team\s+Round)"
    r"(?:\*{0,2})?",
    re.IGNORECASE,
)
_HMMT_DATED_HEADER_RE = re.compile(
    r"(?:\*{0,2}|#{1,6}\s*)?HMMT(?:\s+[A-Za-z]+)?\s+\d{4}\b[^\n]*",
    re.IGNORECASE,
)
_COPYRIGHT_LINE_RE = re.compile(
    r"^\s*(?:[-—]+\s*)?(?:©|\(c\)\s*)\s*\d{4}\s+HMMT\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TIME_LIMIT_RE = re.compile(
    r"(?:\s*[-—]{2,}\s*)?\*{0,2}\s*Time\s+limit\s*:\s*[^.\n]+\.?\s*\*{0,2}",
    re.IGNORECASE,
)
_FORM_TAIL_RE = re.compile(
    r"\s+(?:School|Team\s+ID#?|Team|Name)\s*:?\s*(?:_+|\[\s*\])?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FORM_LINE_RE = re.compile(
    r"^\s*(?:(?:School|Team(?:\s+ID#?)?|Name|Grade|Score)\s*:?"
    r"\s*(?:_+|\[\s*\])?\s*){2,}$",
    re.IGNORECASE | re.MULTILINE,
)
_FORM_LABEL_RE = re.compile(
    r"\b(?:Organization|School|Team(?:\s+ID#?)?|Name|Grade|Score)\b",
    re.IGNORECASE,
)
_ANSWER_BLANK_LINE_RE = re.compile(
    r"^\s*\d{1,3}\s*[.)]\s*(?:\*{0,2})?\[\s*(?:±\s*)?\d+\s*\]"
    r"(?:\*{0,2})?\s*(?:\[?\s*_{3,}\s*\]?)\s*$"
)
_SOURCE_POINT_MARKER_RE = re.compile(
    r"^\s*(?:Problem\s+[A-Za-z]{1,3}\s*)?(\d+)\s*[.)]\s*"
    r"\[\s*(?:±\s*)?\d+\s*\]",
    re.IGNORECASE | re.MULTILINE,
)
_SOURCE_RESPONSE_MARKER_RE = re.compile(
    r"^\s*\d{1,3}\s*[.)]\s*\[\s*(?:±\s*)?\d+\s*\]\s*$",
    re.MULTILINE,
)
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SUBPART_RE = re.compile(r"^\s*(?:\*{1,2})?\([a-z]\)\s+", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _match_marker(text):
    # OCR frequently wraps the printed marker itself in <strong>/<p>. Ignore
    # only leading inline markup, then translate the match end back to the raw
    # string so parse_layout strips the complete marker correctly.
    tag_match = _LEADING_TAG_RE.match(text)
    offset = tag_match.end() if tag_match is not None else 0
    probe = text[offset:]
    match = _WORD_MARKER_RE.match(probe)
    if match is not None:
        return _WORD_NUMBERS[match.group(1).casefold()], offset + match.end()
    match = _POWER_QUESTION_RE.match(probe)
    if match is not None:
        return 1, offset + match.end()
    match = _PREFIXED_MARKER_RE.match(probe)
    if match is not None:
        return int(match.group(1)), offset + match.end()
    match = anchors._match_marker(probe)
    if match is None:
        return None
    return match[0], offset + match[1]


def _clean_hmmt_markdown(markdown: str, *, strip_lists: bool = False) -> str:
    """Remove HMMT page furniture and OCR wrapper markup without touching math."""
    text = _FENCE_RE.sub("", markdown)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _INLINE_TAG_RE.sub("", text)
    text = _COPYRIGHT_LINE_RE.sub("", text)
    text = _BANNER_RE.sub("", text)
    text = _HMMT_DATED_HEADER_RE.sub("", text)
    text = _TIME_LIMIT_RE.sub("", text)
    text = _FORM_LINE_RE.sub("", text)
    text = _FORM_TAIL_RE.sub("", text)
    if strip_lists:
        text = _LIST_TAG_RE.sub("\n", text)
    lines = []
    for line in text.splitlines():
        if _ANSWER_BLANK_LINE_RE.match(line):
            continue
        form_labels = {
            match.group(0).casefold() for match in _FORM_LABEL_RE.finditer(line)
        }
        if (
            len(form_labels) >= 2
            and ("_" in line or "[" in line)
            and "?" not in line
        ):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def _marker_count(text: str) -> int:
    """Count distinct problem starts on one page without assuming its first number."""
    problems = {
        item["problem"]
        for item in parse_layout(
            text,
            _match_marker,
            point_value_list_markers=True,
            strict_section_restarts=True,
            page_initial_point_restart=True,
        )
        if item["problem"] is not None
    }
    return len(problems)


def _source_marker_count(text: str) -> int:
    """Lower-bound starts from the PDF text, preferring unambiguous point rows."""
    point_numbers = {
        int(match.group(1)) for match in _SOURCE_POINT_MARKER_RE.finditer(text)
    }
    return len(point_numbers) if point_numbers else _marker_count(text)


class HmmtSeries(Series):
    name = "hmmt"
    has_solutions = True
    has_answers = True

    @override
    def discover_tests(self, data_dir):
        """Discover every HMMT ``test.pdf`` recursively.

        The full parent path forms the ID, avoiding collisions between the
        February, November, and HMIC collections and their many round types.
        """
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        return [
            Test(
                id="_".join(pdf.relative_to(root).parts[:-1]),
                source=pdf,
            )
            for pdf in sorted(root.glob("**/test.pdf"))
        ]

    @override
    def test_pages(self, test: Test, workdir):
        """Render pages and discard byte-identical PDF duplicates.

        A small number of HMMT source PDFs physically repeat the same compiled
        test page many times (not merely the same text layer). Feeding those
        copies through the cross-page restart logic manufactures extra problem
        ranges and duplicates statement text. Record the number of marker-like
        starts in each retained page's born-digital text layer at the same time;
        ``validate_statement_markdown`` uses it to reject clean-looking OCR
        responses that silently stopped after only part of the page.
        """
        pages = super().test_pages(test, workdir)
        unique = []
        expected_starts = []
        seen = set()
        source_text = {}
        source = Path(test.source)
        if source.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(source) as doc:
                source_text = {
                    page_number + 1: page.get_text()
                    for page_number, page in enumerate(doc)
                }
        for page in pages:
            digest = hashlib.sha256(Path(page).read_bytes()).digest()
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(page)
            match = re.search(r"(\d+)$", Path(page).stem)
            pdf_page = int(match.group(1)) if match is not None else None
            expected_starts.append(
                _source_marker_count(source_text.get(pdf_page, ""))
                if pdf_page is not None
                else 0
            )
        self._expected_statement_starts = expected_starts
        return unique

    @override
    def match_marker(self):
        return _match_marker

    @override
    def layout_options(self):
        """Keep statement and solution figures at their reading-order position."""
        return config.LayoutOptions(
            inline_figures=True,
            point_value_list_markers=True,
            strict_section_restarts=True,
            page_initial_point_restart=True,
            equation_text_overlap=0.3,
            solution_equation_text_overlap=True,
            solution_answer_box_filter=True,
        )

    @override
    def skip_page(self, text: str) -> bool:
        """Drop answer sheets and instruction-only covers before OCR/model work."""
        compact = " ".join(text.split()).casefold()
        if not compact:
            return False
        numbered = re.findall(r"(?m)^\s*\d{1,3}\s*[.)]\s*", text)
        distinct_words = {
            word.casefold() for word in re.findall(r"[A-Za-z]{3,}", text)
        }
        if "answer sheet" in compact and (
            not numbered or len(distinct_words) < 30
        ):
            return True
        form_fields = sum(
            token in compact
            for token in ("name:", "school", "organization", "team id", "score:")
        )
        blank_runs = len(re.findall(r"_{4,}", text))
        response_rows = _SOURCE_RESPONSE_MARKER_RE.findall(text)
        if len(response_rows) >= 6 and len(distinct_words) < 20:
            return True
        if (
            form_fields >= 2
            and numbered
            and (blank_runs >= 6 or (len(compact) < 400 and "?" not in compact))
        ):
            return True
        if (
            "this test consists of" in compact
            and ("enjoy!" in compact or "do not open" in compact)
            and not re.search(r"\b\d+\s*[.)]\s+\S", compact)
        ):
            return True
        return False

    @override
    def clean_statement_markdown(self, page_index: int, markdown: str) -> str:
        return _clean_hmmt_markdown(markdown)

    @override
    def validate_statement_markdown(self, page_index: int, markdown: str) -> bool:
        expected = getattr(self, "_expected_statement_starts", ())
        if page_index >= len(expected) or expected[page_index] == 0:
            return True
        return _marker_count(_clean_hmmt_markdown(markdown)) >= expected[page_index]

    @override
    def clean_solution_markdown(self, page_index: int, markdown: str) -> str:
        return _clean_hmmt_markdown(markdown)

    @override
    def postprocess(self, problems):
        for problem in problems:
            kept = []
            for element in problem.elements:
                if element.kind == "text":
                    element.text = _clean_hmmt_markdown(
                        element.text, strip_lists=True
                    )
                    if not element.text:
                        continue
                kept.append(element)
            problem.elements = kept
        return problems

    @override
    def solution_source(self, test):
        sol = test.source.parent / "solutions.pdf"
        return sol if sol.exists() else None

    @override
    def answer_source(self, test):
        return self.solution_source(test)

    @override
    def parse_solutions(self, full_text):
        """Drop each restated statement and keep only its worked solution.

        Older HMIC documents put the solution directly after ``Answer:``;
        newer documents add ``Proposed by:`` and ``Solution:`` labels. Figure
        placeholders are retained so DETR crops remain inline.

        A Guts round is a special case: its "solutions" document is really an
        answer key -- every block is a bare answer, not a worked solution -- so
        those entries are dropped here (they live only in ``problem_answer.json``
        via `_bare_answer`). This is detected document-wide, never per-problem,
        so a normal round's occasional short one-line solution is still kept.
        """
        full_text = _clean_hmmt_markdown(full_text)
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(
            full_text,
            self.match_marker(),
            point_value_list_markers=True,
            strict_section_restarts=True,
        ):
            if item["problem"] is None:
                continue
            value = item["text"] if item["kind"] == "text" else FIGURE_PLACEHOLDER
            grouped.setdefault(item["problem"], []).append(value)
        bodies = {n: _solution_body("\n".join(parts)) for n, parts in grouped.items()}
        nonempty = [b for b in bodies.values() if b]
        bare = [b for b in nonempty if _bare_answer(b)]
        if nonempty and len(bare) >= _ANSWER_KEY_BARE_FRAC * len(nonempty):
            return {n: b for n, b in bodies.items() if b and not _bare_answer(b)}
        return bodies

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Read each problem's final answer from the solutions document.

        Three tiers, in order: a printed ``Answer:`` line or ``\\boxed{...}``
        (`_answer_value`); a Guts-style answer-key entry, whose whole block is
        just the bare answer with no restated statement or worked solution
        (`_bare_answer`); and, for the older per-subject rounds that bury the
        answer mid-sentence in prose, the answer LLM. Problems that resolve to
        nothing are omitted -- a partial key, never a guessed one.
        """
        answers = {}
        full_text = _clean_hmmt_markdown("\n\n".join(pages_markdown))
        blocks = _group_blocks(full_text, self.match_marker())
        for number, block in blocks.items():
            answer = _answer_value(block) or _bare_answer(block) or answer_llm.extract(block)
            if answer and answer.upper() != "N/A":
                answers[number] = answer
        return answers

    @override
    def postprocess_solutions(
        self, solutions: dict, statements: dict, test: Test = None
    ) -> dict:
        cleaned = {}
        for number, value in solutions.items():
            statement = statements.get(str(number), "")
            cleaned[number] = _strip_restatement(value, statement)
        return cleaned


def _group_blocks(full_text: str, match) -> dict:
    """Group the solution document's text into ``{problem_number: block_text}``.

    Uses the same layout splitter and markers as the solution/figure passes so
    every path numbers problems identically. Figure positions are dropped (the
    answer regexes and the LLM never need them)."""
    grouped: dict[int, list[str]] = {}
    for item in parse_layout(
        full_text,
        match,
        point_value_list_markers=True,
        strict_section_restarts=True,
        page_initial_point_restart=True,
    ):
        if item["problem"] is None or item["kind"] != "text":
            continue
        grouped.setdefault(item["problem"], []).append(item["text"])
    return {number: "\n".join(parts).strip() for number, parts in grouped.items()}


def _answer_value(block: str) -> str:
    """Return an explicit, standalone answer or ``""`` for the LLM fallback.

    Some HMMT solution PDFs put the answer and its proof in one paragraph.
    Whole-page OCR consequently emits lines such as ``Answer: 64 Each term
    ...``. Treating the complete tail as the answer pollutes
    ``problem_answer.json`` and, worse, can preserve a malformed OCR box (the
    2008 November Guts solution for problem 24 loses its denominator). Only
    compact answer-like tails are authoritative; prose-bearing tails fail soft
    so ``parse_answers`` can ask ``answer_llm`` to read the full block.
    """
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _ANSWER_RE.match(line)
        if match is None:
            continue
        inline = match.group(1).strip()
        if inline:
            return _explicit_answer_prefix(inline)
        for following in lines[index + 1 :]:
            if _PROPOSED_RE.match(following) or _SOLUTION_RE.match(following):
                break
            if following.strip():
                return _explicit_answer_prefix(following.strip())
        return ""
    # No explicit "Answer:" line -- fall back to the boxed final answer.
    return _boxed_answer(block) or ""


def _is_standalone_answer(value: str) -> bool:
    """Whether an ``Answer:`` tail contains a value rather than value + proof."""
    if not value or value.startswith(_ANSWER_CHECKBOXES):
        return False

    # Ignore mathematical content while looking for natural-language prose.
    # This preserves arbitrarily rich LaTeX answers while rejecting flattened
    # explanations such as "3 Substitute x ..." and "5.85086 Let the ...".
    prose = _INLINE_MATH_RE.sub(" ", value)
    prose = _LATEX_COMMAND_RE.sub(" ", prose)
    return len(_PROSE_WORD_RE.findall(prose)) <= 1


def _explicit_answer_prefix(value: str) -> str:
    """Recover an unambiguous value prefix; reject the rest for LLM extraction."""
    if _is_standalone_answer(value):
        return _boxed_answer(value) or value.strip(" $")
    if not value or value.startswith(_ANSWER_CHECKBOXES):
        return ""

    # A leading unboxed math span is an answer followed by flattened prose.
    # Do not trust a boxed prefix in this situation: Nanonets can crop a stacked
    # fraction inside the box (2008 November Guts problem 24).
    math = _INLINE_MATH_RE.match(value)
    if math is not None and r"\boxed" not in math.group(0):
        return math.group(0).strip("$").strip()

    # The same flattening commonly produces "64 Each term ..." or
    # "5.85086 Let ...". Keep the numeric prefix unless it is visibly only the
    # first member of a compound answer ("3 and 5 ...").
    number = _LEADING_NUMBER_RE.match(value)
    if number is not None:
        remainder = value[number.end() :].lstrip()
        if not re.match(r"^(?:and|or)\b", remainder, re.IGNORECASE):
            return number.group(0)
    return ""


def _iter_boxes(text: str):
    r"""Yield each answer box's inner text: ``\boxed{...}`` (brace-balanced, so a
    nested ``\frac{a}{b}`` survives) and each ``<box>...</box>`` tag, in order."""
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
                    yield text[open_brace + 1 : k]
                    break
            k += 1
        i = k + 1
    yield from _BOX_TAG_RE.findall(text)


def _boxed_answer(block: str) -> str:
    """The last boxed value in `block` (the final answer), cleaned, or ""."""
    value = ""
    for box in _iter_boxes(block):
        box = box.strip().strip("$").strip().rstrip(".").strip()
        if box:
            value = box
    return value


# A Guts answer-key entry ("3. 57", "7. $(\frac{x^2+1}{2})^m$.") restates no
# problem and works no solution -- the whole block is the bare answer on one
# short line. A real (prose) solution spans multiple lines or runs long, so
# those are left to the answer LLM instead of being mistaken for the answer.
_BARE_ANSWER_MAX_LEN = 80
# A document is treated as a Guts-style answer key (bare answers, no solutions)
# only when this fraction of its non-empty blocks are bare -- so one short
# solution in an otherwise-worked round never trips it.
_ANSWER_KEY_BARE_FRAC = 0.8


def _bare_answer(block: str) -> str:
    """`block` treated as a verbatim bare answer (Guts key), cleaned, or ""."""
    stripped = block.strip()
    if not stripped or "\n" in stripped or stripped.endswith("?"):
        return ""
    if len(stripped) > _BARE_ANSWER_MAX_LEN:
        return ""
    # Drop surrounding math delimiters and trailing punctuation together, so a
    # "$...$." entry loses both the closing "$" and the period after it.
    return stripped.strip(" $.")


def _solution_body(block: str) -> str:
    lines = block.splitlines()

    # Modern PDFs explicitly mark the start of each worked solution.
    for index, line in enumerate(lines):
        match = _SOLUTION_RE.match(line)
        if match is not None:
            first = match.group(1).strip()
            kept = ([first] if first else []) + lines[index + 1 :]
            return "\n".join(kept).strip()

    answer_rows = [
        (index, _ANSWER_RE.match(line))
        for index, line in enumerate(lines)
        if _ANSWER_RE.match(line) is not None
    ]
    # A multi-part problem can print one Answer label per subpart. Keep every
    # answer/proof segment while dropping the next restated "(b) ..." prompt.
    if len(answer_rows) > 1:
        kept = []
        keeping = False
        for line in lines:
            answer = _ANSWER_RE.match(line)
            if answer is not None:
                keeping = True
                value = answer.group(1).strip()
                if value:
                    kept.append(f"Answer: {value}")
                continue
            if _SUBPART_RE.match(line):
                keeping = False
                continue
            if keeping:
                kept.append(line)
        return "\n".join(kept).strip()

    # Older PDFs have no Solution label; their solution begins after Answer.
    for index, line in enumerate(lines):
        match = _ANSWER_RE.match(line)
        if match is not None:
            start = index + 1
            inline = match.group(1).strip()
            # When the value is on the next line, omit that line too. If the
            # value is inline, the proof begins immediately on the next line.
            if not inline:
                while start < len(lines) and not lines[start].strip():
                    start += 1
                inline = lines[start].strip() if start < len(lines) else ""
                start += 1
            kept = ([f"Answer: {inline}"] if inline else []) + lines[start:]
            return "\n".join(kept).strip()

    # Preserve unexpected layouts rather than silently dropping their text.
    return "\n".join(line for line in lines if not _PROPOSED_RE.match(line)).strip()


def _strip_restatement(solution: str, statement: str) -> str:
    """Remove an exact word-for-word statement prefix from an unlabeled solution."""
    if not solution or not statement:
        return solution
    statement = _IMAGE_REF_RE.sub("", statement)
    statement_words = [m.group(0).casefold() for m in _WORD_RE.finditer(statement)]
    solution_matches = list(_WORD_RE.finditer(solution))
    solution_words = [m.group(0).casefold() for m in solution_matches]
    if len(statement_words) < 8 or len(solution_words) < 8:
        return solution
    same_opening = solution_words[:8] == statement_words[:8]
    if len(solution_words) < len(statement_words):
        # The OCR stopped during the restated prompt and never reached a worked
        # solution. Omit the fake solution instead of publishing a truncated
        # copy of the statement.
        return "" if same_opening else solution
    candidate = solution_words[: len(statement_words)]
    exact = candidate == statement_words
    close = SequenceMatcher(None, statement_words, candidate).ratio() >= 0.75
    if not exact and not (same_opening and close):
        return solution
    end = solution_matches[len(statement_words) - 1].end()
    return solution[end:].lstrip(" \t\r\n.:;—-")
