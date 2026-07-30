"""USAMTS: one PDF per test, one solution PDF per test.

On-disk layout (data dir is ``USAMTS/out``)::

    out/<year>/<round>/test.pdf
    out/<year>/<round>/solutions.pdf

so each ``<year>/<round>`` is one test (id ``<year>_<round>``) and its solution
is the fixed-name sibling ``solutions.pdf``.

Numbering quirk: USAMTS prints problems as ``1/3/37.`` (problem / round / year).
The default matcher captures the *year* (the last group), which collapses every
problem on a round to the same number. Here we capture the *first* component --
the problem index within the round -- so problems come out 1, 2, 3, ... (see
TODOS.txt).
"""

import functools
import re
from pathlib import Path

from typing_extensions import override

from .. import anchors, config
from ..nanonets import normalize_img_placeholders
from .base import Series, Test, strip_solution_page_furniture

# Every statement in the corpus uses the distinctive "N/R/Y." form.  Do not
# fall back to bare "N." markers here: USAMTS booklets begin with numbered
# submission instructions and many problems contain numbered rules/steps.  The
# fallback used to turn all of those into phantom problems.
_USAMTS_PATTERNS = [
    re.compile(r"^\s*(\d+)\s*/\s*\d+\s*/\s*\d+\s*\."),  # "1/3/37." -> 1
]

# USAMTS closes the problem set with a rule of asterisks ("**************")
# followed by submission instructions and a mailing address. That trailing
# furniture has no problem marker, so it binds to the last problem -- cut it.
_SEPARATOR_RE = re.compile(r"^\s*\*{5,}\s*$")
_TRAILING_INSTRUCTIONS_RE = re.compile(
    r"(?im)^\s*(?:"
    r"Complete,\s*well-written\s+solutions\b|"
    r"If\s+you\s+have\s+not\s+already\s+sent\s+an\s+Entry\s+Form\b|"
    r"Solutions?\s+(?:must|should)\s+be\s+(?:mailed|submitted)\b"
    r")"
)
_STATEMENT_MARKER_RE = re.compile(
    r"(?m)^\s*[*_#]*\s*\d+\s*/\s*\d+\s*/\s*\d+\s*\."
)
_LOGO_IMG_RE = re.compile(
    r"<img\b[^>]*>\s*(?:USA\s*MTS|USAMTS)\s*</img>",
    re.IGNORECASE | re.DOTALL,
)
_PAGE_FURNITURE_RE = re.compile(
    r"(?im)^\s*(?:"
    r"USA\s+Mathematical\s+Talent\s+Search|"
    r"(?:Round\s+\d+\s+)?Problems?\s*$|"
    r"Round\s+\d+\s*[—-]\s*Year\s+\d+.*$|"
    r"Year\s+\d+\s*[—-]\s*Academic\s+Year.*$|"
    r"Each\s+problem\s+is\s+worth\s+\d+\s+points?\.?|"
    r"These\s+are\s+only\s+part\s+of\s+the\s+complete\s+rules\.?|"
    r"Problem\s+\d+\s+on\s+(?:the\s+)?next\s+page\.?|"
    r"www\.usamts\.org"
    r")\s*$"
)
_INSTRUCTION_SIGNALS = (
    "submit your solutions",
    "every page you submit",
    "usamts id",
    "once you send in your solutions",
    "entry form",
    "postmark deadline",
    "my usamts",
)

# --- Solution-packet structure ---
# A USAMTS solutions PDF ("PROBLEMS / SOLUTIONS / COMMENTS") separates problems
# by the bolded date marker "**3/1/12.**"; within a problem, each solution is a
# "**Solution k by <name>:**" header (sometimes emitted as LaTeX
# "\textbf{Solution k by ...}"), and the trailing "**Editor's Comment:**" is
# commentary, not a solution.
_PROBLEM_MARKER_RE = re.compile(
    r"^\s*(?:\*{1,2}|\\textbf\{)?\s*(\d+)\s*/\s*\d+\s*/\s*\d+\s*\.?\s*(?:\*{1,2}|\})?\s*",
    re.MULTILINE,
)
_SOLUTION_HEADER_RE = re.compile(
    r"^\s*(?:\*{1,2}|\\textbf\{)?\s*Solution"
    r"(?:\s+\d+)?"
    r"(?:\s+(?:for|to)\s+\d+\s*/\s*\d+\s*/\s*\d+)?"
    r"(?:\s+by\b[^:]*)?\s*:?\s*(?:\*{1,2}|\})?\s*",
    re.IGNORECASE,
)
_EDITOR_RE = re.compile(
    r"^\s*(?:\*{1,2}|\\textbf\{)?\s*(?:"
    r"Editor.?s\s+Comments?|Comments?\s+from\s+the\s+solutions?\s+editor"
    r")",
    re.IGNORECASE,
)
_COMMENT_RE = re.compile(
    r"^\s*(?:\*{1,2}|\\textbf\{)?\s*(?:Editor.?s\s+Comments?|Comment\b)",
    re.IGNORECASE,
)
_SOLUTION_ONE_RE = re.compile(
    r"^\s*(?:\*{1,2}|\\textbf\{)?\s*Solution\s+1\b", re.IGNORECASE
)
_SOLUTION_FURNITURE_RE = re.compile(
    r"(?im)^\s*[*_#]*\s*(?:"
    r"USA\s+Mathematical\s+Talent\s+Search|"
    r"PROBLEMS\s*/\s*SOLUTIONS\s*/\s*COMMENTS|"
    r"CREDITS\s+and\s+QUICK\s+ANSWERS|"
    r"Round\s+\d+\s+Solutions|"
    r"Solutions?\s+to\s+Problem\s+\d+\s*/\s*\d+\s*/\s*\d+|"
    r"Round\s+\d+\s*[—-]\s*Year\s+\d+.*$|"
    r"Year\s+\d+\s*[—-]\s*Academic\s+Year.*$|"
    r"www\.usamts\.org"
    r")\s*[*_#]*\s*$"
)
_SOLUTION_TRAILING_RE = re.compile(
    r"(?im)^\s*(?:"
    r"Round\s+\d+\s+Solutions?\s+must\s+be\s+submitted\b|"
    r"Please\s+visit\s+.*usamts\.org.*solution\s+submission\b"
    r")"
)
_SOLUTION_EDITORIAL_FURNITURE_RE = re.compile(
    r"^(?:"
    r"solutions?\s+edited\s+by\b.*"
    r"|all\s+other\s+problems?\s+and\s+solutions?\s+by\b.*"
    r"|page\s+\d+"
    r")$",
    re.IGNORECASE,
)
_SOLUTION_EDITORIAL_TAIL_RE = re.compile(
    r"\s*[*_]*(?:"
    r"solutions?\s+edited\s+by\b.*"
    r"|all\s+other\s+problems?\s+and\s+solutions?\s+by\b.*"
    r")[*_]*\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _cut_at_separator(text: str) -> str:
    """Return `text` truncated at the first ``**************`` rule line."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _SEPARATOR_RE.match(line):
            return "\n".join(lines[:i]).rstrip()
    return text


def _clean_statement_markdown(text: str) -> str:
    """Remove USAMTS instruction pages and recurring statement furniture."""
    compact = re.sub(r"\s+", " ", text).casefold()
    if not _STATEMENT_MARKER_RE.search(text):
        signal_count = sum(signal in compact for signal in _INSTRUCTION_SIGNALS)
        if signal_count >= 2:
            return ""
    text = _LOGO_IMG_RE.sub("", text)
    text = _PAGE_FURNITURE_RE.sub("", text)
    boundary = _TRAILING_INSTRUCTIONS_RE.search(text)
    if boundary:
        text = text[: boundary.start()]
    return _cut_at_separator(text).strip()


def _clean_solution_markdown(text: str) -> str:
    """Remove running solution headers and their inline logo placeholders."""
    compact = re.sub(r"\s+", " ", text).casefold()
    if not _PROBLEM_MARKER_RE.search(text):
        signal_count = sum(signal in compact for signal in _INSTRUCTION_SIGNALS)
        if signal_count >= 2:
            return ""
    text = _LOGO_IMG_RE.sub("", text)
    text = _SOLUTION_FURNITURE_RE.sub("", text)
    boundary = _SOLUTION_TRAILING_RE.search(text)
    if boundary:
        text = text[: boundary.start()]
    text = _SOLUTION_EDITORIAL_TAIL_RE.sub("", text)
    return strip_solution_page_furniture(
        text, line_patterns=(_SOLUTION_EDITORIAL_FURNITURE_RE,)
    )


def _headerless_solution(lines):
    """Drop the restated statement/credit prelude from a quick-answer block."""
    paragraphs = []
    current = []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    if len(paragraphs) <= 1:
        return "\n".join(lines).strip()
    # The marker line's remainder is always the first paragraph: the restated
    # problem.  Years 14-15 then print zero or more provenance/comment
    # paragraphs before beginning the actual worked answer.
    paragraphs = paragraphs[1:]
    while paragraphs and re.match(
        r"(?i)^\s*(?:This|The)\s+problem\b|^\s*(?:Comment|We\s+thank)\b",
        paragraphs[0][0],
    ):
        paragraphs.pop(0)
    return "\n\n".join("\n".join(p) for p in paragraphs).strip()


def _split_solution_blocks(lines):
    """Split one problem body's lines into individual solution texts.

    Lines before the first "Solution k by" header (the restated statement) are
    dropped; the "Editor's Comment" and everything after it is dropped. If no
    solution headers are present, the whole body (minus the editor note) is kept
    as a single block so nothing is lost.
    """
    blocks = []
    buf = None  # None until the first solution header is seen
    for line in lines:
        if _EDITOR_RE.match(line) and buf is not None:
            break
        if _SOLUTION_HEADER_RE.match(line):
            if buf is not None:
                blocks.append("\n".join(buf).strip())
            buf = [line]
            continue
        if buf is not None:
            buf.append(line)
    if buf is not None:
        blocks.append("\n".join(buf).strip())
    blocks = [b for b in blocks if b]
    if not blocks:
        text = _headerless_solution(lines)
        if text:
            blocks = [text]
    return blocks


def _recover_missing_problem_bodies(bodies):
    """Recover two old packets whose OCR omitted a problem marker.

    In both packets the previous problem's editor/comment line survived, and a
    fresh ``Solution 1`` follows it.  That reset is deterministic evidence that
    the following lines belong to the next problem, not another solution to the
    current one.
    """
    recovered = dict(bodies)
    for number in sorted(bodies):
        if number + 1 in recovered:
            continue
        lines = bodies[number]
        for i, line in enumerate(lines):
            if not _COMMENT_RE.match(line):
                continue
            trailing = lines[i + 1 :]
            if (
                any(_SOLUTION_HEADER_RE.match(candidate) for candidate in lines[:i])
                and any(_SOLUTION_ONE_RE.match(candidate) for candidate in trailing)
            ):
                recovered[number] = lines[:i]
                recovered[number + 1] = trailing
                break
    # Year 11 Round 1 lost the marker, statement, and first solution header for
    # problem 4.  The surviving markers jump from 3 to 5, and the last solution
    # block under 3 is therefore the only possible body for the missing 4.
    for following in sorted(bodies):
        missing = following - 1
        previous = following - 2
        if missing in recovered or previous not in recovered:
            continue
        lines = recovered[previous]
        starts = [
            i for i, line in enumerate(lines) if _SOLUTION_HEADER_RE.match(line)
        ]
        if len(starts) >= 2:
            split = starts[-1]
            recovered[previous] = lines[:split]
            recovered[missing] = lines[split:]
    return recovered


class UsamtsSeries(Series):
    name = "usamts"
    has_solutions = True

    @override
    def discover_tests(self, data_dir):
        """One test per ``<year>/<round>/test.pdf`` under the data dir."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for pdf in sorted(root.glob("*/*/test.pdf")):
            test_id = f"{pdf.parent.parent.name}_{pdf.parent.name}"
            tests.append(Test(id=test_id, source=pdf))
        return tests

    @override
    def match_marker(self):
        return functools.partial(anchors._match_marker, patterns=_USAMTS_PATTERNS)

    @override
    def solution_source(self, test):
        """The fixed-name sibling ``solutions.pdf``, or None if absent."""
        sol = test.source.parent / "solutions.pdf"
        return sol if sol.exists() else None

    @override
    def parse_solutions(self, full_text):
        """Parse a USAMTS solutions packet into {problem_number: [solution, ...]}.

        Problems are split on the bolded date marker ("**3/1/12.**"); within each
        problem, every "**Solution k by <name>:**" block (with its diagrams,
        tables and math) is one entry. The restated statement and the trailing
        "Editor's Comment" are dropped -- only the solutions are kept.
        """
        bodies = {}  # problem number -> list of body lines
        current = None
        last = None
        for line in full_text.splitlines():
            m = _PROBLEM_MARKER_RE.match(line)
            if m is not None and (last is None or int(m.group(1)) > last):
                current = last = int(m.group(1))
                bodies[current] = []
                rest = line[m.end() :].rstrip()
                if rest:
                    bodies[current].append(rest)
                continue
            if current is not None:
                bodies[current].append(line)
        bodies = _recover_missing_problem_bodies(bodies)
        # The raw "<img>" tags survive line-based splitting; normalize each block's
        # to the reading-order sentinel so the pipeline can align them with DETR's
        # crops (see pipeline.inline_solution_figures).
        return {
            n: [normalize_img_placeholders(b) for b in _split_solution_blocks(lines)]
            for n, lines in bodies.items()
        }

    @override
    def layout_options(self):
        """Drop the recurring USAMTS masthead DETR reads as a Picture."""
        return config.LayoutOptions(header_picture_frac=0.20)

    @override
    def clean_statement_markdown(self, page_index, markdown):
        return _clean_statement_markdown(markdown)

    @override
    def clean_solution_markdown(self, page_index, markdown):
        return _clean_solution_markdown(markdown)

    @override
    def postprocess(self, problems):
        """Drop the trailing submission-instructions/address footer.

        The ``**************`` rule marks the end of the *textual* problem set;
        everything after it (submission instructions, mailing address) is page
        furniture. Only text is dropped -- image crops are assigned by geometry,
        not reading order, so a figure attached to the last problem must survive
        even though it is appended after that problem's text.
        """
        for p in problems:
            kept = []
            cut = False
            for el in p.elements:
                if el.kind == "text":
                    if cut:
                        continue  # trailing footer text after the separator
                    original = el.text
                    trimmed = _clean_statement_markdown(original)
                    if trimmed != el.text:
                        el.text = trimmed
                        cut = bool(
                            _TRAILING_INSTRUCTIONS_RE.search(original)
                            or any(
                                _SEPARATOR_RE.match(line)
                                for line in original.splitlines()
                            )
                        )
                    if not el.text.strip():
                        continue  # element became empty
                kept.append(el)
            p.elements = kept
        return problems
