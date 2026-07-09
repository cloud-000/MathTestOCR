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

from .. import anchors
from ..nanonets import normalize_img_placeholders
from .base import Series, Test

# USAMTS-specific marker patterns. The "N/R/Y." form must come first and capture
# the leading problem index; the other forms mirror the defaults as a fallback.
_USAMTS_PATTERNS = [
    re.compile(r"^\s*(\d+)\s*/\s*\d+\s*/\s*\d+\s*\."),  # "1/3/37." -> 1
    re.compile(r"^\s*(?:Problem|Question|Prob\.?)\s+(\d+)", re.IGNORECASE),
    re.compile(r"^\s*(\d+)\s*[.)]"),  # "1." / "1)"
]

# USAMTS closes the problem set with a rule of asterisks ("**************")
# followed by submission instructions and a mailing address. That trailing
# furniture has no problem marker, so it binds to the last problem -- cut it.
_SEPARATOR_RE = re.compile(r"^\s*\*{5,}\s*$")

# --- Solution-packet structure ---
# A USAMTS solutions PDF ("PROBLEMS / SOLUTIONS / COMMENTS") separates problems
# by the bolded date marker "**3/1/12.**"; within a problem, each solution is a
# "**Solution k by <name>:**" header (sometimes emitted as LaTeX
# "\textbf{Solution k by ...}"), and the trailing "**Editor's Comment:**" is
# commentary, not a solution.
_PROBLEM_MARKER_RE = re.compile(
    r"^\s*(?:\*{1,2}|\\textbf\{)?\s*(\d+)\s*/\s*\d+\s*/\s*\d+\s*\.\s*(?:\*{1,2}|\})?\s*"
)
_SOLUTION_HEADER_RE = re.compile(
    r"^\s*(?:\*{2}|\\textbf\{)\s*Solution\s+\d+\s+by\b", re.IGNORECASE
)
_EDITOR_RE = re.compile(
    r"^\s*(?:\*{2}|\\textbf\{)\s*Editor.?s\s+Comment", re.IGNORECASE
)


def _cut_at_separator(text: str) -> str:
    """Return `text` truncated at the first ``**************`` rule line."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _SEPARATOR_RE.match(line):
            return "\n".join(lines[:i]).rstrip()
    return text


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
        if _EDITOR_RE.match(line):
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
        body = []
        for line in lines:
            if _EDITOR_RE.match(line):
                break
            body.append(line)
        text = "\n".join(body).strip()
        if text:
            blocks = [text]
    return blocks


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
        # The raw "<img>" tags survive line-based splitting; normalize each block's
        # to the reading-order sentinel so the pipeline can align them with DETR's
        # crops (see pipeline.inline_solution_figures).
        return {
            n: [normalize_img_placeholders(b) for b in _split_solution_blocks(lines)]
            for n, lines in bodies.items()
        }

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
                    trimmed = _cut_at_separator(el.text)
                    if trimmed != el.text:
                        el.text = trimmed
                        cut = True
                    if not el.text.strip():
                        continue  # element became empty
                kept.append(el)
            p.elements = kept
        return problems
