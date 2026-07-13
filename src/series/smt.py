"""SMT (Stanford Math Tournament) tests and solutions.

On-disk layout (data dir is ``SMT/out``)::

    out/<tournament>/<year>/<subject>/test.pdf
    out/<tournament>/<year>/<subject>/solutions.pdf

``<tournament>`` is ``SMT``, ``ASMT``, or ``SM3``. Test IDs mirror their path
joined by underscores: ``SMT_2024_algebra``, ``ASMT_2016_geometry``.

The ``power`` round is skipped by `discover_tests` (a proof-based team round we
don't parse). Every other round numbers its problems plainly (``1.``, ``2.``,
...), so the default marker matcher fits. Unlike BMT, the solutions PDF prints no ``Answer:`` line:
each solution restates the problem, gives the worked solution under a
``Solution:`` label, and ends with the final answer in a ``\\boxed{...}``.
`parse_solutions` keeps the text from ``Solution:`` onward; `parse_answers` reads
the boxed value out of each problem's block (the last box -- the final answer --
when a solution boxes an intermediate result too). Problems whose solution boxes
nothing (proof-style team/power questions) are simply omitted from the key.
"""

import re
from pathlib import Path

from typing_extensions import override

from .. import config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test

# OCR sometimes renders \boxed{X} as a <box>X</box> tag; both are answer boxes.
_BOX_TAG_RE = re.compile(r"<box>\s*(.*?)\s*</box>", re.I | re.S)
# The "Solution:" label arrives markdown-emphasized ("**Solution:** ..."), so
# tolerate leading emphasis/heading chars and a "**" between the word and colon.
_EMPH = r"[*_#]{0,3}"
_SOLUTION_RE = re.compile(
    rf"^\s*{_EMPH}\s*Solution(?:\s+\d+)?\b\s*{_EMPH}\s*:?\s*(.*)$", re.IGNORECASE
)


class SmtSeries(Series):
    name = "smt"
    has_solutions = True
    has_answers = True

    @override
    def discover_tests(self, data_dir):
        """Discover every SMT ``test.pdf`` recursively, minus the power round.

        The full parent path forms the ID, avoiding collisions across the three
        tournaments (``SMT``, ``ASMT``, ``SM3``) and their many rounds.
        """
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        return [
            Test(id="_".join(pdf.relative_to(root).parts[:-1]), source=pdf)
            for pdf in sorted(root.glob("**/test.pdf"))
            if pdf.parent.name != "power"  # power round: not parsed (see module docstring)
        ]

    @override
    def layout_options(self):
        """Keep figures inline, and drop the Stanford shield logo in the header.

        Every page carries a small shield logo (image center at ~0.05 of page
        height) beside the "Stanford Math Tournament" title; DETR reads it as a
        Picture. The first problem marker is at ~0.09, so a 0.07 cutoff drops the
        logo without reaching any real figure. Without this the logo binds to
        problem 1 (the title text above it is a left-margin "start" that defeats
        the drop-above-first-problem guard; see pipeline._assign_pictures).
        """
        return config.LayoutOptions(inline_figures=True, header_picture_frac=0.07)

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

        Figure placeholders are retained so DETR crops stay inline.
        """
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(full_text, self.match_marker()):
            if item["problem"] is None:
                continue
            value = item["text"] if item["kind"] == "text" else FIGURE_PLACEHOLDER
            grouped.setdefault(item["problem"], []).append(value)
        return {n: _solution_body("\n".join(parts)) for n, parts in grouped.items()}

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Read each problem's boxed final answer from the solutions document.

        Problems are grouped into per-problem blocks by the same markers the
        solution/figure passes use, then each block's answer is the last boxed
        value in it (the final answer follows any boxed intermediate). A block
        with no box -- a proof-style problem, or an OCR miss -- is omitted,
        yielding a partial key rather than a guessed one.
        """
        answers: dict[int, str] = {}
        grouped: dict[int, list[str]] = {}
        for item in parse_layout("\n\n".join(pages_markdown), self.match_marker()):
            if item["problem"] is None or item["kind"] != "text":
                continue
            grouped.setdefault(item["problem"], []).append(item["text"])
        for n, parts in grouped.items():
            value = _boxed_answer("\n".join(parts))
            if value:
                answers[n] = value
        return answers


def _solution_body(block: str) -> str:
    """Return just the worked solution from one restated-problem block.

    Everything up to and including the first ``Solution:`` label is the restated
    statement; drop it. With no label present (unexpected layout / OCR miss),
    keep the whole block so nothing is lost.
    """
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _SOLUTION_RE.match(line)
        if match is not None:
            # Drop a "**" emphasis closer the label's colon left behind.
            first = re.sub(r"^[*_]+\s*", "", match.group(1).strip())
            kept = ([first] if first else []) + lines[index + 1 :]
            return "\n".join(kept).strip()
    return block.strip()


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


def _boxed_answer(block: str):
    """The last boxed value in `block` (the final answer), cleaned, or None."""
    value = None
    for box in _iter_boxes(block):
        box = box.strip().strip("$").strip().rstrip(".").strip()
        if box:
            value = box
    return value
