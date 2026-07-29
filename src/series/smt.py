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
_ANSWER_RE = re.compile(
    rf"^\s*{_EMPH}\s*(?:Answer|Ans)\b\s*{_EMPH}\s*:?\s*(.*)$", re.IGNORECASE
)
_INLINE_ANSWER_RE = re.compile(
    r"(?:\*{0,2}\s*)?(?:Answer|Ans)\b\s*(?:\*{0,2}\s*)?:?\s*"
    r"(.*?)(?=(?:\*{0,2}\s*)?Solution\b|$)",
    re.IGNORECASE,
)
_CODED_MARKER_RE = re.compile(
    r"^\s*(?:[*_#]{0,3}[A-Z]{1,4}\d{1,3}[*_#]{0,3}\s+)?"
    r"(?:Problem|Question|Q)?\s*(\d+)\s*[.:]\s*",
    re.IGNORECASE,
)
_TREELAY_MARKER_RE = re.compile(r"^\s*(\d)\s*([A-H])\.\s*", re.IGNORECASE)
_TREELAY_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TREELAY_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_HTML_RE = re.compile(r"<[^>]+>")


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
        return config.LayoutOptions(
            inline_figures=True,
            header_picture_frac=0.07,
            # Several solution packets put the next bare ``N.`` directly after
            # a display equation.  Without this, parse_layout absorbs the next
            # problem into the preceding solution.
            split_glued_bare_markers=True,
        )

    @override
    def test_pages(self, test, workdir):
        """Record born-digital solution starts for OCR completeness checks."""
        pages = super().test_pages(test, workdir)
        self._active_test_id = test.id
        self._expected_solution_starts = ()
        if Path(test.source).name.lower() == "solutions.pdf":
            import pymupdf

            matcher = self.match_marker()
            with pymupdf.open(test.source) as document:
                expected = []
                last = 0
                for page in document:
                    starts = set()
                    for line in page.get_text().splitlines():
                        marker = matcher(line)
                        if marker is not None and marker[0] == last + 1:
                            starts.add(marker[0])
                            last = marker[0]
                    expected.append(frozenset(starts))
                self._expected_solution_starts = tuple(expected)
        return pages

    @override
    def validate_solution_markdown(self, page_index, markdown):
        """Reject a clean-looking solution OCR page that omitted a source start."""
        expected = getattr(self, "_expected_solution_starts", ())
        if page_index >= len(expected) or not expected[page_index]:
            return True
        found = {
            marker[0]
            for line in markdown.splitlines()
            if (marker := self.match_marker()(line.lstrip("*_# "))) is not None
        }
        return expected[page_index].issubset(found)

    @override
    def match_marker(self):
        """Accept both ordinary starts and SMT's coded solution starts.

        Team solution packets prefix starts with identifiers such as ``JZ03``;
        treating those as furniture used to make every block invisible.
        Treelay uses ``1A`` through ``5H`` instead, mapped column-major to the
        pipeline's integer keys 1 through 40.
        """
        treelay = getattr(self, "_active_test_id", "").endswith("_treelay")

        def matcher(text):
            if treelay:
                m = _TREELAY_MARKER_RE.match(text)
                if m:
                    return ((ord(m.group(2).upper()) - ord("A")) * 5 + int(m.group(1)), m.end())
            m = _CODED_MARKER_RE.match(text)
            return (int(m.group(1)), m.end()) if m else None

        return matcher

    @override
    def clean_statement_markdown(self, page_index, markdown):
        """Keep one printable copy of each duplicated Treelay answer form."""
        if not getattr(self, "_active_test_id", "").endswith("_treelay"):
            return markdown
        # Each page prints the same question twice.  The OCR reliably separates
        # the copies with a horizontal rule, so discard the second before marker
        # grouping can append it to the first problem.
        return re.split(r"\n\s*(?:---|\*\*\*)\s*\n", markdown, maxsplit=1)[0]

    @override
    def solution_source(self, test):
        sol = test.source.parent / "solutions.pdf"
        return sol if sol.exists() else None

    @override
    def answer_source(self, test):
        return self.solution_source(test)

    @override
    def parse_solutions(self, full_text, test: Test = None):
        """Drop each restated statement and keep only its worked solution.

        Figure placeholders are retained so DETR crops stay inline.
        """
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(
            full_text,
            self.match_marker(),
            split_glued_bare_markers=self.layout_options().split_glued_bare_markers,
        ):
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
        if test.id.endswith("_treelay"):
            return _treelay_answers("\n".join(pages_markdown))

        answers: dict[int, str] = {}
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(
            "\n\n".join(pages_markdown),
            self.match_marker(),
            split_glued_bare_markers=self.layout_options().split_glued_bare_markers,
        ):
            if item["problem"] is None or item["kind"] != "text":
                continue
            grouped.setdefault(item["problem"], []).append(item["text"])
        for n, parts in grouped.items():
            value = _answer_value("\n".join(parts))
            if value:
                answers[n] = value
        statement_numbers = _statement_numbers(test, self.match_marker())
        return (
            {n: value for n, value in answers.items() if n in statement_numbers}
            if statement_numbers is not None
            else answers
        )

    @override
    def postprocess_solutions(self, solutions, statements, test: Test = None):
        """Never export a second solution section absent from the test itself."""
        if not statements:
            return solutions
        statement_numbers = {int(number) for number in statements}
        return {
            number: value
            for number, value in solutions.items()
            if number in statement_numbers
        }


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
    # 2011-era packets print a restated question, an ``Answer:`` line, and then
    # immediately begin the derivation without a ``Solution:`` label.
    for index, line in enumerate(lines):
        if _ANSWER_RE.match(line):
            return "\n".join(lines[index + 1 :]).strip()
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


def _clean_answer(value: str):
    """Normalize a printed answer without guessing at prose-only conclusions."""
    value = re.sub(r"^[*_\s]+|[*_\s]+$", "", value)
    value = value.strip().strip("$").strip().rstrip(".").strip()
    return value or None


def _answer_value(block: str):
    """Prefer SMT's explicit ``Answer:`` label, falling back to a final box."""
    for line in block.splitlines():
        match = _ANSWER_RE.match(line)
        if match and (value := _clean_answer(match.group(1))):
            return value
        # Some compact OCR pages put ``statement. Answer: X Solution: ...`` on
        # one line, so labels need not be line-leading.
        match = _INLINE_ANSWER_RE.search(line)
        if match and (value := _clean_answer(match.group(1))):
            return value
    return _boxed_answer(block)


def _treelay_answers(markdown: str) -> dict[int, str]:
    """Read the 2025 Treelay 5-by-8 answer grid into keys 1 through 40."""
    answers: dict[int, str] = {}
    for row in _TREELAY_ROW_RE.findall(markdown):
        cells = [_HTML_RE.sub(" ", cell).strip() for cell in _TREELAY_CELL_RE.findall(row)]
        if len(cells) < 9 or not cells[0].isdigit() or not 1 <= int(cells[0]) <= 5:
            continue
        row_number = int(cells[0])
        for column, value in enumerate(cells[1:9]):
            value = re.sub(r"\s+", " ", value).strip()
            if value:
                answers[column * 5 + row_number] = value
    return answers


def _statement_numbers(test: Test, match_marker):
    """Return the consecutive main-question numbers printed in a test PDF.

    A few modern solution packets append a distinct section whose numbering
    restarts at 1.  The statement PDF is the authority for whether that section
    belongs in this output.  Requiring consecutive forward starts ignores
    numbered subparts (including a line-leading ``100``) without hard-coding a
    round size.
    """
    source = Path(test.source)
    if source.suffix.lower() != ".pdf" or not source.exists():
        return None
    import pymupdf

    numbers = []
    last = 0
    offset = 0
    with pymupdf.open(source) as document:
        for page in document:
            for line in page.get_text().splitlines():
                marker = match_marker(line)
                if marker is None:
                    continue
                raw_number = marker[0]
                number = raw_number + offset
                if number == last + 1:
                    numbers.append(number)
                    last = number
                elif (
                    raw_number == 1
                    and last > 0
                    and re.search(r"\bprove\b", line, re.IGNORECASE)
                ):
                    # Team packets may append a genuine proof round after the
                    # short-answer round.  It restarts at 1, unlike numbered
                    # subparts; make its output keys continue after the prior
                    # section just as parse_layout does.
                    offset = last
                    numbers.append(offset + raw_number)
                    last += raw_number
    return set(numbers)
