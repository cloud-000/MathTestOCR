"""FARML (Florida ARML): competition series parser.

On-disk layout (data dir is ``FARML/out``)::

    out/<year>/packet.pdf       # combined contest packet (intro, statements, answer key, solutions)
    out/<year>/test.pdf         # alternative combined packet filename (e.g. 2022)
    out/<year>/0514.pdf         # alternative combined packet filename (e.g. 2017)
    out/<year>/solutions.pdf    # standalone solutions PDF (e.g. 2018)
    out/<year>/team.pdf         # standalone event PDFs (e.g. 2016)
    out/<year>/indy.pdf
    out/<year>/relay.pdf
    out/<year>/answers.pdf

Discovery finds every contest under ``out/``. When a year folder contains a combined
packet PDF (e.g. ``packet.pdf``), that single PDF holds problem statements, an answer
key page, and worked solution pages. `test_pages` segments the PDF into statement pages
for `main.py parse` and solution/answer pages for `main.py solutions`.

Problem numbering maps event prefixes to sequential 1-based integer problem numbers:
  - Team Event (T1..T10)        -> 1..10
  - Individual Event (I1..I10)  -> 11..20
  - Relay Event (R1/1..R2/3)    -> 21..26
  - Tiebreaker Event (TB/TB1/TB2) -> 27..28
"""

import re
from copy import copy
from pathlib import Path
from typing_extensions import override

import pymupdf

from .. import config, pdf_io
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test, _natural_pages

PathClass = type(Path("."))


class FarmlSolutionPath(PathClass):
    is_solution = True


class FarmlAnswerPath(PathClass):
    is_answer = True


FARML_PATTERNS = [
    (re.compile(r"^\s*T([1-9]|10)\b", re.I), lambda m: (int(m.group(1)), m.end())),
    (re.compile(r"^\s*I([1-9]|10)\b", re.I), lambda m: (10 + int(m.group(1)), m.end())),
    (re.compile(r"^\s*R1[/.-]1\b", re.I), lambda m: (21, m.end())),
    (re.compile(r"^\s*R1[/.-]2\b", re.I), lambda m: (22, m.end())),
    (re.compile(r"^\s*R1[/.-]3\b", re.I), lambda m: (23, m.end())),
    (re.compile(r"^\s*R2[/.-]1\b", re.I), lambda m: (24, m.end())),
    (re.compile(r"^\s*R2[/.-]2\b", re.I), lambda m: (25, m.end())),
    (re.compile(r"^\s*R2[/.-]3\b", re.I), lambda m: (26, m.end())),
    (re.compile(r"^\s*R([1-6])\b", re.I), lambda m: (20 + int(m.group(1)), m.end())),
    (re.compile(r"^\s*Tiebreaker\s*2\b", re.I), lambda m: (28, m.end())),
    (re.compile(r"^\s*TB2\b", re.I), lambda m: (28, m.end())),
    (
        re.compile(
            r"^\s*(?:TB1?\b|Tiebreaker(?:\s*1)?\b(?!\s+EVENT\b))",
            re.I,
        ),
        lambda m: (27, m.end()),
    ),
]

_APPENDIX_MARKER_RE = re.compile(r"^\s*Appendix\s+to\s+T([1-9]|10)\b", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_FARML_YEAR_RE = re.compile(r"^FARML\s+20\d{2}$", re.I)
_EVENT_HEADING_RE = re.compile(
    r"^(?:TEAM|INDIVIDUAL|RELAY|TIEBREAKER)(?:\s*(?:/|&|AND)\s*TIEBREAKER)?\s+EVENTS?$",
    re.I,
)
_SOLUTION_HEADING_RE = re.compile(r"^(?:ANSWER\s+KEY|SOLUTIONS?)$", re.I)
_CLOSING_LINE_RE = re.compile(
    r"^(?:Good luck at ARML,? y[’']all!|Come to the Estimathon.*)$", re.I
)


def farml_match_marker(text: str):
    """Match a FARML problem marker (T1..T10, I1..I10, R1/1..R2/3, TB/Tiebreaker)."""
    for pat, handler in FARML_PATTERNS:
        m = pat.match(text)
        if m:
            return handler(m)
    return None


def farml_solution_match_marker(text: str):
    """FARML solution marker, including a labelled late appendix."""
    appendix = _APPENDIX_MARKER_RE.match(text)
    if appendix:
        return int(appendix.group(1)), appendix.end()
    return farml_match_marker(text)


def _plain_line(line: str) -> str:
    plain = _TAG_RE.sub(" ", line)
    plain = re.sub(r"[*_#]", "", plain)
    return re.sub(r"\s+", " ", plain).strip()


def _marker_numbers(
    text: str, matcher=farml_match_marker, *, source_text: bool = False
) -> set[int]:
    """Problem numbers visible in plain or HTML-table OCR/source text."""
    numbers = set()
    lines = [_plain_line(line) for line in text.splitlines()]
    for index, line in enumerate(lines):
        # The born-digital layer can put a lowercase recurrence variable such
        # as "r1 = a1" at line start. Printed FARML event labels are uppercase;
        # do not turn the variable into Relay problem 1 in the expected set.
        if source_text and re.match(r"^[tir]\d", line):
            continue
        # A few embedded PDF fonts split "Tiebreaker 2" into the three lines
        # "Tiebreak" / "er" / "2". Conversely, the section title is split as
        # "TIEBREAKER" / "EVENT" and must not count as TB1.
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if line.casefold() == "tiebreaker" and following.casefold() == "event":
            continue
        if (
            line.casefold() == "tiebreak"
            and following.casefold() == "er"
        ):
            suffix = lines[index + 2] if index + 2 < len(lines) else ""
            line = f"Tiebreaker {suffix}" if suffix in {"1", "2"} else "Tiebreaker"
        marker = matcher(line)
        if marker is not None:
            numbers.add(marker[0])
    return numbers


def _clean_farml_page(markdown: str) -> str:
    """Strip stable FARML page furniture without touching problem prose."""
    kept = []
    for line in markdown.splitlines():
        plain = _plain_line(line)
        if (
            _FARML_YEAR_RE.fullmatch(plain)
            or _EVENT_HEADING_RE.fullmatch(plain)
            or _SOLUTION_HEADING_RE.fullmatch(plain)
            or plain.casefold() == "(figures)"
            or _CLOSING_LINE_RE.fullmatch(plain)
        ):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _test_number_range(test: Test):
    """Global FARML problem range for a separately-discovered 2016 event."""
    test_id = test.id.casefold()
    if test_id.endswith("_team"):
        return range(1, 11)
    if test_id.endswith("_indy"):
        return range(11, 21)
    if test_id.endswith("_relay"):
        return range(21, 28)
    return None


def _get_page_splits(pdf_path: Path):
    """Return (statement_page_numbers, solution_page_numbers) for a FARML PDF."""
    doc = pymupdf.open(pdf_path)
    stmt_pages = []
    sol_pages = []
    seen_end_of_stmts = False
    in_power_event = False

    for i in range(len(doc)):
        page = doc.load_page(i)
        txt = page.get_text()
        txt_lower = txt.lower()
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        top_lines = [l.strip().lower() for l in lines[:15]]
        top_text = " ".join(top_lines)
        # Some FARML PDFs use a font whose text layer splits words into short
        # fragments ("po" / "wer" / "event"). Compact matching keeps page
        # classification deterministic without relying on OCR.
        top_compact = re.sub(r"[^a-z0-9]+", "", top_text)

        # FARML 2020 embeds a separately numbered Power Event between the Team
        # and Individual rounds, both in the statements and worked solutions.
        # It is not part of this adapter's T/I/R/TB numbering scheme.
        if in_power_event and any(
            label in top_compact
            for label in ("individualevent", "relayevent", "tiebreakerevent")
        ):
            in_power_event = False
        if "powerevent" in top_compact:
            in_power_event = True
        if in_power_event:
            continue

        # Skip intro page (page 1) if it carries no problem markers
        if (
            i == 0
            and ("welcome to farml" in txt_lower or "introduction" in top_text)
            and not re.search(r"\b(T1|I1)\b", txt)
        ):
            continue

        if "answer key" in top_text or (
            "solutions" in top_text
            and not any(
                k in top_text
                for k in [
                    "sum of the solutions",
                    "number of solutions",
                    "number of real solutions",
                ]
            )
        ):
            seen_end_of_stmts = True

        if seen_end_of_stmts:
            sol_pages.append(i + 1)
        else:
            stmt_pages.append(i + 1)

    doc.close()
    return stmt_pages, sol_pages


class FarmlSeries(Series):
    name = "farml"
    has_solutions = True
    has_answers = True

    @override
    def discover_tests(self, data_dir):
        """Discover FARML contests in `data_dir` (one per year folder)."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for year_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            combined = None
            for fname in ["packet.pdf", "test.pdf", "0514.pdf"]:
                if (year_dir / fname).exists():
                    combined = year_dir / fname
                    break
            if combined:
                tests.append(Test(id=year_dir.name, source=combined))
            elif (year_dir / "solutions.pdf").exists() and not any(
                (year_dir / f).exists() for f in ["team.pdf", "indy.pdf"]
            ):
                tests.append(Test(id=year_dir.name, source=year_dir / "solutions.pdf"))
            else:
                for pdf in sorted(year_dir.glob("*.pdf")):
                    if pdf.stem.lower() not in [
                        "answers",
                        "solutions",
                        "solution",
                        "packet",
                        "test",
                    ]:
                        tests.append(Test(id=f"{year_dir.name}_{pdf.stem}", source=pdf))
        return tests

    @override
    def test_pages(self, test: Test, workdir):
        """Return page-image paths for `test`.

        For combined contest PDFs, statement parsing gets only statement pages,
        while solution/answer parsing gets solution and answer key pages.
        """
        src = test.source
        if isinstance(src, (str, Path)):
            src = Path(src) if not isinstance(src, Path) else src
        if src.is_dir():
            return _natural_pages(src)
        if src.suffix.lower() == ".pdf":
            self._active_test_id = test.id
            self._shared_team_figure_regions = None
            stmt_pages, sol_pages = _get_page_splits(src)
            is_sol = getattr(src, "is_solution", False) or getattr(
                src, "is_answer", False
            )
            target_pages = set(sol_pages if is_sol else stmt_pages)

            out = Path(workdir)
            out.mkdir(parents=True, exist_ok=True)
            doc = pymupdf.open(src)
            paths = []
            expected_markers = []
            for page_num in range(len(doc)):
                p1_num = page_num + 1
                if target_pages and p1_num not in target_pages:
                    continue
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                image_path = out / f"page_{p1_num}.png"
                pix.save(image_path)
                paths.append(image_path)
                if not is_sol:
                    expected_markers.append(
                        _marker_numbers(page.get_text(), source_text=True)
                    )
                    compact = re.sub(
                        r"[^a-z0-9]+", "", page.get_text().casefold()
                    )
                    if test.id == "2020" and "teameventfigures" in compact:
                        marker_y = {}
                        for word in page.get_text("words"):
                            label = word[4].rstrip(".").upper()
                            if label in {"T5", "T8"}:
                                marker_y[label] = word[1]
                        if marker_y.keys() >= {"T5", "T8"}:
                            scale = pix.height / page.rect.height
                            # The shared DETR crop encloses the page frame. Use
                            # the printed labels as stable anchors and retain a
                            # little vertical padding around each diagram.
                            split = marker_y["T8"] - 95
                            self._shared_team_figure_regions = {
                                5: (
                                    max(0, marker_y["T5"] - 135) * scale,
                                    split * scale,
                                ),
                                8: (
                                    split * scale,
                                    min(page.rect.height, marker_y["T8"] + 125)
                                    * scale,
                                ),
                            }
            doc.close()
            if not is_sol:
                self._expected_statement_markers = expected_markers
            return paths
        raise ValueError(f"unsupported test source: {src}")

    @override
    def skip_page(self, text: str) -> bool:
        """Skip intro, answer key, or solutions headers during basic PDF page filtering."""
        if not text:
            return False
        t_lower = text.lower()
        lines = [l.strip().lower() for l in text.splitlines() if l.strip()]
        top_text = " ".join(lines[:15])
        if "welcome to farml" in t_lower or "introduction" in top_text:
            return True
        if "answer key" in top_text or "solutions" in top_text:
            return True
        return False

    @override
    def match_marker(self):
        return farml_match_marker

    @override
    def solution_match_marker(self):
        return farml_solution_match_marker

    @override
    def layout_options(self):
        return config.LayoutOptions(
            inline_figures=True,
            split_marker_table_rows=True,
            backreference_problem_markers=True,
            equation_text_overlap=0.3,
            solution_equation_text_overlap=True,
        )

    @override
    def clean_statement_markdown(self, page_index: int, markdown: str) -> str:
        return _clean_farml_page(markdown)

    @override
    def validate_statement_markdown(self, page_index: int, markdown: str) -> bool:
        expected = getattr(self, "_expected_statement_markers", ())
        if page_index >= len(expected) or not expected[page_index]:
            return True
        return expected[page_index].issubset(_marker_numbers(markdown))

    @override
    def clean_solution_markdown(self, page_index: int, markdown: str) -> str:
        # Cache-only reparsing still contains the old 2020 Power Event pages,
        # so keep a page-sequential state in addition to excluding them during
        # fresh rendering in `_get_page_splits`.
        if page_index == 0:
            self._solution_in_power_event = False
        plain = " ".join(
            _plain_line(line).casefold() for line in markdown.splitlines()[:20]
        )
        if self._solution_in_power_event and re.search(
            r"\b(?:individual|relay|tiebreaker)\s+events?\b", plain
        ):
            self._solution_in_power_event = False
        if "power event" in plain:
            self._solution_in_power_event = True
        if self._solution_in_power_event or "answer key" in plain:
            return ""
        return _clean_farml_page(markdown)

    @override
    def postprocess(self, problems):
        """Split FARML 2020's shared T5/T8 figure page into two crops."""
        regions = getattr(self, "_shared_team_figure_regions", None)
        if getattr(self, "_active_test_id", None) != "2020" or not regions:
            return problems
        by_number = {problem.number: problem for problem in problems}
        p5 = by_number.get(5)
        p8 = by_number.get(8)
        if p5 is None or p8 is None:
            return problems
        candidates = [
            element
            for element in p5.elements
            if element.kind == "image"
            and element.crop is not None
            and element.crop.width >= 1000
            and element.crop.height >= 900
        ]
        if len(candidates) != 1:
            return problems
        original = candidates[0]
        p5.elements.remove(original)
        box_x0, box_y0, box_x1, _ = original.box
        for number, (global_y0, global_y1) in regions.items():
            local_y0 = max(0, int(round(global_y0 - box_y0)))
            local_y1 = min(original.crop.height, int(round(global_y1 - box_y0)))
            if local_y1 <= local_y0:
                continue
            element = copy(original)
            element.box = [box_x0, box_y0 + local_y0, box_x1, box_y0 + local_y1]
            element.crop = original.crop.crop(
                (0, local_y0, original.crop.width, local_y1)
            )
            by_number[number].elements.append(element)
        return problems

    @override
    def solution_source(self, test: Test):
        """Return solution source for `test` (sibling solutions.pdf or combined PDF)."""
        parent = Path(test.source).parent
        sol = parent / "solutions.pdf"
        if not sol.exists():
            sol = parent / "solution.pdf"
        if sol.exists():
            return FarmlSolutionPath(sol)
        return FarmlSolutionPath(test.source)

    @override
    def answer_source(self, test: Test):
        """Return answer source for `test` (sibling answers.pdf or solution_source)."""
        parent = Path(test.source).parent
        ans = parent / "answers.pdf"
        if ans.exists():
            return FarmlAnswerPath(ans)
        return self.solution_source(test)

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Parse answer key entries from answer document markdown into {problem_number: answer}."""
        answers: dict[int, str] = {}
        for md in pages_markdown:
            if "answer key" not in md.lower():
                continue
            curr_p = None
            curr_ans = []
            for line in md.splitlines():
                sline = line.strip()
                if not sline:
                    continue
                m = farml_match_marker(sline)
                if m:
                    if curr_p is not None and curr_ans:
                        answers[curr_p] = " ".join(curr_ans).strip()
                    curr_p = m[0]
                    rest = sline[m[1] :].strip()
                    curr_ans = [rest] if rest else []
                elif curr_p is not None:
                    if any(
                        h in sline.lower()
                        for h in [
                            "farml",
                            "answer key",
                            "team event",
                            "individual event",
                            "relay event",
                            "tiebreaker event",
                            "solutions",
                        ]
                    ):
                        if curr_ans:
                            answers[curr_p] = " ".join(curr_ans).strip()
                            curr_p = None
                            curr_ans = []
                    else:
                        curr_ans.append(sline)
            if curr_p is not None and curr_ans:
                answers[curr_p] = " ".join(curr_ans).strip()
        allowed = _test_number_range(test)
        if allowed is not None:
            allowed = set(allowed)
            answers = {
                number: answer
                for number, answer in answers.items()
                if number in allowed
            }
        return answers

    @override
    def parse_solutions(self, full_text: str, test: Test = None) -> dict:
        """Segment solution document OCR text into {problem_number: solution_text}."""
        opts = self.layout_options()
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(
            full_text,
            self.solution_match_marker(),
            split_marker_table_rows=opts.split_marker_table_rows,
            backreference_problem_markers=opts.backreference_problem_markers,
        ):
            if item["problem"] is None:
                continue
            if item["kind"] == "text":
                grouped.setdefault(item["problem"], []).append(item["text"])
            elif item["kind"] == "image":
                grouped.setdefault(item["problem"], []).append(FIGURE_PLACEHOLDER)
        return {n: "\n".join(parts) for n, parts in grouped.items()}

    @override
    def postprocess_solutions(
        self, solutions: dict, statements: dict, test: Test = None
    ) -> dict:
        """Keep only solutions belonging to this output test."""
        allowed = {int(number) for number in statements} if statements else set()
        if not allowed and test is not None:
            event_range = _test_number_range(test)
            allowed = set(event_range or ())
        return (
            {number: value for number, value in solutions.items() if number in allowed}
            if allowed
            else solutions
        )

    @override
    def postprocess_solution_figures(
        self, figures: dict, test: Test = None, full_text: str = ""
    ) -> dict:
        """Do not copy the shared 2016 packet's other events into each test."""
        event_range = _test_number_range(test) if test is not None else None
        if event_range is None:
            return figures
        allowed = set(event_range)
        return {number: value for number, value in figures.items() if number in allowed}
