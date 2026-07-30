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
    (re.compile(r"^\s*(?:TB1?|Tiebreaker\s*1?)\b", re.I), lambda m: (27, m.end())),
]


def farml_match_marker(text: str):
    """Match a FARML problem marker (T1..T10, I1..I10, R1/1..R2/3, TB/Tiebreaker)."""
    for pat, handler in FARML_PATTERNS:
        m = pat.match(text)
        if m:
            return handler(m)
    return None


def _get_page_splits(pdf_path: Path):
    """Return (statement_page_numbers, solution_page_numbers) for a FARML PDF."""
    doc = pymupdf.open(pdf_path)
    stmt_pages = []
    sol_pages = []
    seen_end_of_stmts = False

    for i in range(len(doc)):
        page = doc.load_page(i)
        txt = page.get_text()
        txt_lower = txt.lower()
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        top_lines = [l.strip().lower() for l in lines[:15]]
        top_text = " ".join(top_lines)

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
            stmt_pages, sol_pages = _get_page_splits(src)
            is_sol = getattr(src, "is_solution", False) or getattr(
                src, "is_answer", False
            )
            target_pages = set(sol_pages if is_sol else stmt_pages)

            out = Path(workdir)
            out.mkdir(parents=True, exist_ok=True)
            doc = pymupdf.open(src)
            paths = []
            for page_num in range(len(doc)):
                p1_num = page_num + 1
                if target_pages and p1_num not in target_pages:
                    continue
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                image_path = out / f"page_{p1_num}.png"
                pix.save(image_path)
                paths.append(image_path)
            doc.close()
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
    def layout_options(self):
        return config.LayoutOptions(inline_figures=True)

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
                if not any(
                    farml_match_marker(line.strip()) for line in md.splitlines()
                ):
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
        return answers

    @override
    def parse_solutions(self, full_text: str, test: Test = None) -> dict:
        """Segment solution document OCR text into {problem_number: solution_text}."""
        grouped: dict[int, list[str]] = {}
        for item in parse_layout(full_text, self.match_marker()):
            if item["problem"] is None:
                continue
            if item["kind"] == "text":
                grouped.setdefault(item["problem"], []).append(item["text"])
            elif item["kind"] == "image":
                grouped.setdefault(item["problem"], []).append(FIGURE_PLACEHOLDER)
        return {n: "\n".join(parts) for n, parts in grouped.items()}
