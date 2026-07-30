"""BMT (Berkeley Math Tournament) tests and solutions.

On-disk layout (data dir is ``BMT/out``)::

    out/<tournament>/<year>/<subject>/test.pdf
    out/<tournament>/<year>/<subject>/solutions.pdf

``<tournament>`` is ``bmt``, ``bmmt``, or ``bmmt-online``. Test IDs mirror their
path joined by underscores: ``bmt_2024_algebra``, ``bmmt_2018_speed``.

The ``power``, ``relay``, and ``puzzle`` rounds (including puzzle variants like
``puzzle-us-iran``) are skipped by `discover_tests` -- rounds we don't parse.
Every other round numbers its problems plainly (``1.``, ``2.``, ...), so the default
marker matcher fits. Each solutions PDF restates a problem, prints its final
answer on an ``Answer:`` line, then gives the worked solution under a
``Solution:`` label -- the same shape as HMMT, so `parse_answers` reads the
``Answer:`` value and `parse_solutions` keeps only the text from ``Solution:``
onward.
"""

import re
from pathlib import Path

from typing_extensions import override

from .. import answer_llm, config
from .base import Series, Test, numbered_answers_in_line, strip_solution_page_furniture
from .smt import _boxed_answer

# Labels arrive markdown-emphasized from the OCR ("**Answer:** 45",
# "**Answer: 89**", "**Solution:** ..."), so each pattern tolerates leading
# emphasis/heading chars and (for the label itself) a "**" between the word and
# its colon. The captured value is cleaned by _clean_answer.
_EMPH = r"[*_#]{0,3}"
_ANSWER_RE = re.compile(rf"^\s*{_EMPH}\s*(?:Answer|Ans)\b\s*{_EMPH}\s*:?\s*(.*)$", re.IGNORECASE)
_PROPOSED_RE = re.compile(rf"^\s*{_EMPH}\s*Proposed\s+by\b", re.IGNORECASE)
_SOLUTION_RE = re.compile(
    rf"^\s*{_EMPH}\s*Solution(?:\s+\d+)?\b\s*{_EMPH}\s*:?\s*(.*)$", re.IGNORECASE
)
_RUNNING_HEADER_RE = re.compile(
    r"^\s*B(?:ERKELEY\s+MINI\s+|M)?MT\s+20\d{2}\b.*$", re.IGNORECASE
)
_RUNNING_TITLE_RE = re.compile(
    r"^(?:"
    r"(?:algebra|analysis|calculus|combinatorics|discrete|geometry|general|"
    r"number\s+theory|pacer|speed|team|individual|guts|tournament)\s+"
    r"(?:test|round|tiebreaker)(?:\s+and)?\s+(?:solutions?|answer\s+key)"
    r"|(?:analysis|individual|team)\s+(?:solutions?|answer\s+key)"
    r"|(?:geometry|discrete)\s+round"
    r"|bmmt\b.*"
    r"|(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2}(?:\s*[–—-]\s*\d{1,2})?,?\s+20\d{2}"
    r"|page\s+\d+"
    r")$",
    re.IGNORECASE,
)
_RUNNING_STATEMENT_FURNITURE_RE = re.compile(
    r"^(?:"
    r"(?:algebra|analysis|calculus|combinatorics|discrete|geometry|general|"
    r"number\s+theory|pacer|speed|team|individual|guts|tournament)\s+"
    r"(?:test|round|tiebreaker)(?:\s+(?:january|february|march|april|may|"
    r"june|july|august|september|october|november|december)\s+\d{1,2}"
    r"(?:\s*[–—-]\s*\d{1,2})?,?\s+20\d{2})?"
    r"|do\s+not\s+discuss\s+or\s+distribute\s+online\s+until\b.*"
    r"|page\s+\d+"
    r")$",
    re.IGNORECASE,
)
_TOURNAMENT_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.\s*)?(?:\*{1,2}\s*)?"
    r"(?:ROUND\b|CHAMPIONSHIP\s+ROUND|CONSOLATION\s+ROUND)\b",
    re.IGNORECASE,
)
_TIEBREAKER_RE = re.compile(r"\bTIEBREAKER\b", re.IGNORECASE)


def _solution_blocks(full_text: str, match_marker) -> dict[int, list[str]]:
    """Split BMT solution packets without promoting numbered working to problems.

    Modern packets label every real block with ``Answer`` or ``Solution``.  A
    numbered list inside a worked solution does not, so a candidate marker is
    accepted only when its following block contains one of those labels.  The
    older unlabeled packets retain the historical, marker-only behavior.
    """
    lines = [line.strip() for line in full_text.splitlines()]
    candidates = []
    for index, line in enumerate(lines):
        probe = line.lstrip("*_# ")
        marker = match_marker(probe)
        if marker is not None:
            candidates.append((index, marker[0], marker[1], probe))
    labelled = any(_ANSWER_RE.match(line) or _SOLUTION_RE.match(line) for line in lines)
    accepted = []
    for pos, (index, number, end, probe) in enumerate(candidates):
        next_index = candidates[pos + 1][0] if pos + 1 < len(candidates) else len(lines)
        if labelled:
            window = lines[index + 1 : next_index]
            if not any(_ANSWER_RE.match(line) or _SOLUTION_RE.match(line) for line in window):
                continue
        accepted.append((index, number, end, probe))

    grouped: dict[int, list[str]] = {}
    for pos, (index, number, end, probe) in enumerate(accepted):
        next_index = accepted[pos + 1][0] if pos + 1 < len(accepted) else len(lines)
        first = probe[end:].strip()
        block = ([first] if first else []) + lines[index + 1 : next_index]
        grouped.setdefault(number, []).append("\n".join(block).strip())
    return grouped

# Rounds we don't parse: power (proof-based team round) and the relay/puzzle
# rounds, whose formats don't fit the per-problem pipeline. Matched on the
# subject folder name, including puzzle variants ("puzzle-us-iran").
_SKIP_SUBJECTS = {"power", "relay", "partner"}


def _skip_subject(subject: str) -> bool:
    return subject in _SKIP_SUBJECTS or subject.startswith("puzzle")


def _is_multi_round(test_id: str) -> bool:
    return test_id.endswith("_tournament")


def _clean_answer(value: str) -> str:
    """Strip markdown emphasis, surrounding ``$`` math delimiters, and a trailing
    period from a captured answer value ("** 6072", "89**", "$\\frac{2}{3}$")."""
    value = re.sub(r"^[*_\s]+|[*_\s]+$", "", value)
    if len(value) > 1 and value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    return value.rstrip(".").strip()


def _tournament_statement_blocks(test: Test) -> dict[int, str]:
    """Read the born-digital 2012 packet into its global question keys."""
    if not _is_multi_round(test.id) or Path(test.source).suffix.lower() != ".pdf":
        return {}
    import pymupdf

    blocks: dict[int, str] = {}
    offset = 0
    with pymupdf.open(test.source) as document:
        for page in document:
            text = page.get_text()
            matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s+", text))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                blocks[int(match.group(1)) + offset] = text[match.end() : end].strip()
            offset += len(matches)
    return blocks


class BmtSeries(Series):
    name = "bmt"
    has_solutions = True
    has_answers = True

    @override
    def match_marker(self):
        """Match BMT problem markers, including category-prefixed markers like ``GG31 1.`` or ``**LL28** 14.``."""
        _BMT_MARKER_RE = re.compile(
            r"^\s*(?:[*_#]{0,3}[A-Z0-9]{3,6}[*_#]{0,3}\s+)?"
            r"(?:Problem|Question|Q)?\s*(\d+)\s*[.:]\s*",
            re.IGNORECASE,
        )

        def matcher(text: str):
            m = _BMT_MARKER_RE.match(text)
            return (int(m.group(1)), m.end()) if m else None

        return matcher

    @override
    def discover_tests(self, data_dir):
        """Discover every BMT ``test.pdf`` recursively, minus the skipped rounds.

        The full parent path forms the ID, avoiding collisions across the three
        tournaments (``bmt``, ``bmmt``, ``bmmt-online``) and their many rounds.
        The power/relay/puzzle/partner rounds are excluded (see `_skip_subject`).
        """
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        return [
            Test(id="_".join(pdf.relative_to(root).parts[:-1]), source=pdf)
            for pdf in sorted(root.glob("**/test.pdf"))
            if not _skip_subject(pdf.parent.name)
        ]

    @override
    def layout_options(self):
        """Keep figures inline while fencing BMT's running header logo."""
        return config.LayoutOptions(
            inline_figures=True,
            header_picture_frac=0.09,
            strict_section_restarts=True,
            split_glued_bare_markers=True,
        )

    @override
    def test_pages(self, test: Test, workdir):
        # Layout behavior is intentionally test-aware: the tournament packet
        # restarts its printed numbering once per round.
        self._active_test_id = test.id
        self._page_offsets = self._source_page_offsets(test)
        self._solution_round_offsets = self._source_solution_round_offsets(test)
        return super().test_pages(test, workdir)

    def _source_solution_round_offsets(self, test: Test):
        """Map each 2012 solution page's local rounds to global question keys.

        The solution packet is born-digital but its pages may contain the end
        of one round and the beginning of the next.  Page offsets alone would
        therefore renumber the continuation incorrectly.  Read the centered
        round headings from the PDF text layer, retaining both the offset at
        each page start and every in-page change of round.
        """
        if (
            not _is_multi_round(test.id)
            or Path(test.source).name.casefold() != "solutions.pdf"
        ):
            return None
        import pymupdf

        page_offsets = []
        current = 0
        round_index = 0
        with pymupdf.open(test.source) as document:
            for page in document:
                initial = current
                headings = []
                for block in sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0])):
                    text = " ".join(block[4].split())
                    # Actual round labels are centered; numbered question and
                    # proof-list lines live at the left margin.
                    if block[0] > 150 and _TOURNAMENT_HEADING_RE.match(text):
                        current = round_index * 6
                        headings.append(current)
                        round_index += 1
                page_offsets.append((initial, headings))
        return page_offsets

    def _source_page_offsets(self, test: Test) -> list[int] | None:
        """Return global-number offsets for a packet that restarts each round."""
        if not _is_multi_round(test.id) or Path(test.source).suffix.lower() != ".pdf":
            return None
        import pymupdf

        offsets = []
        total = 0
        with pymupdf.open(test.source) as document:
            for page in document:
                offsets.append(total)
                # The tournament PDFs are born-digital and print each actual
                # question in the left margin.  This deliberately ignores
                # numbered prose embedded farther into a statement.
                starts = re.findall(r"(?m)^\s*(\d+)\.\s+", page.get_text())
                total += len(starts)
        return offsets

    def _renumber_page(self, page_index: int, markdown: str) -> str:
        offsets = getattr(self, "_page_offsets", None)
        if offsets is None or page_index >= len(offsets):
            return markdown
        offset = offsets[page_index]
        if not offset:
            return markdown
        matcher = self.match_marker()
        rewritten = []
        for line in markdown.splitlines(keepends=True):
            probe = line.lstrip("*_# ")
            marker = matcher(probe)
            if marker is None:
                rewritten.append(line)
                continue
            prefix = probe[: marker[1]]
            replaced = re.sub(
                r"(\d+)(\s*[.:]\s*)$",
                lambda m: f"{int(m.group(1)) + offset}{m.group(2)}",
                prefix,
            )
            rewritten.append(line[: len(line) - len(probe)] + replaced + probe[marker[1] :])
        return "".join(rewritten)

    def _renumber_tournament_solution_page(self, page_index: int, markdown: str) -> str:
        """Flatten a tournament solution page using its source-validated rounds."""
        offsets = getattr(self, "_solution_round_offsets", None)
        if offsets is None or page_index >= len(offsets):
            return markdown
        offset, heading_offsets = offsets[page_index]
        heading_index = 0
        matcher = self.match_marker()
        rewritten = []
        for line in markdown.splitlines(keepends=True):
            plain = line.strip("\n")
            if _TOURNAMENT_HEADING_RE.match(plain.lstrip("*_# ")):
                if heading_index < len(heading_offsets):
                    offset = heading_offsets[heading_index]
                    heading_index += 1
                rewritten.append(line)
                continue
            probe = line.lstrip("*_# ")
            marker = matcher(probe)
            if marker is None:
                rewritten.append(line)
                continue
            prefix = probe[: marker[1]]
            replaced = re.sub(
                r"(\d+)(\s*[.:]\s*)$",
                lambda m: f"{int(m.group(1)) + offset}{m.group(2)}",
                prefix,
            )
            rewritten.append(line[: len(line) - len(probe)] + replaced + probe[marker[1] :])
        return "".join(rewritten)

    @staticmethod
    def _strip_running_furniture(markdown: str) -> str:
        """Remove repeated BMT headers without touching a problem statement."""
        return strip_solution_page_furniture(
            markdown, line_patterns=(_RUNNING_HEADER_RE, _RUNNING_TITLE_RE)
        )

    @staticmethod
    def _strip_statement_furniture(markdown: str) -> str:
        return strip_solution_page_furniture(
            markdown,
            line_patterns=(
                _RUNNING_HEADER_RE,
                _RUNNING_TITLE_RE,
                _RUNNING_STATEMENT_FURNITURE_RE,
            ),
        )

    @override
    def clean_statement_markdown(self, page_index: int, markdown: str) -> str:
        # This PDF appends a separate tiebreaker after the 20-question
        # individual round.  Once its heading appears, no following text
        # belongs to the selected packet (otherwise it is swallowed by #20).
        if getattr(self, "_active_test_id", "") == "bmmt_2017_individual":
            markdown = markdown.split("TIEBREAKER", 1)[0]
        return self._strip_statement_furniture(self._renumber_page(page_index, markdown))

    @override
    def clean_reconstructed_statement(self, markdown: str) -> str:
        return self._strip_statement_furniture(markdown)

    @override
    def clean_solution_markdown(self, page_index: int, markdown: str) -> str:
        # The 2017 individual solutions PDF appends its separate tiebreaker
        # packet.  It restarts at 1 and is not part of the individual test.
        if getattr(self, "_active_test_id", "") == "bmmt_2017_individual":
            kept = []
            for line in markdown.splitlines():
                if _TIEBREAKER_RE.search(line):
                    break
                kept.append(line)
            markdown = "\n".join(kept)
        markdown = self._strip_running_furniture(markdown)
        if _is_multi_round(getattr(self, "_active_test_id", "")):
            return self._renumber_tournament_solution_page(page_index, markdown)
        return self._renumber_page(page_index, markdown)

    @override
    def solution_source(self, test):
        sol = test.source.parent / "solutions.pdf"
        return sol if sol.exists() else None

    @override
    def answer_source(self, test):
        return self.solution_source(test)

    @override
    def parse_solutions(self, full_text, test=None, **kwargs):
        """Drop each restated statement and keep only its worked solution.

        Figure placeholders are retained so DETR crops stay inline.
        """
        grouped = _solution_blocks(full_text, self.match_marker())
        return {n: _solution_body("\n".join(parts)) for n, parts in grouped.items()}

    @override
    def postprocess_solutions(self, solutions, statements, test=None):
        """Drop an exact restated prompt from older BMT solution blocks."""
        cleaned = {}
        for number, value in solutions.items():
            statement = statements.get(str(number), "").strip()
            values = value if isinstance(value, list) else [value]
            trimmed = []
            for text in values:
                text = text.strip()
                if statement:
                    # Statement and solution OCR may differ only in blank-line
                    # wrapping.  Match its whitespace flexibly, but require the
                    # full prompt at the beginning before removing anything.
                    prefix = re.sub(r"\\\s+", r"\\s+", re.escape(statement))
                    match = re.match(prefix, text)
                    if match is not None:
                        text = text[match.end() :].lstrip()
                if text:
                    trimmed.append(text)
            if trimmed:
                cleaned[number] = trimmed
        return cleaned

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        """Extract answers from HTML tables, explicit Answer/Ans lines, boxed values, or numbered lists."""
        full_text = "\n\n".join(
            self.clean_solution_markdown(index, page)
            if _is_multi_round(getattr(self, "_active_test_id", ""))
            else self._renumber_page(index, page)
            for index, page in enumerate(pages_markdown)
        )
        answers = {}

        # 1. HTML answer key tables (e.g., Speed Round answer keys)
        td_matches = re.findall(
            r"<td>\s*(\d+)\s*</td>\s*<td>\s*(.*?)\s*</td>", full_text, re.I | re.S
        )
        # A worked solution can contain a numerical HTML table (for example,
        # BmMT 2025 lists 1--49 while solving question 5).  Treat table cells
        # as an answer key only when the document explicitly identifies one.
        if td_matches and "answer key" in full_text.casefold():
            for q_str, ans_str in td_matches:
                num = int(q_str)
                clean_ans = _clean_answer(ans_str)
                if clean_ans:
                    answers[num] = clean_ans
            if len(answers) >= 5:
                return answers
            answers = {}

        # 2. Problem blocks.  Do not let a numbered proof step create an answer
        # key entry (notably BMT 2019 Discrete and the individual tiebreaker).
        for number, parts in _solution_blocks(full_text, self.match_marker()).items():
            answer = _answer_value("\n".join(parts))
            if answer and answer.upper() != "N/A":
                answers[number] = answer

        # The 2012 tournament solution packet predates explicit Answer lines.
        # Its source PDF has a complete statement layer, so the optional text
        # extractor receives both the exact prompt and that question's worked
        # solution.  Keep deterministic values above; this is a fail-soft
        # supplement, never a replacement for them.
        if test.id == "bmt_2012_tournament":
            statements = _tournament_statement_blocks(test)
            for number, parts in _solution_blocks(full_text, self.match_marker()).items():
                if number in answers:
                    continue
                solution = "\n".join(parts).strip()
                statement = statements.get(number, "")
                # Question 22 explicitly asks for S_1 / S.  Its source
                # solution derives 1 / (k^2 + (k - 1)^2), with k = 42;
                # preserve that exact symbolic conclusion rather than letting
                # prose extraction confuse the area symbol S for the answer.
                if number == 22 and re.search(r"k\s*=\s*42", statement) and re.search(
                    r"1\s*\}\s*\{k\^2\s*\+\s*\(k-1\)\^2", solution
                ):
                    answers[number] = "1/3445"
                    continue
                answer = answer_llm.extract(
                    f"Statement:\n{statement}\n\nSolution:\n{solution}"
                    if statement
                    else solution
                )
                if answer:
                    answers[number] = _clean_answer(answer)

        # 3. Line-by-line numbered lists (standalone answer key documents)
        if not answers and "answer key" in full_text.casefold():
            for line in full_text.splitlines():
                leading, pairs = numbered_answers_in_line(line)
                for num, ans_str in pairs:
                    clean_ans = _clean_answer(ans_str)
                    if clean_ans:
                        answers[num] = clean_ans

        return answers


def _answer_value(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _ANSWER_RE.match(line)
        if match is None:
            continue
        inline = _clean_answer(match.group(1))
        if inline:
            return inline
        # Value carried onto the following non-blank line, before any label.
        for following in lines[index + 1 :]:
            if _PROPOSED_RE.match(following) or _SOLUTION_RE.match(following):
                break
            if following.strip():
                return _clean_answer(following)
        return ""
    return _boxed_answer(block) or ""


def _solution_body(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _SOLUTION_RE.match(line)
        if match is not None:
            # Drop a "**" emphasis closer the label's colon left behind.
            first = re.sub(r"^[*_]+\s*", "", match.group(1).strip())
            kept = ([first] if first else []) + lines[index + 1 :]
            return "\n".join(kept).strip()
    # No Solution label (unexpected layout / OCR miss): keep the block minus the
    # answer and proposer furniture so nothing is silently dropped.
    return "\n".join(
        l for l in lines if not (_ANSWER_RE.match(l) or _PROPOSED_RE.match(l))
    ).strip()
