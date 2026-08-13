"""Caltech Math Meet / Caltech-Harvey Mudd Math Competition.

On-disk layout (data dir is ``CHMM/out``)::

    out/<year>/<season>/<round>/test.pdf
    out/<year>/<season>/<round>/solutions.pdf

Test IDs mirror the directory path, joined by underscores, for example
``2025_annual_individual`` and ``2018_fall_team``.  Power rounds are omitted:
their section/definition numbering is hierarchical and does not fit the
pipeline's one-flat-problem-per-number output.  Other proof-style rounds are
kept when their top-level questions are numbered normally.
"""

import re
import textwrap
from pathlib import Path

from typing_extensions import override

from .. import anchors, config
from ..nanonets import FIGURE_PLACEHOLDER, parse_layout
from .base import Series, Test, route_section_preambles, strip_section_tail
from .smt import _boxed_answer


# Fall 2012 prefixes each round's problems (IR1, MR2, TR3, TBR4).  The 2015
# packets instead print "Problem 0.1", "Problem 0.2", ...; the leading zero is
# a section number, not part of the stable problem number.
_PREFIXED_MARKER_RE = re.compile(r"^\s*(?:IR|MR|TR|TBR)\s*(\d+)\s*[.)]?", re.I)
_ZERO_SECTION_RE = re.compile(r"^\s*Problem\s+0[.](\d+)\s*[.)]?", re.I)
# Several solution-only packets use "Solution 1." as the sole block marker.
_SOLUTION_MARKER_RE = re.compile(r"^\s*Solution\s+(\d+)\s*[.:)]?", re.I)
_MIXER_PART_BOUNDARY_RE = re.compile(
    r"(?im)^[ \t]*\*\*[ \t]*Part\s+(?:\d+|[IVXLC]+)[ \t]*\*\*[ \t]*(?:\n|$)"
)
_INTEGRAL_MARKER_RE = re.compile(
    r"^\s*(?:\\textbf\{\s*)?Integral\s+(\d+)\b(?:\s+Answer)?\s*\}?", re.I
)
_INTEGRAL_HEADING_RE = re.compile(
    r"(?im)^\s*(?:[*_#]+\s*|\\textbf\{\s*)*"
    r"Integral\s+(\d+)\s*(Answer)?\s*(?:\}|[*_#])*\s*$"
)
_DECIMAL_SUBSECTION_RE = re.compile(
    r"^\s*(?:Lemma|Proof|Observation|Case)?\s*\d+\.\d+", re.I
)

_SOLUTION_LINE_RE = re.compile(
    r"^\s*(?:[*_#]{0,3}\s*)?(?:\d+(?:\.\d+)?\s+)?"
    r"(?:[A-Za-z]+\s+)?Solution(?:\s+\d+)?\s*[.:]?\s*[*_]{0,2}\s*(.*)$",
    re.I,
)
_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:[*_#]{0,3}\s*)?Answer\s*[*_]{0,2}\s*[.:]?\s*(.*)$", re.I
)
_SOLUTION_BOUNDARY = "[[CHMM_SOLUTION_BOUNDARY]]"

# Numbered rules on modern cover pages share a page with problem 1.  Rejecting
# these known rule sentences in the matcher avoids turning the real problem 1
# into problem 8 after parse_layout's section-restart offset.
_RULE_START_RE = re.compile(
    r"^(?:"
    r"you have \d+ minutes|"
    r"do not (?:look|flip|turn)|"
    r"this test consists|"
    r"no (?:collaboration|computational aids)|"
    r"you (?:are,?|may|can) (?:however, )?(?:permitted|collaborate|message)|"
    r"you may not collaborate|"
    r"congratulations for scoring|"
    r"there are \d+ questions|"
    r"the top \d+|"
    r"the time limit|"
    r"on the back side|"
    r"write (?:your|answers|legibly)|"
    r"all (?:the )?answers (?:are|must|should)|"
    r"answers (?:are|must|should)|"
    r"if you believe|"
    r"for multi-part problems|"
    r"number of (?:correct\s+)?answers|"
    r"time at which|"
    r"ties (?:will\s+be\s+)?broken"
    r")\b",
    re.I,
)


def _match_marker(text: str):
    if _DECIMAL_SUBSECTION_RE.match(text):
        return None
    if re.match(r"^\s*(\d+)\.(\d+)", text):
        return None
    for pattern in (
        _INTEGRAL_MARKER_RE,
        _PREFIXED_MARKER_RE,
        _ZERO_SECTION_RE,
        _SOLUTION_MARKER_RE,
    ):
        match = pattern.match(text)
        if match is not None:
            return int(match.group(1)), match.end()
    result = anchors._match_marker(text)
    if result is None:
        return None
    if result[0] > 100:
        return None
    if _RULE_START_RE.match(text[result[1] :].strip()):
        return None
    return result


class ChmmSeries(Series):
    name = "chmm"
    has_solutions = True
    has_answers = True
    split_multiple_solutions = True
    validate_answer_candidates = True

    ignored_test_substrings = ("math-talk", "tcs")
    proof_test_patterns = (r"^\d{4}_(?:fall|winter|spring|annual)_proof$",)

    @override
    def discover_tests(self, data_dir):
        """Discover flat-numbered rounds and omit hierarchical power packets."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        return [
            Test(id="_".join(pdf.relative_to(root).parts[:-1]), source=pdf)
            for pdf in sorted(root.glob("**/test.pdf"))
            if pdf.parent.name != "power"
        ]

    @override
    def match_marker(self):
        return _match_marker

    @override
    def layout_options(self):
        return config.LayoutOptions(
            inline_figures=True,
            strict_section_restarts=True,
            flat_problem_numbering=True,
            # Some solution pages place a bare ``N.`` immediately after the
            # preceding problem's closing display equation.  Split that safe,
            # otherwise line-invisible boundary before grouping solutions.
            split_glued_bare_markers=True,
            max_picture_area_frac=0.2,
            header_picture_frac=0.1,
            # Integration Bee slides contain a long, 16-pixel-high answer
            # rule that DETR otherwise saves as a figure for every integral.
            # Genuine CHMM diagrams are substantially taller.
            min_picture_height_frac=0.025,
            solution_answer_box_filter=True,
            solution_answer_box_max_width_frac=0.40,
            statement_answer_table_filter=True,
        )

    @override
    def clean_statement_markdown(self, page_index: int, markdown: str) -> str:
        # Modern cover sheets contain only numbered rules. Suppressing the page
        # prevents those numbers from becoming the carry into the real problem
        # page, and also prevents their logos from being assigned to problem 1.
        has_rules_heading = re.search(
            r"(?im)^\s*(?:[*_#]+\s*)?Rules(?:\s+and\s+Directions)?"
            r"\s*(?:[*_]+)?\s*$",
            markdown,
        )
        has_problem_heading = re.search(
            r"(?im)^\s*(?:[*_#]+\s*)?(?:Problem|Question|Integral)\s+1\b",
            markdown,
        )
        if has_rules_heading and not has_problem_heading:
            return ""

        # Finals alternate question and answer slides. Answer slides are parsed
        # separately by parse_answers and must never be appended to statements.
        if _is_integral_answer_page(markdown):
            return ""

        # Preserve an explicitly numbered but intentionally blank source item.
        # Without a small truthful placeholder, the writer drops the empty
        # statement and turns the source's blank Problem 11 into a sequence gap.
        markdown = re.sub(
            r"(?m)^(\d+)\.\s*$",
            r"\1. [No statement was printed in the source.]",
            markdown,
        )

        # The 2026 qualifying sheet places all integrals on one OCR line. Split
        # only markers followed by an integral, not prose such as "area 1.
        # Points..." or "digits 1 to 5. Ryan...".
        markdown = re.sub(
            r"(?<=\S)\s+(?=(?:[1-9]\d?)\s*[.)]\s+\$?\s*\\int)",
            "\n",
            markdown,
        )
        # Strip answer table footers printed at bottom of statement pages
        markdown = re.sub(
            r"(?i)\n+\s*(?:\*{1,2}|#+)?\s*(?:\d+\s+)?Answers?\s*(?:\*{1,2})?\s*\n+<table>[\s\S]*?</table>",
            "",
            markdown,
        )
        return markdown

    @override
    def postprocess(self, problems):
        return route_section_preambles(problems, _MIXER_PART_BOUNDARY_RE)

    @override
    def solution_source(self, test: Test):
        # Finals packets contain question/answer slides, not worked solutions.
        if "integration-bee-finals" in test.id.casefold():
            return None
        solution = test.source.parent / "solutions.pdf"
        if solution.exists():
            return solution
        season = test.source.parent.parent
        if season.is_dir():
            candidates = [
                p
                for p in sorted(season.glob("**/solutions.pdf"))
                if p.parent.name not in (
                    "power",
                    "individual",
                    "team",
                    "mixer",
                    "tiebreaker",
                    "integration-bee-finals",
                    "integration-bee-qualifying",
                )
            ]
            if candidates:
                return candidates[0]
        return None

    @override
    def clean_solution_markdown(self, page_index: int, markdown: str) -> str:
        # Markdown OCR preserves indentation on nested numbered instructions,
        # but parse_layout deliberately normalizes line whitespace before
        # matching problem markers. Turn those nested markers into bullets here
        # so they remain solution prose instead of starting false problems.
        markdown = textwrap.dedent(markdown)
        return re.sub(r"(?m)^[ \t]{2,}(\d+[.)][ \t]+)", r"- \1", markdown)

    @override
    def answer_source(self, test: Test):
        # Every numbered final integral is immediately followed by its answer in
        # the test presentation, including 2026 (which has no solutions.pdf).
        if "integration-bee-finals" in test.id.casefold():
            return test.source
        return self.solution_source(test)

    @override
    def parse_solutions(self, full_text: str, test: Test = None) -> dict:
        if test is not None:
            full_text = _filter_round_text(full_text, test.id)
        if not full_text.strip():
            return {}
        full_text = _preserve_solution_boundaries(full_text)
        grouped = _group_blocks(
            full_text,
            split_glued_bare_markers=self.layout_options().split_glued_bare_markers,
        )
        # A few early files are short answer keys, not worked solutions.  They
        # belong in problem_answer.json only.
        if _is_answer_key(full_text):
            return {}
        bodies = {
            number: _solution_body(block) for number, block in grouped.items()
        }
        if self.split_multiple_solutions:
            solutions = {}
            for n, text in bodies.items():
                if text and text.strip():
                    chunks = self.split_solution_block(text)
                    solutions[n] = chunks if len(chunks) > 1 else chunks[0]
        else:
            solutions = bodies

        # Two mixer packets present a dependency-chain solution only after the
        # final problem in each part. Associate that shared derivation with all
        # problems it solves instead of retaining the repeated problem prompts
        # as if they were solutions.
        if "This completes part 1" in solutions.get(6, ""):
            shared = solutions[6]
            for number in range(1, 7):
                solutions[number] = shared
        if "This completes the round" in solutions.get(12, ""):
            shared = solutions[12]
            for number in range(7, 13):
                solutions[number] = shared
        if "We define, as in the problems above" in solutions.get(16, ""):
            shared = solutions[16]
            for number in range(13, 17):
                solutions[number] = shared
        return solutions

    @override
    def postprocess_solutions(self, solutions, statements, test: Test = None):
        return {
            number: strip_section_tail(value, _MIXER_PART_BOUNDARY_RE)
            for number, value in solutions.items()
        }

    @override
    def parse_answers(self, test: Test, pages_markdown: list) -> dict:
        if "integration-bee-finals" in test.id.casefold():
            return _parse_integration_final_answers(pages_markdown)
        # Filter the complete document, not each page independently. A shared
        # packet's continuation pages have no repeated round heading and must
        # inherit the section established on the preceding page.
        cleaned_pages = [
            self.clean_solution_markdown(index, markdown)
            for index, markdown in enumerate(pages_markdown)
        ]
        full_text = _filter_round_text("\n\n".join(cleaned_pages), test.id)
        if not full_text.strip():
            return {}
        # Older mixer packets are headed "Mixer Round Answers", but several
        # entries continue with an explanation immediately after the answer.
        # They are answer keys, not ordinary solution packets; route them
        # through the format-specific parser before the generic block parser.
        if "mixer" in test.id.casefold() and _is_mixer_answer_key(full_text):
            return _parse_mixer_answers(full_text)
        grouped = _group_blocks(
            full_text,
            split_glued_bare_markers=self.layout_options().split_glued_bare_markers,
        )
        answer_key = _is_answer_key(full_text)
        answers = {}
        for number, block in grouped.items():
            value = _answer_value(block)
            if not value and answer_key:
                value = _clean_value(block)
            if _usable_answer(value):
                answers[number] = value
        # The 2010 mixer is a dependency-chain puzzle: its combined solution
        # assigns bold letter variables instead of repeating a conventional
        # answer line under every problem. Recover those explicit assignments.
        answers.update(_symbolic_mixer_answers(full_text))
        answers.update(
            {
                number: value
                for number, value in _direct_solution_answers(full_text).items()
                if number not in answers
            }
        )
        # Some born-digital boxed values are exposed as a checkbox glyph by the
        # VLM even though the PDF text layer contains the exact value. Fill only
        # missing entries from that authoritative layer; keep richer LaTeX OCR
        # for answers it already parsed successfully.
        native_source = self.solution_source(test) if test.source is not None else None
        for number, value in _native_pdf_answers(test, native_source).items():
            answers.setdefault(number, value)
        return answers


_MIXER_ANSWER_HEADER_RE = re.compile(
    r"(?im)^\s*mixer\s+round\s+answers\b"
)
_MIXER_ANSWER_MARKER_RE = re.compile(r"(?m)^\s*(\d+)\.\s*(.*)$")


def _is_mixer_answer_key(full_text: str) -> bool:
    """Return whether text has CHMM's distinctive mixer answer-key heading."""
    return bool(_MIXER_ANSWER_HEADER_RE.search(full_text))


def _parse_mixer_answers(full_text: str) -> dict[int, str]:
    """Parse the legacy CHMM mixer answer-key format.

    Entries normally look like ``N. answer; source; acceptable range``.  A
    few older entries append an explanation without a delimiter, so only the
    leading answer expression is retained.  This is deliberately separate
    from the generic CHMM parser because ordinary mixer solution packets use
    numbered problem blocks and must keep their existing semantics.
    """
    answers: dict[int, str] = {}
    for match in _MIXER_ANSWER_MARKER_RE.finditer(full_text):
        number = int(match.group(1))
        value = match.group(2).strip()
        # Problem 9 puts the actual requested value at the end of a one-line
        # derivation (``... he should sell when b = 20, or k = 345``).
        if value.casefold().startswith("solving "):
            result = re.search(r"\bk\s*=\s*([^.,;\s]+)", value, re.IGNORECASE)
            if result:
                value = result.group(1).strip("$")
        value = re.split(
            r"\s*;\s*|\s+(?=(?:Let|Solving|Therefore|We\s|By\s|Since\s|Telescoping))",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        value = _clean_value(value)
        if value:
            answers[number] = value
    return answers


def _filter_round_text(full_text: str, test_id: str) -> str:
    """If full_text contains multi-round section headers, filter to test_id's round."""
    round_keywords = ("individual", "team", "mixer", "tiebreaker", "integration")
    target = next((r for r in round_keywords if r in test_id.lower()), None)
    if not target:
        return full_text

    header_pattern = re.compile(
        r"(?im)^[ \t]*(?:#+[ \t]*|\*{1,3}[ \t]*)?"
        r"(?:Fall|Spring|Winter|Annual)?[ \t]*(?:\d{4})?[ \t]*"
        r"(?:Caltech[- ]Harvey Mudd Math Competition)?\s*"
        r"(Individual|Team|Mixer|Tiebreaker|Power|Integration(?:\s+Bee)?)"
        r"\s+(?:Round\s*)?(?:Solutions|Answers|Round)?"
        r"[ \t]*(?:\*{1,3})?[ \t]*$"
    )
    matches = list(header_pattern.finditer(full_text))
    if not matches:
        return full_text

    sections = []
    for i, m in enumerate(matches):
        r_name = m.group(1).lower()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        sections.append((r_name, full_text[start:end]))

    matching_text = [sec_text for r_name, sec_text in sections if target in r_name]
    return "\n\n".join(matching_text) if matching_text else ""


def _group_blocks(
    full_text: str, *, split_glued_bare_markers: bool = False
) -> dict[int, str]:
    grouped: dict[int, list[str]] = {}
    for item in parse_layout(
        full_text,
        _match_marker,
        strict_section_restarts=True,
        flat_problem_numbering=True,
        split_glued_bare_markers=split_glued_bare_markers,
    ):
        if item["problem"] is None:
            continue
        value = item["text"] if item["kind"] == "text" else FIGURE_PLACEHOLDER
        grouped.setdefault(item["problem"], []).append(value)
    return {number: "\n".join(parts).strip() for number, parts in grouped.items()}


def _is_answer_key(full_text: str) -> bool:
    head = "\n".join(full_text.splitlines()[:12])
    has_answer_header = bool(
        re.search(
            r"\b(?:Individual|Team|Tiebreaker|Mixer)\s+(?:Answers|Round Solutions|Solutions)\b",
            head,
            re.I,
        )
    )
    has_solution_prose = bool(
        re.search(
            r"(?im)^\s*(?:#+\s*|\*{1,2}\s*)?(?:\d+(?:\.\d+)?\s+)?Solution[s.:\s]?\b|\b(?:Proof|Lemma|Observation|Proposed by)\b",
            full_text,
        )
    )
    return has_answer_header and not has_solution_prose


def _solution_body(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith(_SOLUTION_BOUNDARY):
            inline = line.strip()[len(_SOLUTION_BOUNDARY) :].strip()
            return "\n".join(([inline] if inline else []) + lines[index + 1 :]).strip()
        match = _SOLUTION_LINE_RE.match(line)
        if match is not None:
            first = match.group(1).lstrip("*_ ").strip()
            return "\n".join(([first] if first else []) + lines[index + 1 :]).strip()
    # "Solution N." can itself be the marker and is removed by parse_layout.
    return block.strip()


def _preserve_solution_boundaries(full_text: str) -> str:
    """Keep same-number Problem/Solution transitions visible to _solution_body."""
    pattern = re.compile(
        r"(?im)^\s*(?:[*_#]+\s*)?(?:"
        r"Solution\s+(\d+)\s*[.:]\s*(?:[*_]+\s*)?(.*)|"
        r"(\d+)\.\d+\s+(?:[A-Za-z]+\s+)?Solution\s*(?:[*_]+)?"
        r")\s*$"
    )

    def replace(match):
        number = int(match.group(1) or match.group(3))
        before = full_text[: match.start()]
        if not re.search(rf"(?i)\bProblem\s+(?:0\.)?{number}\b", before):
            return match.group(0)
        inline = (match.group(2) or "").strip()
        # Keep a terse numeric answer on the sentinel line. Moving "22." onto
        # its own line would make parse_layout mistake it for Problem 22.
        return _SOLUTION_BOUNDARY + (f" {inline}" if inline else "")

    return pattern.sub(replace, full_text)


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
    boxed = _boxed_answer(block)
    if boxed:
        return boxed

    # Common prose forms in proof and modern solution packets.
    for line in lines:
        match = re.search(
            r"(?i)\b(?:the\s+)?answers?\s+(?:is|are)\s+(.+?)(?:[.]\s|$)",
            line,
        )
        if match is not None:
            value = re.split(
                r"[.]\s+(?=(?:First|We|Consider|Let|Since|Indeed|Thus)\b)",
                match.group(1),
                maxsplit=1,
                flags=re.I,
            )[0]
            value = _clean_value(value)
            if value:
                return value

    # Older packets often finish with an unboxed declarative result.
    tail_patterns = (
        r"(?i)\bour total sum is\s+(.+?)(?:[.]|$)",
        r"(?i)\bmaximum value of\s+(.+?)(?:[.]|$)",
        r"(?i)\bor\s+(\$?\\sqrt\{[^$]+\}\$?)(?:[.]|$)",
        r"(?i)\band so\s+\$?[A-Za-z]\s*=.*=\s*(.+?)(?:\$?[.]|$)",
        r"(?i)\b(\d[\d,]*(?:\.\d+)?)\$?\s+moves(?:[.]|$)",
    )
    tail = "\n".join(lines[-8:])
    for pattern in tail_patterns:
        matches = list(re.finditer(pattern, tail))
        if matches:
            value = _clean_value(matches[-1].group(1))
            if "=" in value:
                value = _clean_value(value.rsplit("=", 1)[-1])
            if value:
                return value

    # Some packets put a short answer directly after "Solution:".
    for line in lines:
        match = _SOLUTION_LINE_RE.match(line)
        if match is None:
            continue
        value = _clean_value(match.group(1))
        if value and len(value) <= 120 and not re.search(
            r"(?i)\b(?:because|suppose|note that|we (?:claim|first|have|will))\b",
            value,
        ):
            return value
    return ""


def _usable_answer(value: str) -> bool:
    return bool(value and value.strip() not in {"☑", "☐", "□", "■", "�"})


def _direct_solution_answers(full_text: str) -> dict[int, str]:
    """Recover terse answers that parse_layout intentionally strips as markers."""
    markers = list(
        re.finditer(
            r"(?im)^\s*(?:[*_#]+\s*)*Solution\s+(\d+)\s*"
            r"[.:]\s*(?:[*_]+\s*)?(.*)$",
            full_text,
        )
    )
    answers = {}
    for index, marker in enumerate(markers):
        number = int(marker.group(1))
        end = markers[index + 1].start() if index + 1 < len(markers) else len(full_text)
        block = full_text[marker.end() : end]
        inline = marker.group(2).strip()
        searchable = "\n".join(part for part in (inline, block) if part)

        image = re.search(r"<img>\s*([^<\n]{1,120})\s*</img>", searchable, re.I)
        if image is not None:
            value = _clean_value(image.group(1))
            if _usable_answer(value):
                answers[number] = value
                continue

        if inline.startswith("$"):
            value = _clean_value(inline)
            if "=" in value:
                value = _clean_value(value.rsplit("=", 1)[-1]).strip("$").strip()
            if _usable_answer(value) and len(value) <= 120:
                answers[number] = value
                continue

        candidates = (
            r"(?i)\btotal of\s+\$([^$]+)\$\s+possibilit",
            r"(?i)\bThere are\s+\$([^$]+)\$\s+numbers\b",
            r"(?i)\$x\s*=\s*(.+?)\$",
        )
        for pattern in candidates:
            matches = list(re.finditer(pattern, searchable))
            if matches:
                value = _clean_value(matches[-1].group(1))
                if _usable_answer(value):
                    answers[number] = value
                    break
    return answers


def _native_pdf_answers(test: Test, source: Path | None) -> dict[int, str]:
    """Read terse ``Solution N. value`` entries from a PDF's text layer."""
    if source is None or not source.exists():
        return {}
    try:
        import fitz

        with fitz.open(source) as document:
            text = "\n".join(page.get_text() for page in document)
    except (ImportError, OSError, RuntimeError, ValueError):
        return {}
    text = _filter_round_text(text, test.id)
    answers = {}
    markers = list(
        re.finditer(r"(?im)^\s*Solution\s+(\d+)\s*[.:]\s*(.*)$", text)
    )
    for marker in markers:
        number = int(marker.group(1))
        line = marker.group(2).strip()
        inline = bool(line)
        if not line:
            following = text[marker.end() :].splitlines()
            line = next((part.strip() for part in following if part.strip()), "")
        # TeX's boxed value is often followed by a separately extracted period
        # and then prose on the same physical line.
        parts = re.split(r"\s+[.]\s+", line, maxsplit=1)
        has_value_separator = len(parts) == 2
        line = parts[0].strip()
        value = _clean_value(line)
        if (
            _usable_answer(value)
            and len(value) <= 120
            and (not inline or has_value_separator or " " not in value)
            and not re.match(
                r"(?i)^(?:the|we|let|first|if|suppose|consider|using|note|since)\b",
                value,
            )
        ):
            answers[number] = value
    return answers


def _symbolic_mixer_answers(full_text: str) -> dict[int, str]:
    """Resolve explicit ``A = value`` assignments in linked mixer solutions."""
    symbol_to_problem = {
        symbol: int(number)
        for symbol, number in re.findall(
            r"\\mathbf\{([A-Z])\}\$?\s+be\s+the\s+answer\s+to\s+"
            r"problem(?:\s+number)?\s+(\d+)",
            full_text,
            re.I,
        )
    }
    answers = {}
    for symbol, number in symbol_to_problem.items():
        match = re.search(
            rf"\\mathbf\{{{re.escape(symbol)}\}}\s*=\s*"
            r"([^$,\n.;]+)",
            full_text,
        )
        if match is not None:
            value = _clean_value(match.group(1))
            if value:
                answers[number] = value

    # One link in the 2010 packet is written only as prose, with no E = ...
    # assignment. Its local "For problem N" paragraph still states the answer.
    for match in re.finditer(
        r"(?is)\bFor problem\s+(\d+),(.+?)(?=\n\s*For problem\s+\d+,|\n\s*"
        r"(?:This completes|Part\s+II)\b|$)",
        full_text,
    ):
        value = _answer_value(match.group(2))
        if value:
            answers.setdefault(int(match.group(1)), value)

    # The 2013 Part 4 chain uses lowercase w/x/y/z and states the three
    # interdependent values as one tuple after the derivation.
    lower_to_problem = {
        symbol.lower(): int(number)
        for number, symbol in re.findall(
            r"answer\s+to(?:\s+number)?\s+(\d+)\s+is\s+\$?([wxyz])\$?",
            full_text,
            re.I,
        )
    }
    tuple_match = re.search(
        r"\(\s*w\s*,\s*x\s*,\s*z\s*\)\s*=\s*\(\s*"
        r"([^,]+),\s*([^,]+),\s*([^)]+)\)",
        full_text,
        re.I,
    )
    if tuple_match is not None:
        for symbol, raw in zip(("w", "x", "z"), tuple_match.groups()):
            if symbol in lower_to_problem:
                answers[lower_to_problem[symbol]] = _clean_value(raw)
    if "y" in lower_to_problem:
        y_match = re.search(
            r"(?is)From our earlier formula,\s*\$EG\s*=.*?=\s*([^$\n]+)\$",
            full_text,
        )
        if y_match is not None:
            answers[lower_to_problem["y"]] = _clean_value(y_match.group(1))
    return answers


def _is_integral_answer_page(markdown: str) -> bool:
    return any(match.group(2) for match in _INTEGRAL_HEADING_RE.finditer(markdown))


def _parse_integration_final_answers(pages_markdown: list[str]) -> dict[int, str]:
    answers = {}
    for markdown in pages_markdown:
        match = next(
            (m for m in _INTEGRAL_HEADING_RE.finditer(markdown) if m.group(2)),
            None,
        )
        if match is None:
            continue
        number = int(match.group(1))
        body = re.sub(r"<page_number>[\s\S]*?</page_number>", "", markdown[match.end() :])
        value = _boxed_answer(body)
        if not value:
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            value = _clean_value(lines[0]) if lines else ""
        if value:
            answers[number] = value
    return answers


def _clean_value(value: str) -> str:
    value = value.strip().strip("*_").strip()
    if value == FIGURE_PLACEHOLDER or value.lower().startswith("proposed by"):
        return ""
    return value.rstrip(".").strip()
