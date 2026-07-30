"""Mandelbrot: several tests per season, each with a sibling solution PDF.

On-disk layout (data dir is ``Mandelbrot/out``)::

    out/<season>/tmctest<n>{N,R}.pdf   individual rounds (National / Regional)
    out/<season>/tmcsoln<n>{N,R}.pdf   their solutions
    out/<season>/mtptest<n>.pdf        team-play rounds
    out/<season>/mtpsoln<n>.pdf        their solutions
    out/<season>/mtptopics<n>.pdf      topic lists (not problems -- skipped)

Discovery finds every ``*test*.pdf`` (id ``<season>_<stem>``, e.g.
``2017-18_tmctest1N``), then the series-wide ``mtp`` exclusion removes team-play
rounds before parsing or OCR. ``mtptopics*`` is never discovered because it has
no ``test`` in the name. The solution is the sibling with ``test`` swapped to
``soln``.

The solution PDF opens with an "Answer Key" box -- short ``N. answer`` entries
in column order (``4. 12  1. 5  5. 3025 ...``) -- before the worked solutions
begin. Its out-of-order markers would corrupt the strictly-increasing problem
segmentation, so `clean_solution_markdown` strips the box before any problem
tagging, and `parse_answers` reads the same box back out as the answer key.
"""

import re
from pathlib import Path

from .. import config
from .base import Series, Test, numbered_answers_in_line, strip_solution_page_furniture

_TAG_RE = re.compile(r"<[^>]+>")
# A line that is only table scaffolding (or blank). The key box is often OCR'd
# as an HTML table; its <table>/<tr> wrapper lines must be stripped along with
# the entries, or a dangling "<table>" left at the top of the page pairs with
# some later table's "</table>" and swallows every solution between them.
_TABLE_MARKUP_ONLY_RE = re.compile(
    r"^\s*(?:</?(?:table|thead|tbody|tr|td|th)\b[^>]*>\s*)*$", re.IGNORECASE
)
# A key entry is a bare value ("12", "9/17", "3025"); solution prose after a
# marker ("1. It is possible to fit...") is far longer. Used to find where the
# box ends.
_MAX_ANSWER_LEN = 24

# The solutions PDF closes with a back cover -- a "© Proof School <year>"
# copyright line, then a PROBLEM/CREDITS table naming each solution's author and
# publisher boilerplate ("★ REGIONAL LEVEL ★", "Produced by Proof School", the
# org's website). It shares the last page with the tail of the final solution,
# so left in, every one of those lines binds to that problem and pollutes its
# text (and the credits table's "1. ... 7. ..." cells look like stray markers).
# The copyright line marks the end of real content, so everything from it to the
# end of the page is dropped. Matched leniently -- the © glyph does not always
# OCR -- but anchored to a line that *starts* with "Proof School": the copyright
# line does (after its ©), while the later "Produced by Proof School" line does
# not, so `search` lands on the true boundary even without the glyph.
_BACK_COVER_RE = re.compile(
    r"(?im)^[^\S\n]*(?:©|\(c\)|copyright)?[^\S\n]*proof school\b"
)

# The PDF text-layer form of that same copyright line, for fencing figures (see
# solution_figure_floor). In the born-digital text the © glyph decodes as a
# leading combining ring, so the line -- "<ring>Proof School 2018" -- is not
# anchored the way `_BACK_COVER_RE` expects; match "Proof School" adjacent to a
# four-digit year instead, which the copyright line carries but the lone
# "Proof School" footer line does not, so this lands on the true content boundary.
_PDF_COPYRIGHT_RE = re.compile(
    r"proof school\s*\d{4}|\d{4}\s*proof school", re.IGNORECASE
)
_RUNNING_SOLUTION_FURNITURE_RE = re.compile(
    r"^(?:"
    r"(?:★\s*)?(?:regional|national)\s+level(?:\s*★)?"
    r"|the\s+mandelbrot\s+competition"
    r"|round\s+(?:one|two|three|four|\d+)\s+solutions?"
    r"|greater\s+testing\s+concepts"
    r"|www\.mandelbrot\.org"
    r"|page\s+\d+"
    r")$",
    re.IGNORECASE,
)


def _strip_back_cover(text: str) -> str:
    """Drop the trailing back-cover boilerplate from a solution page's markdown."""
    m = _BACK_COVER_RE.search(text)
    return text[: m.start()].rstrip() if m else text


def _is_answer_key_heading(line: str) -> bool:
    """True when `line` carries the "Answer Key" heading.

    Nanonets renders the box's small-caps title with markup *between the
    letters* ("<strong>A</strong><sub>NSWER</sub> ..."), so tags become nothing
    and all spacing is dropped before matching.
    """
    return "answerkey" in re.sub(r"[^a-z]", "", _TAG_RE.sub("", line).lower())


def _split_answer_key(text: str):
    """Split OCR markdown into ``(answers, text_without_the_key)``.

    The key region starts at the "Answer Key" heading (entries may share that
    line when the box is OCR'd as one row) and ends at the first line that is
    neither blank/markup-only nor made of short ``N. answer`` entries -- in
    practice solution 1's opening prose. Returns ``({}, text)`` unchanged when
    no heading or no entries are found, so pages without a key pass through.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _is_answer_key_heading(ln)), None)
    if start is None:
        return {}, text
    answers = {}

    def consume(line, allow_leading=False):
        """Record the line's entries; False if it isn't a pure answer-key line."""
        leading, pairs = numbered_answers_in_line(line)
        if (leading and not allow_leading) or not pairs:
            return False
        if any(len(a) > _MAX_ANSWER_LEN for _, a in pairs):
            return False
        for n, a in pairs:
            if a:
                answers.setdefault(n, a)
        return True

    # Entries can share the heading line (the box OCR'd as one row); the
    # heading text itself is the leading part, so it is allowed there.
    consume(lines[start], allow_leading=True)
    end = start + 1
    for i in range(start + 1, len(lines)):
        if _TABLE_MARKUP_ONLY_RE.match(lines[i]):
            end = i + 1  # blank / table-scaffolding line inside the box
            continue
        if not consume(lines[i]):
            break
        end = i + 1
    if not answers:
        return {}, text  # heading matched but no entries -- don't strip anything
    while start > 0 and _TABLE_MARKUP_ONLY_RE.match(lines[start - 1]):
        start -= 1  # the box's own <table>/<tr> wrapper lines above the heading
    return answers, "\n".join(lines[:start] + lines[end:])


class MandelbrotSeries(Series):
    name = "mandelbrot"
    has_solutions = True
    has_answers = True
    ignored_test_substrings = ("mtp",)

    def layout_options(self):
        """Nudge the OCR temperature off greedy, and unpack the problem table.

        Greedy decoding (temperature 0.0) sometimes gets stuck repeating a
        ``<table>`` row when a page shows a grid/diagram, blowing past the
        runaway guard or padding the transcription. A small bump breaks the loop
        while keeping transcription faithful; the layout heuristics stay at the
        conservative base defaults. Raise further only if grids still loop.

        ``split_marker_table_rows``: every Mandelbrot test page lays its problems
        out as a two-column table -- ``<td>N. <statement></td>`` beside a
        point-value cell -- so the problem markers live *inside* table cells. On
        pages whose cells also carry figure ``<img>`` tags the ``<img>`` splits
        the block apart and the markers fall through to the plain-text scanner;
        but on pages without figures the whole ``<table>`` stays in one token and
        (kept verbatim, the default) yields no markers at all -- an empty
        ``problems.json``. Unpacking marker rows into plain statements recovers
        those pages. The point-value cell (a circled digit ``①`` or a ``☐`` box)
        is not an answer blank, so it survives cell cleanup as furniture; it is
        dropped by nanonets' point-value filter.

        ``ordered_list_markers``: one round (2010-11 Round Three National) has the
        model number problems with an ``<ol>/<li>`` list -- the printed number
        OCR'd into the point-value column as a stray ``<img>`` -- so the
        statements carry no literal ``N.`` marker and the page would otherwise
        parse to nothing. Enabling it makes each list item the next problem.

        Figure detection: every page prints its problems in a two-column table
        with small answer-column diagrams that DETR scores well below the text
        threshold, so at the default threshold only the boldest figure per page
        survives -- and on some pages (e.g. 2018-19 Round Three National) every
        diagram scores under 0.2. ``picture_detect_threshold`` lowers the figure
        confidence to 0.15 (text/problem-start geometry stays at the default),
        and the region filters keep the extra low-confidence boxes clean:
        ``max_picture_area_frac`` drops the page-region boxes DETR emits at this
        confidence (real diagrams occupy a tiny fraction of the page),
        ``header_picture_frac`` drops the running header (fractal logo + title
        banner), ``right_margin_picture_frac`` drops the fixed right-hand
        furniture column -- the point-value circle beside every problem and the
        wide bottom-right "SCORE:" box -- and ``footer_picture_frac`` drops the
        stylized "SCORE:" label at the foot of the page (which sits in the
        statement column, so the right-margin filter alone misses it). Between
        them the right-edge and bottom-band tests fence off the answer/scoring
        furniture without touching the statement-column figures, which DETR
        scores low but which the counts confirm are real diagrams.
        ``point_marker_row_anchor`` reuses the vertical positions of those
        discarded point-value circles as a fallback row anchor when DETR's
        left-margin text boxes do not yield exactly one start per problem. The
        circle outlines are read from the page image, not trusted to DETR (which
        misses some at even the low figure threshold), and are accepted only
        when exactly one marker is found for every OCR-parsed problem.
        The last false positives the low threshold surfaces are the problems'
        own math typeset as figures. ``min_picture_height_frac`` drops the short
        ones -- inline equation strips and lone symbols, well under any real 2D
        diagram's height -- and ``equation_text_overlap`` drops the tall ones --
        stacked display fractions -- by the tell that they are wide and sit under
        a Text box, which a real (even digit-labeled) diagram does not.
        """
        return config.LayoutOptions(
            nanonets_temperature=0.1,
            split_marker_table_rows=True,
            ordered_list_markers=True,
            picture_detect_threshold=0.15,
            max_picture_area_frac=0.15,
            header_picture_frac=0.20,
            right_margin_picture_frac=0.20,
            point_marker_row_anchor=True,
            footer_picture_frac=0.14,
            min_picture_height_frac=0.036,
            equation_text_overlap=0.3,
        )

    def discover_tests(self, data_dir):
        """One test per ``*test*.pdf`` inside each season folder."""
        root = Path(data_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"data dir not found: {root}")
        tests = []
        for season in sorted(p for p in root.iterdir() if p.is_dir()):
            for pdf in sorted(season.glob("*test*.pdf")):
                tests.append(Test(id=f"{season.name}_{pdf.stem}", source=pdf))
        return tests

    def solution_source(self, test):
        """Sibling solution PDF: the test name with ``test`` swapped to ``soln``."""
        src = test.source
        sol = src.with_name(src.name.replace("test", "soln"))
        return sol if sol.exists() else None

    def answer_source(self, test):
        """The key lives inside the solutions PDF itself (its front box)."""
        return self.solution_source(test)

    def clean_solution_markdown(self, page_index, markdown):
        """Strip the front answer-key box so problem tagging sees only solutions.

        Left in place, the box's out-of-order entries ("4. 12" before "1. 5")
        would be read as problem markers, mis-numbering every solution and
        figure after them. The answers themselves are recovered separately by
        `parse_answers` from the raw markdown. The symmetric back cover at the
        document's end (`_strip_back_cover`) is dropped for the same reason: its
        credits/boilerplate would otherwise trail into the last solution.
        """
        text = _strip_back_cover(_split_answer_key(markdown)[1])
        return strip_solution_page_furniture(
            text, line_patterns=(_RUNNING_SOLUTION_FURNITURE_RE,)
        )

    def solution_figure_floor(self, pdf_page, image):
        """Fence out the back-cover credits box printed after the last solution.

        The figure-side partner of `_strip_back_cover`: the solutions PDF's final
        page ends with a "© Proof School <year>" copyright line, then a boxed
        "Problem Credits" table. DETR crops that box as a Picture and binds it to
        the last problem (e.g. 2018-19 Round Four Regional's problem 7). The
        copyright line is the boundary where real content stops -- everything
        below it is furniture -- so any figure whose centre falls below it is
        dropped. The line is read from the born-digital text layer (topmost match,
        since only the last page carries it) and scaled to rendered coordinates;
        pages without it (every page but the last) return None and keep every
        figure.
        """
        ys = [
            line["bbox"][1]
            for block in pdf_page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            if _PDF_COPYRIGHT_RE.search("".join(s["text"] for s in line["spans"]))
        ]
        if not ys or not pdf_page.rect.height:
            return None
        return min(ys) * (image.height / pdf_page.rect.height)

    def parse_answers(self, test, pages_markdown):
        """Read the "Answer Key" box from the first page that carries one."""
        for markdown in pages_markdown:
            answers, _ = _split_answer_key(markdown)
            if answers:
                return answers
        return {}

    def duplicate_scope(self, test_id, across=False):
        """Bucket by season. Mandelbrot reuses problems across a season's
        sibling rounds (a National and Regional individual round, or the
        team-play round, of the same season share problems), so every
        ``<season>_*`` test -- ``2017-18_tmctest1N``, ``2017-18_tmctest1R``,
        ``2017-18_mtptest1``, ... -- is compared together, but never against a
        different season's tests.

        With ``across=True`` (``dedup --across-years``) the season is dropped
        and every test shares one bucket, catching problems recycled across
        seasons."""
        if across:
            return "all"
        return test_id.split("_", 1)[0]

    # match_marker stays default: rounds number "1." / "1)" normally. Add an
    # override here (mirroring UsamtsSeries) if a real run reveals a quirk.
