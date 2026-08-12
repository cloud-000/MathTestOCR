"""Orchestration: page image -> structured problems.

Engines:
  * mlx       -- detect -> OCR each text box -> find anchors -> group -> assemble.
                 No global VLM reasoning; all segmentation is deterministic geometry.
  * nanonets  -- one whole-page OCR pass returns problem-segmented markdown with
                 inline <img> tags; DETR supplies only the image crops, mapped to
                 problems by reading-order ordinal (see process_image_markdown).
  * llama     -- same whole-page-markdown path as nanonets, but the page is OCR'd
                 by the hosted LlamaCloud parsing API instead of the local
                 endpoint (see src/llama.py). Both are config.MARKDOWN_ENGINES,
                 so process_image_markdown drives either interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import TYPE_CHECKING

from PIL import Image

from . import anchors as anchors_mod
from . import config, detect, grouping
from . import nanonets as nanonets_mod

if TYPE_CHECKING:
    from .ocr import OCRModel


_FIGURE_CUE_RE = re.compile(
    r"\b(?:shown|diagram|figure|graph|grid|number line|chart|pictured|"
    r"illustration|below|at right)\b",
    re.IGNORECASE,
)
_SPONSOR_WATERMARK_RE = re.compile(
    r"<watermark\b[^>]*>[^<]*(?:lockheed(?:\s+martin)?|raytheon)[^<]*"
    r"</watermark>|printing\s+of\s+this\s+competition\s+is\s+underwritten\s+by",
    re.IGNORECASE,
)


@dataclass
class ProblemElement:
    kind: str  # "text" | "image"
    label: str  # DETR label
    box: list
    text: str = ""
    crop: Image.Image | None = None


@dataclass
class Problem:
    number: int
    elements: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(e.text for e in self.elements if e.kind == "text")


def _ocr_text_boxes(detections, image, ocr: OCRModel):
    """Plain-text transcription for every text-like box (used for anchors)."""
    for d in detections:
        if d["label"] in config.TEXT_LABELS:
            crop = image.crop(tuple(d["box"]))
            d["text"] = ocr.read_text(crop)
        else:
            d["text"] = ""


def _assemble(groups, image, ocr: OCRModel, match=None):
    problems = []
    for number in sorted(groups):
        dets = sorted(groups[number], key=lambda d: (round(d["box"][1] / 10), d["box"][0]))
        elements = []
        for d in dets:
            box = d["box"]
            crop = image.crop(tuple(box))
            if d["label"] in config.IMAGE_LABELS:
                elements.append(ProblemElement("image", d["label"], box, crop=crop))
            else:
                # Re-OCR as LaTeX for the final output. The crop still shows the
                # printed problem number for inline anchors, so strip it back off.
                latex = ocr.latex_ocr(crop)
                if d.get("is_anchor") and d.get("marker_inline"):
                    latex = anchors_mod.strip_leading_marker(latex, match)
                elements.append(ProblemElement("text", d["label"], box, text=latex))
        problems.append(Problem(number=number, elements=elements))
    return problems


def process_image(image_path, ocr: OCRModel, threshold=config.DETECT_THRESHOLD, match=None):
    """Run the full pipeline on one page. Returns (problems, detections, groups).

    `match` is an optional series-specific marker matcher (see
    anchors._match_marker) for competition-specific numbering quirks.
    """
    if config.PRINT_TIME:
        from datetime import datetime
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Parsing page {image_path}...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calling OCR for page {image_path} (mlx)...")
    image = Image.open(image_path).convert("RGB")
    detections = detect.detect(image, threshold)
    if not detections:
        return [], [], {}

    _ocr_text_boxes(detections, image, ocr)
    anchors = anchors_mod.detect_anchors(detections, image.width, match)

    if anchors:
        groups = grouping.group_by_anchors(detections, anchors, image)
    else:
        groups = grouping.fallback_group_by_gaps(detections, image, image.height)

    problems = _assemble(groups, image, ocr, match)
    return problems, detections, groups


def _sorted_pictures(detections, image, max_area_frac=None, header_frac=None,
                     right_margin_frac=None, footer_frac=None, min_height_frac=None,
                     equation_text_overlap=None, equation_text_boxes=None,
                     equation_picture_min_aspect=None, text_layer_coverage=None,
                     pdf_page=None):
    """Non-blank DETR Picture detections, top-to-bottom (reading order).

    `max_area_frac` (from a series' LayoutOptions) optionally drops any Picture
    covering more than that fraction of the page -- a whole-page layout
    misclassification. None (the default) keeps every non-blank Picture.

    `header_frac` (also from LayoutOptions) optionally drops any Picture whose
    vertical center is within that fraction of the page height from the top --
    the running-header logo/title band (see config.header_picture_frac).

    `right_margin_frac` (also from LayoutOptions) optionally drops any Picture
    whose right edge reaches into that fraction of the page width at the right --
    the answer/scoring gutter (see config.right_margin_picture_frac).

    `footer_frac` is the vertical mirror of `header_frac`: drop any Picture whose
    vertical center is within that fraction of the page height from the bottom --
    the running page-footer band (see config.footer_picture_frac).

    `min_height_frac` (also from LayoutOptions) optionally drops any Picture
    shorter than that fraction of the page height -- inline equations/symbols
    that DETR emits as short strips at a low threshold (see
    config.min_picture_height_frac).

    `equation_text_overlap` (also from LayoutOptions) optionally drops any *wide*
    Picture (aspect ratio over config.EQUATION_PICTURE_MIN_ASPECT) whose area is
    more than that fraction covered by a Text detection -- a display equation
    (see config.equation_text_overlap). `equation_text_boxes` supplies the boxes
    to test coverage against; when None the Text-labeled boxes in `detections`
    are used, but a caller running figures below the text threshold passes a
    lower-confidence text set so equations DETR was unsure about are still caught.

    `text_layer_coverage` (also from LayoutOptions) is the born-digital version of
    that filter, needing no aspect guard: with `pdf_page` supplied, drop any
    Picture the source PDF's own glyphs tile that densely and no vector path is
    drawn inside (see config.text_layer_equation_coverage). Both are None/absent
    by default, and either may be used without the other.
    """
    pics = [
        d
        for d in detections
        if d["label"] == "Picture" and not grouping.is_blank_crop(image, d["box"])
    ]
    if max_area_frac is not None:
        max_area = max_area_frac * image.width * image.height
        pics = [d for d in pics if _box_area(d["box"]) <= max_area]
    if header_frac is not None:
        cutoff = header_frac * image.height
        pics = [d for d in pics if (d["box"][1] + d["box"][3]) / 2 > cutoff]
    if right_margin_frac is not None:
        cutoff = (1 - right_margin_frac) * image.width
        pics = [d for d in pics if d["box"][2] < cutoff]
    if footer_frac is not None:
        cutoff = (1 - footer_frac) * image.height
        pics = [d for d in pics if (d["box"][1] + d["box"][3]) / 2 < cutoff]
    if min_height_frac is not None:
        min_h = min_height_frac * image.height
        pics = [d for d in pics if (d["box"][3] - d["box"][1]) >= min_h]
    if equation_text_overlap is not None:
        text_boxes = (
            equation_text_boxes
            if equation_text_boxes is not None
            else [d["box"] for d in detections if d["label"] in config.TEXT_LABELS]
        )

        def _is_equation(box):
            w, h = box[2] - box[0], box[3] - box[1]
            min_aspect = (
                config.EQUATION_PICTURE_MIN_ASPECT
                if equation_picture_min_aspect is None
                else equation_picture_min_aspect
            )
            if h <= 0 or w / h <= min_aspect:
                return False
            return any(_contained_frac(box, t) > equation_text_overlap for t in text_boxes)

        pics = [d for d in pics if not _is_equation(d["box"])]
    if text_layer_coverage is not None and pdf_page is not None:
        pics = _drop_text_layer_equations(pics, pdf_page, image, text_layer_coverage)
    pics = _drop_nested_pictures(pics)
    pics.sort(key=lambda d: (d["box"][1], d["box"][0]))
    return pics


def _box_area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def _rect_union_area(rects, clip):
    """Area of the union of `rects`, clipped to `clip`.

    A union rather than a sum: glyph boxes on the same line overlap slightly, and
    double-counting them would inflate a figure's label coverage. Exact, via
    coordinate compression -- the inputs are the handful of words inside one
    Picture box, so the cell scan is cheap.
    """
    boxes = []
    for rect in rects:
        x0, y0 = max(rect[0], clip[0]), max(rect[1], clip[1])
        x1, y1 = min(rect[2], clip[2]), min(rect[3], clip[3])
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    if not boxes:
        return 0.0
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    ys = sorted({y for b in boxes for y in (b[1], b[3])})
    total = 0.0
    for x0, x1 in zip(xs, xs[1:]):
        for y0, y1 in zip(ys, ys[1:]):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes):
                total += (x1 - x0) * (y1 - y0)
    return total


def _drop_text_layer_equations(pics, pdf_page, image, min_coverage):
    """Drop the Pictures the source PDF shows to be display equations.

    A display equation is set from the same glyphs as the prose around it, so the
    text layer tiles its whole box and no vector path is drawn inside it; a real
    figure is the inverse -- drawn rules and edges, with a few point labels. That
    reading of the born-digital source needs no aspect-ratio guard, so it catches
    the roughly-square stacked fractions and summations `equation_text_overlap`
    is built to leave alone. A page with no text layer (a scan) yields no words
    and nothing is dropped.
    """
    if pdf_page.rect.width <= 0 or pdf_page.rect.height <= 0:
        return pics
    sx = image.width / pdf_page.rect.width
    sy = image.height / pdf_page.rect.height

    def scaled(rect):
        return (rect[0] * sx, rect[1] * sy, rect[2] * sx, rect[3] * sy)

    words = [scaled(word[:4]) for word in pdf_page.get_text("words")]
    if not words:
        return pics
    drawings = [scaled(drawing["rect"]) for drawing in pdf_page.get_drawings()]

    def is_equation(box):
        area = _box_area(box)
        if not area:
            return False
        if any(
            _contained_frac(d, box) > config.TEXT_LAYER_DRAWING_FRAC for d in drawings
        ):
            return False
        return _rect_union_area(words, box) / area >= min_coverage

    return [d for d in pics if not is_equation(d["box"])]


def _drop_solution_answer_boxes(
    pics, pdf_page, image, max_vector_box_width_frac=0.30
):
    """Drop compact Pictures enclosing text on a born-digital ``Answer:`` row.

    A boxed answer has figure-like borders, so DETR can label it Picture even
    though the PDF text layer still exposes the value inside. Prefer an
    ``Answer:`` row when present, and also recognize the compact axis-aligned
    vector rectangle TeX draws around a boxed value. The latter covers CHMM
    packets that put a bare box at the start or end of a solution.
    """
    words = pdf_page.get_text("words")
    if not words or pdf_page.rect.width <= 0 or pdf_page.rect.height <= 0:
        return pics
    sx = image.width / pdf_page.rect.width
    sy = image.height / pdf_page.rect.height
    answer_words = [
        word
        for word in words
        if word[4].strip().rstrip(":").casefold() == "answer"
    ]

    def shares_answer_row(word):
        for answer in answer_words:
            if word[5] != answer[5] or word[0] < answer[2]:
                continue
            overlap = max(0, min(word[3], answer[3]) - max(word[1], answer[1]))
            min_height = min(word[3] - word[1], answer[3] - answer[1])
            if min_height > 0 and overlap / min_height >= 0.5:
                return True
        return False

    value_boxes = [
        (word[0] * sx, word[1] * sy, word[2] * sx, word[3] * sy)
        for word in words
        if word[4].strip().rstrip(":").casefold() != "answer"
        and shares_answer_row(word)
    ]
    max_answer_row_width = image.width * 0.30
    # DETR sometimes joins a boxed final value to the display equation leading
    # into it. Allow that wider crop only when the PDF supplies the stronger
    # evidence of a reconstructed vector box containing text.
    max_vector_box_width = image.width * max_vector_box_width_frac
    max_height = image.height * 0.08

    # PyMuPDF reports the four sides of a TeX \boxed value as separate line
    # drawings. Reassemble small matching horizontal/vertical segments into
    # rectangles and require text inside, which avoids treating an unlabeled
    # diagram edge as an answer box.
    horizontal = []
    vertical = []
    for drawing in pdf_page.get_drawings():
        for item in drawing.get("items", ()):
            if not item or item[0] != "l":
                continue
            p0, p1 = item[1], item[2]
            if abs(p0.y - p1.y) <= 0.75:
                horizontal.append(
                    (min(p0.x, p1.x), (p0.y + p1.y) / 2, max(p0.x, p1.x))
                )
            elif abs(p0.x - p1.x) <= 0.75:
                vertical.append(
                    ((p0.x + p1.x) / 2, min(p0.y, p1.y), max(p0.y, p1.y))
                )

    vector_boxes = []
    tolerance = 1.5
    for top in horizontal:
        for bottom in horizontal:
            if bottom[1] <= top[1] + 4:
                continue
            if bottom[1] - top[1] > pdf_page.rect.height * 0.08:
                continue
            if abs(top[0] - bottom[0]) > tolerance or abs(top[2] - bottom[2]) > tolerance:
                continue
            if top[2] - top[0] > pdf_page.rect.width * 0.20:
                continue
            left = any(
                abs(line[0] - top[0]) <= tolerance
                and abs(line[1] - top[1]) <= tolerance
                and abs(line[2] - bottom[1]) <= tolerance
                for line in vertical
            )
            right = any(
                abs(line[0] - top[2]) <= tolerance
                and abs(line[1] - top[1]) <= tolerance
                and abs(line[2] - bottom[1]) <= tolerance
                for line in vertical
            )
            if not (left and right):
                continue
            rect = (top[0], top[1], top[2], bottom[1])
            if any(
                rect[0] - 1 <= (word[0] + word[2]) / 2 <= rect[2] + 1
                and rect[1] - 1 <= (word[1] + word[3]) / 2 <= rect[3] + 1
                for word in words
            ):
                vector_boxes.append(
                    (rect[0] * sx, rect[1] * sy, rect[2] * sx, rect[3] * sy)
                )

    def is_answer_box(pic):
        box = pic["box"]
        width, height = box[2] - box[0], box[3] - box[1]
        if height > max_height:
            return False
        on_answer_row = width <= max_answer_row_width and any(
            box[0] <= (value[0] + value[2]) / 2 <= box[2]
            and box[1] <= (value[1] + value[3]) / 2 <= box[3]
            for value in value_boxes
        )
        encloses_boxed_text = width <= max_vector_box_width and any(
            box[0] - 4 <= (value[0] + value[2]) / 2 <= box[2] + 4
            and box[1] - 4 <= (value[1] + value[3]) / 2 <= box[3] + 4
            for value in vector_boxes
        )
        return on_answer_row or encloses_boxed_text

    return [pic for pic in pics if not is_answer_box(pic)]


def _without_answer_table_picture(detections, image, markdown, enabled):
    """Drop a wide lower-page response grid identified by its OCR table."""
    if not enabled or not re.search(
        r"(?is)(?:^|\n)\s*(?:[*_#]+\s*)*(?:\d+\s+)?Answers?\b"
        r"[\s\S]{0,300}<table>",
        markdown,
    ):
        return detections
    return [
        det
        for det in detections
        if not (
            det["label"] in config.IMAGE_LABELS
            and det["box"][2] - det["box"][0] >= image.width * 0.50
            and (det["box"][1] + det["box"][3]) / 2 >= image.height * 0.45
        )
    ]


def _contained_frac(inner, outer):
    """Fraction of `inner`'s area that lies inside `outer`."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = _box_area(inner)
    return inter / area if area else 0.0


def _drop_nested_pictures(pics):
    """Drop each Picture nested inside a strictly larger one (keep the enclosing).

    DETR frequently returns both a figure group and its individual panels; a
    problem should get one crop of the whole figure, not the whole plus each
    part (see config.NESTED_PICTURE_FRAC).
    """
    kept = []
    for d in pics:
        da = _box_area(d["box"])
        nested = any(
            o is not d
            and _box_area(o["box"]) > da
            and _contained_frac(d["box"], o["box"]) >= config.NESTED_PICTURE_FRAC
            for o in pics
        )
        if not nested:
            kept.append(d)
    return kept


def _problem_start_ys(detections, image, headers_only=False):
    """Vertical positions where problems start, from DETR's left-margin text boxes.

    A content text box whose left edge sits at the page's left text margin begins
    a problem (its statement/number). Centered headers and indented continuation
    lines start further right and are excluded; boxes on the same row are merged.
    Returns the start y_top values sorted top-to-bottom.

    `headers_only` restricts the candidates to heading boxes (config.HEADER_LABELS):
    for series whose problem number sits on its own heading line, the statement
    below it is a separate left-margin box and would otherwise be counted as a
    second start (see LayoutOptions.problem_start_from_headers). Falls back to the
    full text-box scan if the page has no left-margin heading.

    In the default (body-text) mode the page title / section headers
    (config.HEADER_LABELS) are dropped first: they are furniture, not problem
    starts, and a title that begins slightly left of the body column would
    otherwise define `left` and push every real problem out of `x_tol` (yielding
    a single bogus start), while a section header between problems would add a
    spurious one.
    """
    labels = config.HEADER_LABELS if headers_only else config.TEXT_LABELS
    cand = [
        d
        for d in detections
        if d["label"] in labels and not grouping.is_blank_crop(image, d["box"])
    ]
    if not headers_only:
        body = [d for d in cand if d["label"] not in config.HEADER_LABELS]
        cand = body or cand
    if not cand and headers_only:
        return _problem_start_ys(detections, image)
    if not cand:
        return []
    left = min(d["box"][0] for d in cand)
    x_tol = config.NANONETS_START_X_TOL_FRAC * image.width
    ys = sorted(d["box"][1] for d in cand if d["box"][0] <= left + x_tol)
    starts = []
    for y in ys:
        if starts and y - starts[-1] <= config.Y_TOL:
            continue  # same row as the previous start (e.g. number box + statement)
        starts.append(y)
    return starts


def _gap_based_starts(detections, image, n):
    """Split all non-blank text content into exactly `n` top-to-bottom bands.

    Fallback for when `_problem_start_ys`'s left-margin heuristic mis-counts --
    e.g. a problem's own number/blank box scored just under the detection
    threshold, so it never became a left-margin candidate and that problem's
    start silently disappeared (its statement box, further right, doesn't
    qualify either). Nanonets' own count of problems on the page (`n`) is
    reliable, so instead: merge every candidate box into non-overlapping
    vertical intervals and cut at the `n - 1` largest gaps between them,
    regardless of x-position. Returns `n` y_top values, or fewer if there
    isn't enough distinct content to split.
    """
    cand = [
        d
        for d in detections
        if d["label"] in config.TEXT_LABELS and not grouping.is_blank_crop(image, d["box"])
    ]
    if not cand:
        return []
    intervals = sorted((d["box"][1], d["box"][3]) for d in cand)
    merged = [intervals[0]]
    for y0, y1 in intervals[1:]:
        if y0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], y1))
        else:
            merged.append((y0, y1))
    if len(merged) <= n:
        return [y0 for y0, _ in merged]
    gaps = sorted(
        range(len(merged) - 1), key=lambda i: merged[i + 1][0] - merged[i][1], reverse=True
    )
    cut_after = sorted(gaps[: n - 1])
    starts = [merged[0][0]]
    starts.extend(merged[i + 1][0] for i in cut_after)
    return starts


def _point_marker_row_starts(image, right_margin_frac, header_frac, footer_frac, n):
    """Return problem-row boundaries from point markers in the right gutter.

    Some fixed-table layouts print one circled point value (or ballot box) at
    the vertical centre of every problem row. Their dark, roughly-square
    outlines are more stable than DETR's left-margin text segmentation. Find
    those outlines directly in the gutter, then put each boundary halfway
    between adjacent marker centres so a figure anywhere within a row maps to
    that row rather than requiring its centre to sit below the point marker.

    This is deliberately fail-closed: unless exactly ``n`` plausible markers
    survive the series' header/footer bands, return no fallback at all.
    """
    if right_margin_frac is None or n <= 0:
        return []
    width, height = image.size
    x0 = round((1 - right_margin_frac) * width)
    y0 = round((header_frac or 0) * height)
    y1 = round((1 - (footer_frac or 0)) * height)
    if x0 >= width or y0 >= y1:
        return []

    # Threshold once through Pillow's lookup-table path, then scan row bytes.
    # The printed outlines are near-black; 200 retains antialiasing while the
    # white/off-white paper stays empty.
    ink = image.convert("L").crop((x0, y0, width, y1)).point(
        lambda px: 255 if px < config.POINT_MARKER_INK_THRESHOLD else 0
    )
    gutter_w, gutter_h = ink.size
    data = ink.tobytes()
    # The problem table's right border runs through this same gutter. It cleanly
    # separates the row rectangles (left) from the point markers (right), so
    # restrict the scan to the marker side. Merely erasing the border is not
    # enough: on short rows a horizontal rule can sit within a few pixels of a
    # circle and merge their y-runs despite the two shapes never touching.
    border_cols = []
    for x in range(gutter_w):
        ink_count = sum(bool(data[y * gutter_w + x]) for y in range(gutter_h))
        if ink_count > config.POINT_MARKER_VERTICAL_LINE_FRAC * gutter_h:
            border_cols.append(x)
    if border_cols and max(border_cols) + 1 < gutter_w:
        ink = ink.crop((max(border_cols) + 1, 0, gutter_w, gutter_h))
        gutter_w, gutter_h = ink.size
    data = ink.tobytes()
    active = [
        any(data[y * gutter_w : (y + 1) * gutter_w])
        for y in range(gutter_h)
    ]

    # Merge tiny antialiasing gaps within an outline, but not separate rows.
    bands = []
    start = previous = None
    for y, has_ink in enumerate(active):
        if not has_ink:
            continue
        if start is None:
            start = previous = y
        elif y - previous <= config.POINT_MARKER_ROW_GAP:
            previous = y
        else:
            bands.append((start, previous + 1))
            start = previous = y
    if start is not None:
        bands.append((start, previous + 1))

    min_h = config.POINT_MARKER_HEIGHT_FRAC[0] * height
    max_h = config.POINT_MARKER_HEIGHT_FRAC[1] * height
    min_aspect, max_aspect = config.POINT_MARKER_ASPECT
    centers = []
    for top, bottom in bands:
        bbox = ink.crop((0, top, gutter_w, bottom)).getbbox()
        if bbox is None:
            continue
        marker_w = bbox[2] - bbox[0]
        marker_h = bbox[3] - bbox[1]
        if min_h <= marker_h <= max_h and min_aspect <= marker_w / marker_h <= max_aspect:
            centers.append(y0 + top + (bbox[1] + bbox[3]) / 2)
    if len(centers) != n:
        return []
    if n == 1:
        return [centers[0]]
    starts = [centers[0] - (centers[1] - centers[0]) / 2]
    starts.extend((a + b) / 2 for a, b in zip(centers, centers[1:]))
    return starts


def _assign_pictures(detections, image, problem_seq, layout, items=None, carry=None,
                     equation_text_boxes=None, engine="ocr", pdf_page=None):
    """Map each non-blank DETR Picture to a problem number by vertical position.

    `problem_seq` is the ordered list of problem numbers (top-to-bottom). Each
    picture is assigned to the problem whose start is the lowest one at or above
    the picture's vertical centre. A picture above this page's first problem
    belongs to `carry` (the problem continued from the previous page) when one is
    given -- its figure spilled onto this page; with no carry (the first page) it
    is page furniture (header/logo) and is dropped. Returns {problem_number:
    [picture_det, ...]}.

    `layout` is the series' LayoutOptions: it controls the page-spanning Picture
    filter (`max_picture_area_frac`) and whether the gap-based problem-start
    fallback is used when DETR's left-margin count disagrees with nanonets'.

    `items` is the page's parse_layout output. Problem-tagged inline <img>
    positions can replace untrustworthy geometry, but only when their count
    matches both the retained Pictures and the pre-position-filter candidate
    count. That last guard prevents a filtered right-side figure from making an
    unrelated remaining crop accidentally line up with the wrong tag.
    """
    pre_position_pics = _sorted_pictures(
        detections,
        image,
        layout.max_picture_area_frac,
        None,
        None,
        None,
        layout.min_picture_height_frac,
        layout.equation_text_overlap,
        equation_text_boxes,
        layout.equation_picture_min_aspect,
        layout.text_layer_equation_coverage,
        pdf_page,
    )
    pics = _sorted_pictures(
        detections,
        image,
        layout.max_picture_area_frac,
        layout.header_picture_frac,
        layout.right_margin_picture_frac,
        layout.footer_picture_frac,
        layout.min_picture_height_frac,
        layout.equation_text_overlap,
        equation_text_boxes,
        layout.equation_picture_min_aspect,
        layout.text_layer_equation_coverage,
        pdf_page,
    )
    if not pics:
        return {}
    if not problem_seq:
        # No problem starts on this page: it only continues the previous page's
        # problem (a figure that spilled over, no new statement), so every crop
        # belongs to that carried-in problem. Without a carry there is nothing to
        # attach them to.
        return {carry: list(pics)} if carry is not None else {}
    starts = _problem_start_ys(detections, image, layout.problem_start_from_headers)
    indexed = list(enumerate(items or []))
    img_tags = [
        (i, it["problem"])
        for i, it in indexed
        if it["kind"] == "image" and it["problem"] is not None
    ]

    def _appended(idx, prob):
        return not any(
            it["kind"] == "text"
            and it["problem"] is not None
            and it["problem"] > prob
            for j, it in indexed
            if j > idx
        )

    tag_numbers = [number for _, number in img_tags]
    increasing_tags = all(
        previous < current
        for previous, current in zip(tag_numbers, tag_numbers[1:])
    )
    text_by_problem = {}
    for _, item in indexed:
        if item["kind"] == "text" and item["problem"] is not None:
            text_by_problem.setdefault(item["problem"], []).append(item["text"])
    cue_problems = {
        number
        for number, parts in text_by_problem.items()
        if _FIGURE_CUE_RE.search("\n".join(parts))
    }
    trusted_tags = (
        bool(img_tags)
        and len(img_tags) == len(pics) == len(pre_position_pics)
        and increasing_tags
        and not any(_appended(i, number) for i, number in img_tags)
    )

    # One printed illustration may be detected as several adjacent Picture
    # crops even though OCR emits one inline <img> marker for the whole group
    # (PUMaC's four-symbol alphabet is the canonical example). Cluster
    # vertically overlapping crops and trust the tags at cluster granularity
    # when every tagged problem also has an explicit figure cue. This is much
    # safer than zipping individual crops to tags and avoids letting unreliable
    # left-margin text geometry shift the entire page by several problems.
    picture_clusters = []
    cluster_slop = image.height * 0.02
    for pic in pics:
        y0, y1 = pic["box"][1], pic["box"][3]
        if not picture_clusters or y0 > picture_clusters[-1][1] + cluster_slop:
            picture_clusters.append([[pic], y1])
        else:
            picture_clusters[-1][0].append(pic)
            picture_clusters[-1][1] = max(picture_clusters[-1][1], y1)
    trusted_cluster_tags = (
        bool(img_tags)
        and len(picture_clusters) == len(img_tags)
        and tag_numbers == sorted(tag_numbers)
        and set(tag_numbers).issubset(cue_problems)
        and not any(_appended(i, number) for i, number in img_tags)
    )

    def _groups_from_tags():
        groups = {}
        for pic, (_, number) in zip(pics, img_tags):
            groups.setdefault(number, []).append(pic)
        return groups

    def _groups_from_cluster_tags():
        groups = {}
        for (cluster, _), (_, number) in zip(picture_clusters, img_tags):
            groups.setdefault(number, []).extend(cluster)
        return groups

    # Geometry is the primary signal, but only trustworthy when DETR found exactly
    # one start per problem: a figure's vertical position then pins it to the
    # problem whose text sits directly above. When the counts disagree the starts
    # have drifted (a statement split into several left-margin boxes, or a missed
    # one), so before falling back to fuzzy geometry try nanonets' inline <img>
    # tags (zip pics top-to-bottom to the tags in reading order). This runs only
    # on a start/problem mismatch -- when the starts agree geometry is preferred,
    # because nanonets' tag order is the less reliable of the two.
    if len(starts) != len(problem_seq):
        if trusted_cluster_tags:
            return _groups_from_cluster_tags()
        # Trust the tags only when their problem-tagged count matches DETR's
        # pictures *and* none is "appended" past its problem: nanonets sometimes
        # dumps a figure's <img> at the very bottom of the page (after the last
        # problem) instead of beside it, mis-tagging it -- e.g. an octagon diagram
        # belonging to problem 4 emitted after problem 5. The tell is that no
        # later, higher-numbered problem's text follows the tag (trailing footer
        # text inherits the last problem's number, so a plain position check is
        # not enough); when that happens we distrust the tags and keep geometry.
        if trusted_tags and not layout.prefer_inline_picture_tags:
            return _groups_from_tags()
        if layout.point_marker_row_anchor:
            marker_starts = _point_marker_row_starts(
                image,
                layout.right_margin_picture_frac,
                layout.header_picture_frac,
                layout.footer_picture_frac,
                len(problem_seq),
            )
            starts = marker_starts or starts
        if len(starts) != len(problem_seq) and layout.gap_based_picture_fallback:
            starts = _gap_based_starts(detections, image, len(problem_seq)) or starts
    groups = {}
    if not starts:
        # No usable text geometry: keep every figure, on the first problem.
        groups[problem_seq[0]] = list(pics)
    else:
        if len(starts) != len(problem_seq):
            print(
                f"[{engine}] problem-count mismatch: {len(problem_seq)} problem(s) from "
                f"text vs {len(starts)} from DETR layout; figure assignment may drift"
            )
        for pic in pics:
            yc = (pic["box"][1] + pic["box"][3]) / 2
            if yc + config.Y_TOL < starts[0]:
                # Above this page's first problem: a figure carried over from the
                # previous page's problem, or (no carry) a page header / logo to drop.
                if carry is not None:
                    groups.setdefault(carry, []).append(pic)
                continue
            idx = 0
            for i, sy in enumerate(starts):
                if sy <= yc + config.Y_TOL:
                    idx = i
                else:
                    break
            idx = min(idx, len(problem_seq) - 1)
            groups.setdefault(problem_seq[idx], []).append(pic)

    if trusted_tags and layout.prefer_inline_picture_tags:
        # Reconcile each tag with geometry instead of replacing every assignment
        # wholesale. A one-row geometry drift is corrected when it puts a crop
        # on a statement with no figure cue and the tag points to one that
        # explicitly needs a figure. Conversely, if geometry already lands on a
        # figure-bearing statement, keep it: DETR may have missed a different
        # figure and made the remaining Picture/tag counts coincide by accident.
        geometry_problem = {
            id(pic): number
            for number, assigned in groups.items()
            for pic in assigned
        }
        reconciled = {}
        for pic, (_, tagged_number) in zip(pics, img_tags):
            geometric_number = geometry_problem.get(id(pic))
            if (
                tagged_number in cue_problems
                and geometric_number not in cue_problems
            ):
                number = tagged_number
            else:
                number = geometric_number
            if number is not None:
                reconciled.setdefault(number, []).append(pic)
        return reconciled
    return groups


def _without_sponsor_watermark_picture(detections, image, raw_markdown, enabled):
    """Remove a sponsor logo only when OCR independently identifies its footer.

    Older MATHCOUNTS pages put a very wide Lockheed Martin or Raytheon mark near
    the bottom edge. A general footer band is unsafe because real diagrams also
    extend that low. The explicit raw-OCR watermark/underwriting marker makes
    the filter page-specific; selecting only the lowest wide Picture keeps any
    other figures on the page.
    """
    if not enabled or not _SPONSOR_WATERMARK_RE.search(raw_markdown):
        return detections
    width, height = image.size
    candidates = []
    for detection in detections:
        if detection["label"] not in config.IMAGE_LABELS:
            continue
        x1, y1, x2, y2 = detection["box"]
        box_height = y2 - y1
        if box_height <= 0:
            continue
        center_y = (y1 + y2) / 2
        aspect = (x2 - x1) / box_height
        # Sponsor marks are short wordmarks (the observed Lockheed crops are
        # about 5.1--6.2 times wider than tall). Requiring both that shape and a
        # true footer position prevents a low number line or chart from becoming
        # the fallback candidate when DETR did not detect the logo.
        if (
            center_y >= 0.86 * height
            and box_height <= 0.08 * height
            and aspect >= 4.5
        ):
            candidates.append(detection)
    if not candidates:
        return detections
    logo = max(candidates, key=lambda detection: (
        (detection["box"][1] + detection["box"][3]) / 2,
        detection["box"][2] - detection["box"][0],
    ))
    return [detection for detection in detections if detection is not logo]


def _ocr_page(
    client,
    image,
    base_temp,
    cache=None,
    cache_key=None,
    mask_boxes=None,
    validate=None,
    merge_incomplete=None,
):
    """Whole-page OCR with runaway recovery; returns the page markdown.

    Nanonets' decoding can get stuck in a verbatim loop on a grid/dense figure
    (endless ``<td>`` rows, a repeated figure description). The stream guard
    aborts it, but that truncates every problem below it on the page. Instead of
    losing the page we recover it: re-OCR at each escalating
    `config.NANONETS_RETRY_TEMPS` above `base_temp` (breaking most loops), and as
    a guaranteed-terminating last resort blank the `mask_boxes` figure regions
    (whose interiors the text pass never needs -- DETR keeps the crops) and OCR
    once more. Only a clean (non-runaway) transcription is cached, so ``--cache``
    never re-serves a truncated page; if every rung still loops we return the
    best-effort truncated text uncached.

    `cache_key` is the page path used for cache lookup/store (None disables
    caching for this call). `mask_boxes` are the figure rectangles for the final
    rung; when empty the masking rung is skipped (e.g. answer pages with no
    detection pass).
    """
    engine = getattr(client, "name", "ocr")
    if config.PRINT_TIME:
        from datetime import datetime
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Parsing page {cache_key or 'unknown'}...")

    if cache is not None and cache_key is not None:
        hit = cache.get(cache_key)
        # Ignore a cached page that is itself a runaway. An older guard version
        # may have stored a loop it couldn't yet detect (nanonets._is_runaway has
        # since learned new loop shapes); serving it would silently drop every
        # problem below the loop. Re-OCR heals it and re-caches a clean pass.
        if (
            hit is not None
            and not nanonets_mod._is_runaway(hit)
            and (validate is None or validate(hit))
        ):
            if config.PRINT_TIME:
                from datetime import datetime
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Page {cache_key} found in cache.")
            return hit
    temps = [base_temp] + [t for t in config.NANONETS_RETRY_TEMPS if t > base_temp]
    for i, temp in enumerate(temps):
        if i:
            print(f"[{engine}] runaway; retrying at temperature {temp}")
        if config.PRINT_TIME:
            from datetime import datetime
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calling OCR for page {cache_key or 'unknown'} (temp={temp})...")
        markdown, runaway = client.parse_page(image, temp)
        if not runaway:
            if validate is None or validate(markdown):
                break
            # A cleanly terminated but incomplete response normally exhausted
            # the model's output budget. Temperature retries cannot add budget;
            # continue below with bounded vertical slices instead.
            break
    else:
        if mask_boxes:
            print(f"[{engine}] runaway persists; masking figures and re-OCRing")
            if config.PRINT_TIME:
                from datetime import datetime
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calling OCR for page {cache_key or 'unknown'} (masking figures)...")
            markdown, runaway = client.parse_page(
                image, temps[-1], mask_boxes=mask_boxes
            )
    valid = validate is None or validate(markdown)
    if not valid:
        print(
            f"[{engine}] completeness validation failed; "
            "OCRing three vertical slices"
        )
        parts = []
        slice_runaway = False
        for top, bottom in _vertical_slices(image, count=3):
            crop = image.crop((0, top, image.width, bottom))
            crop_masks = []
            for box in mask_boxes or ():
                x0, y0, x1, y1 = box
                if y1 <= top or y0 >= bottom:
                    continue
                crop_masks.append(
                    (
                        x0,
                        max(0, y0 - top),
                        x1,
                        min(bottom - top, y1 - top),
                    )
                )
            part, part_runaway = client.parse_page(
                crop,
                temps[-1],
                mask_boxes=crop_masks or None,
            )
            parts.append(part)
            slice_runaway = slice_runaway or part_runaway
        combined = "\n\n".join(parts)
        if validate(combined):
            # Slices recover problem starts that a bounded whole-page response
            # missed, but they can lose a display equation or a line that
            # straddles a slice boundary.  When the caller can identify
            # problem blocks, keep the whole-page transcription and replace
            # only the span around its missing marker(s).  Falling back to the
            # combined slices retains the old, safe behavior for layouts that
            # cannot be block-merged.
            merged = (
                merge_incomplete(markdown, combined)
                if merge_incomplete is not None
                else None
            )
            markdown = merged if merged is not None and validate(merged) else combined
            runaway = slice_runaway
            valid = True
    if not valid:
        raise RuntimeError(
            f"{engine} OCR remained incomplete after all recovery attempts "
            f"for {cache_key or 'page'}; refusing to return truncated text"
        )
    if cache is not None and cache_key is not None and not runaway and valid:
        cache.put(cache_key, markdown)
    return markdown


def _marker_blocks(markdown, match_marker):
    """Return ordered ``(number, start, end)`` blocks from OCR markdown.

    This deliberately recognizes only line-leading markers.  It is used before
    ``parse_layout`` purely to localize a known omission; the real parser still
    owns all the series-specific rules for glued markers, tables, restarts, and
    numbered lists.  Repeated marker numbers are ambiguous (answer columns and
    section restarts), so a document with any repeats is not mergeable.
    """
    starts = []
    offset = 0
    for line in markdown.splitlines(keepends=True):
        probe = line.lstrip().lstrip("*_# ")
        marker = match_marker(probe)
        if marker is not None:
            starts.append((marker[0], offset))
        offset += len(line)
    numbers = [number for number, _ in starts]
    if not starts or len(numbers) != len(set(numbers)):
        return None
    return [
        (number, start, starts[index + 1][1] if index + 1 < len(starts) else len(markdown))
        for index, (number, start) in enumerate(starts)
    ]


def _merge_incomplete_problem_blocks(whole, sliced, match_marker):
    """Fill missing whole-page problem blocks from a validated slice OCR.

    A missing marker often means that the whole-page response joined that
    problem onto the preceding block.  Replacing that predecessor-plus-gap
    range prevents the recovered statement from being duplicated under the
    preceding problem.  Blocks present in both transcriptions remain from the
    whole-page OCR, which is the important non-lossy property of this recovery.
    Returns ``None`` when the marker structure is not unambiguous.
    """
    whole_blocks = _marker_blocks(whole, match_marker)
    sliced_blocks = _marker_blocks(sliced, match_marker)
    if whole_blocks is None or sliced_blocks is None:
        return None
    whole_numbers = [number for number, _, _ in whole_blocks]
    sliced_numbers = [number for number, _, _ in sliced_blocks]
    missing = set(sliced_numbers) - set(whole_numbers)
    # The slice OCR must be a strict superset of the whole-page anchors.  This
    # avoids treating a differently-numbered section restart as a repair.
    if (
        not missing
        or not set(whole_numbers).issubset(sliced_numbers)
        or [number for number in sliced_numbers if number in whole_numbers]
        != whole_numbers
    ):
        return None

    sliced_by_number = {number: (start, end) for number, start, end in sliced_blocks}
    replacements = []
    index = 0
    while index < len(sliced_numbers):
        if sliced_numbers[index] not in missing:
            index += 1
            continue
        first = index
        while index + 1 < len(sliced_numbers) and sliced_numbers[index + 1] in missing:
            index += 1
        last = index

        # Include the immediately preceding known block: it is where a missing
        # marker's text was attached in the whole-page response.  If the first
        # page marker is missing, inserting its slice block before the first
        # whole marker is the only non-destructive option.
        start_index = first - 1 if first else first
        source_start = sliced_by_number[sliced_numbers[start_index]][0]
        next_number = sliced_numbers[last + 1] if last + 1 < len(sliced_numbers) else None
        source_end = (
            sliced_by_number[next_number][0]
            if next_number is not None
            else len(sliced)
        )

        if first:
            predecessor = sliced_numbers[first - 1]
            whole_start = next(start for number, start, _ in whole_blocks if number == predecessor)
        else:
            following = next_number
            whole_start = (
                next(start for number, start, _ in whole_blocks if number == following)
                if following is not None
                else len(whole)
            )
        whole_end = (
            next(start for number, start, _ in whole_blocks if number == next_number)
            if next_number is not None
            else len(whole)
        )
        replacements.append((whole_start, whole_end, sliced[source_start:source_end]))
        index += 1

    for start, end, replacement in reversed(replacements):
        whole = whole[:start] + replacement + whole[end:]
    return whole


def _vertical_slices(image, count=3):
    """Split a dense page near whitespace rows for bounded OCR responses."""
    if count <= 1 or image.height < count * 100:
        return [(0, image.height)]
    # Downscale each row to one grayscale pixel. The lightest row near each
    # target is the safest cut and avoids bisecting a line of mathematics.
    rows = image.convert("L").resize((1, image.height))
    brightness = list(rows.get_flattened_data())
    cuts = [0]
    radius = max(20, image.height // 20)
    for index in range(1, count):
        target = image.height * index // count
        lower = max(cuts[-1] + 50, target - radius)
        upper = min(image.height - 50, target + radius)
        cut = max(range(lower, upper + 1), key=lambda row: brightness[row])
        cuts.append(cut)
    cuts.append(image.height)
    return list(zip(cuts, cuts[1:]))


_RENDERED_PAGE_RE = re.compile(r"page_(\d+)")


def _pdf_page_for(doc, page_path):
    """The source PDF page a rendered page image came from, or None.

    `pdf_io` names rendered pages "page_<pdf page number>.png", and a page the
    series skipped leaves a gap in the list, so the filename is the only reliable
    index back into the document. A series that renders its pages some other way
    (or a test that is a folder of images, where `doc` is None) simply gets no
    text layer -- every filter that reads one is opt-in and skips when it is
    absent.
    """
    if doc is None:
        return None
    match = _RENDERED_PAGE_RE.fullmatch(Path(page_path).stem)
    if match is None:
        return None
    index = int(match.group(1)) - 1
    return doc[index] if 0 <= index < len(doc) else None


def _figure_mask_boxes(detections):
    """Figure rectangles (DETR Picture/Table) to blank on the masking rung."""
    return [d["box"] for d in detections if d["label"] in config.IMAGE_LABELS]


def process_image_markdown(
    image_path,
    client,
    threshold=config.NANONETS_DETECT_THRESHOLD,
    match_marker=None,
    cache=None,
    layout=None,
    carry=None,
    clean_markdown=None,
    validate_markdown=None,
    pdf_page=None,
):
    """Whole-page OCR via the nanonets engine; DETR supplies the image crops.

    Nanonets does all text transcription and problem segmentation. Figures are
    mapped to problems geometrically from DETR (see _assign_pictures); Nanonets'
    inline <img> tags are used only as a fallback when the geometry is ambiguous,
    since the model both invents them on text-only problems and omits them on real
    figures. Returns (problems, detections, groups); `groups` maps each problem to
    its Picture detections (debug overlay).

    `match_marker` is an optional series-specific marker matcher for
    competition-specific numbering quirks. `cache` is an optional OCRCache: when
    given, the (slow) whole-page OCR is served from / written to it, while DETR
    detection still runs every time. `layout` is the series' LayoutOptions (its
    table/figure heuristic knobs); None uses the conservative defaults. `carry` is
    the problem in progress at the top of this page (from the previous page of a
    multi-page test): it binds this page's leading text and any figure above the
    first problem to that problem instead of dropping them (see process_test).
    `pdf_page` is this page's source PDF page when the test was rendered from one
    -- the born-digital text layer behind `LayoutOptions.text_layer_equation_
    coverage`, mirroring what process_solution_document already gives the
    solution side. None (a folder of page images, or a series that has not opted
    into a text-layer filter) simply skips those filters.
    """
    layout = layout or config.LayoutOptions()
    engine = getattr(client, "name", "ocr")
    print(f"[{engine}] Starting pipeline...")
    image = Image.open(image_path).convert("RGB")
    print(f"[{engine}] Running layout detection (DETR)...")
    # Faint printed figures score below the text threshold, so a series can ask
    # for Picture/Table boxes at a lower confidence (picture_detect_threshold)
    # than the text used for problem-start geometry. Scan once at the lower of
    # the two, then keep text/masking at the text threshold and let only the
    # figures reach down to the picture threshold -- lowering the text threshold
    # would inject spurious left-margin starts and drift figure assignment.
    pic_thr = layout.picture_detect_threshold
    scan_thr = min(threshold, pic_thr) if pic_thr is not None else threshold
    scanned = detect.detect(image, scan_thr)
    detections = [d for d in scanned if d["score"] >= threshold]
    if pic_thr is not None and pic_thr < threshold:
        faint_figures = [d for d in scanned if d["label"] in config.IMAGE_LABELS]
        assign_detections = [
            d for d in detections if d["label"] not in config.IMAGE_LABELS
        ] + faint_figures
    else:
        assign_detections = detections
    # The equation_text_overlap filter needs to see the Text/Formula boxes DETR
    # placed under a faint figure to tell a display equation from a real diagram.
    # Those boxes often score below the text threshold, so gather them from the
    # low-confidence scan (down to config.EQUATION_TEXT_MIN_SCORE) -- kept out of
    # assign_detections proper so they never register as problem starts.
    equation_text_boxes = None
    if layout.equation_text_overlap is not None:
        equation_text_boxes = [
            d["box"]
            for d in scanned
            if d["label"] in config.TEXT_LABELS
            and d["score"] >= config.EQUATION_TEXT_MIN_SCORE
        ]
    print(f"[{engine}] Layout detection done.")
    print(f"[{engine}] Running whole-page OCR...")
    markdown = _ocr_page(
        client,
        image,
        layout.nanonets_temperature,
        cache=cache,
        cache_key=image_path,
        # Mask only the confident figures (the text-threshold detections); the
        # faint extras include page-spanning false positives that must never
        # blank the whole page on the masking rung.
        mask_boxes=_figure_mask_boxes(detections),
        validate=validate_markdown,
        merge_incomplete=(
            (lambda whole, sliced: _merge_incomplete_problem_blocks(
                whole, sliced, match_marker or nanonets_mod._match_marker
            ))
            if validate_markdown is not None
            else None
        ),
    )
    raw_markdown = markdown
    assign_detections = _without_sponsor_watermark_picture(
        assign_detections,
        image,
        raw_markdown,
        layout.drop_sponsor_watermark_picture,
    )
    assign_detections = _without_answer_table_picture(
        assign_detections,
        image,
        raw_markdown,
        layout.statement_answer_table_filter,
    )
    if clean_markdown is not None:
        markdown = clean_markdown(markdown)
        # A series cleanup that deliberately removes every byte has classified
        # this as a furniture-only page (MATHCOUNTS Target divider/score sheet).
        # Do not let its DETR logos fall through the no-starts branch and attach
        # to the problem carried from the preceding page.
        if raw_markdown.strip() and not markdown.strip():
            print(f"[{engine}] Page suppressed by series cleanup.")
            return [], detections, {}
    print(f"[{engine}] Whole-page OCR done.")
    items = nanonets_mod.parse_layout(
        markdown,
        match_marker,
        split_marker_table_rows=layout.split_marker_table_rows,
        start_problem=carry,
        ordered_list_markers=layout.ordered_list_markers,
        point_value_list_markers=layout.point_value_list_markers,
        point_value_marker_consistency=layout.point_value_marker_consistency,
        heading_problem_markers=layout.heading_problem_markers,
        strict_section_restarts=layout.strict_section_restarts,
        consecutive_problem_markers=layout.consecutive_problem_markers,
        page_initial_point_restart=layout.page_initial_point_restart,
        flat_problem_numbering=layout.flat_problem_numbering,
        backreference_problem_markers=layout.backreference_problem_markers,
        split_glued_bare_markers=layout.split_glued_bare_markers,
    )

    print(f"[{engine}] Assembling problems...")
    problems = {}  # number -> Problem, insertion-ordered (numbers increase)
    problem_seq = []  # problem numbers in reading order

    def problem_for(number):
        if number not in problems:
            problems[number] = Problem(number=number)
            problem_seq.append(number)
        return problems[number]

    for it in items:
        number = it["problem"]
        if number is None:  # page header (title/banner before problem 1)
            continue
        if it["kind"] != "text":
            # <img> descriptions are unreliable and the crop comes from DETR, but
            # when the series inlines figures we keep the tag's reading-order
            # position as a sentinel for inline_problem_figures to fill later.
            if layout.inline_figures:
                prob = problem_for(number)
                prob.elements.append(
                    ProblemElement("text", "Figure", [], text=nanonets_mod.FIGURE_PLACEHOLDER)
                )
            continue
        prob = problem_for(number)
        lines = [ln for ln in it["text"].splitlines() if not grouping.is_footer_text(ln)]
        text = "\n".join(lines).strip()
        if text:
            prob.elements.append(ProblemElement("text", "Text", [], text=text))

    # Geometric figure assignment from DETR. The carry problem continues from the
    # previous page, so it has no problem-start on this page to align against (a
    # marker equal to carry can't occur under the increasing guard, so it only
    # enters problem_seq via seeded continuation text) -- exclude it so the
    # remaining, page-starting problems line up 1:1 with DETR's starts. A figure
    # above the first start still goes to carry (handled inside _assign_pictures).
    geom_seq = [n for n in problem_seq if n != carry]
    groups = _assign_pictures(
        assign_detections, image, geom_seq, layout, items, carry, equation_text_boxes,
        engine=engine, pdf_page=pdf_page,
    )
    for number in sorted(groups):
        prob = problem_for(number)
        for pic in groups[number]:
            box = pic["box"]
            prob.elements.append(
                ProblemElement("image", "Picture", box, crop=image.crop(tuple(box)))
            )

    print(f"[{engine}] Problem assembly done.")
    return [problems[n] for n in sorted(problems)], detections, groups


def process_test(
    page_paths,
    engine,
    model,
    threshold=None,
    match=None,
    cache=None,
    layout=None,
    clean_page=None,
    validate_page=None,
    source_pdf=None,
):
    """Parse every page of a multi-page test and merge problems by number.

    A test (e.g. a USAMTS PDF rendered to page PNGs) may span several pages, with
    a single problem's statement and figures split across them. Each page is run
    through the chosen engine, then problems sharing a number are merged (their
    elements concatenated in page order). `engine` is a whole-page-markdown
    engine (config.MARKDOWN_ENGINES: "nanonets" / "llama") or "mlx"; `model` is
    the matching client (NanonetsClient / LlamaClient / OCRModel). `match` is the series-specific
    marker matcher. `cache` is an optional OCRCache for the nanonets whole-page OCR
    (ignored by the mlx engine, which OCRs per crop). `layout` is the series'
    LayoutOptions (nanonets only). Returns the merged [Problem], number-sorted.

    For the nanonets engine the highest problem number seen so far is carried into
    the next page (`carry`) so that page's leading text and any figure sitting
    above its first problem attach to the problem continued from the previous page
    instead of being dropped (see process_image_markdown / _assign_pictures).

    `source_pdf` is the PDF `page_paths` were rendered from, when there is one:
    each page's born-digital text layer is passed through for the figure filters
    that read it (`LayoutOptions.text_layer_equation_coverage`), the same service
    `process_solution_document` performs for solution pages. A folder of page
    images, or a series opting into none of those filters, passes None.
    """
    merged = {}  # number -> Problem
    carry = None  # nanonets: problem in progress at the top of the next page
    doc = None
    if source_pdf is not None and Path(source_pdf).suffix.lower() == ".pdf":
        import pymupdf

        doc = pymupdf.open(source_pdf)
    for page_index, path in enumerate(page_paths):
        pdf_page = _pdf_page_for(doc, path)
        if engine in config.MARKDOWN_ENGINES:
            thr = threshold if threshold is not None else config.NANONETS_DETECT_THRESHOLD
            problems, _, _ = process_image_markdown(
                path,
                model,
                thr,
                match_marker=match,
                cache=cache,
                layout=layout,
                carry=carry,
                clean_markdown=(
                    (lambda markdown, i=page_index: clean_page(i, markdown))
                    if clean_page is not None
                    else None
                ),
                validate_markdown=(
                    (lambda markdown, i=page_index: validate_page(i, markdown))
                    if validate_page is not None
                    else None
                ),
                pdf_page=pdf_page,
            )
            page_numbers = [p.number for p in problems]
            if page_numbers:
                page_max = max(page_numbers)
                carry = page_max if carry is None else max(carry, page_max)
        else:
            thr = threshold if threshold is not None else config.DETECT_THRESHOLD
            problems, _, _ = process_image(path, model, thr, match=match)
        for p in problems:
            if p.number not in merged:
                merged[p.number] = Problem(number=p.number)
            merged[p.number].elements.extend(p.elements)
    if doc is not None:
        doc.close()
    return [merged[n] for n in sorted(merged)]


def ocr_pages(page_paths, client, cache=None, layout=None):
    """Whole-page OCR of every page to markdown; one string per page.

    Unlike `process_test`, this runs no layout detection and does no problem
    segmentation -- it just returns the raw Nanonets markdown per page, in
    reading order. Used for answer-key documents (`Series.parse_answers`), whose
    parsers select pages themselves. `client` is a NanonetsClient. `cache` is an
    optional OCRCache serving / storing each page's markdown. `layout` is the
    series' LayoutOptions (only its OCR temperature is used here).
    """
    layout = layout or config.LayoutOptions()
    parts = []
    for path in page_paths:
        markdown = _ocr_page(
            client,
            Image.open(path).convert("RGB"),
            layout.nanonets_temperature,
            cache=cache,
            cache_key=path,
        )
        parts.append(markdown)
    return parts


def _find_gutter(rects, page_width):
    """x of the vertical gutter splitting two-column content, else None.

    `rects` are content bounding boxes in rendered coordinates. Blocks wide
    enough to span columns (banners, footers) are ignored; the remaining
    x-intervals are merged and the first gap wide enough and central enough
    (config.SOLUTION_GUTTER_*) is the gutter. Single-column pages merge into
    one interval and return None.
    """
    narrow = sorted(
        (r[0], r[2])
        for r in rects
        if r[2] - r[0] <= config.SOLUTION_COLUMN_MAX_SPAN_FRAC * page_width
    )
    if not narrow:
        return None
    merged = [narrow[0]]
    for x0, x1 in narrow[1:]:
        if x0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
        else:
            merged.append((x0, x1))
    lo, hi = (frac * page_width for frac in config.SOLUTION_GUTTER_BAND)
    for (_, a1), (b0, _) in zip(merged, merged[1:]):
        if b0 - a1 >= config.SOLUTION_GUTTER_MIN_FRAC * page_width and lo <= (a1 + b0) / 2 <= hi:
            return (a1 + b0) / 2
    return None


def _text_layer_markers(pdf_page, image, match, carry):
    """Problem-marker positions from a solution page's embedded text layer.

    The most reliable figure-assignment signal, available only for born-digital
    PDFs: each text block that *starts* with a problem marker and carries a real
    paragraph of text (config.SOLUTION_MARKER_MIN_CHARS -- which keeps
    marker-shaped furniture like answer-key cells "4. 12" out) is a problem
    start at an exact position. Returns ``(markers, gutter_x)`` where `markers`
    is [(column, y, problem)] in reading order (column-major, strictly
    increasing from `carry`); empty when the page has no text layer or no
    confident markers.
    """
    match = match or anchors_mod._match_marker
    scale = image.height / pdf_page.rect.height
    blocks = [
        (x0 * scale, y0 * scale, x1 * scale, y1 * scale, text)
        for x0, y0, x1, y1, text, _, block_type in pdf_page.get_text("blocks")
        if block_type == 0 and text.strip()
    ]
    gutter = _find_gutter(blocks, image.width)
    starts = []
    for x0, y0, x1, y1, text in blocks:
        text = text.lstrip()
        m = match(text)
        if m is None or len(text) - m[1] < config.SOLUTION_MARKER_MIN_CHARS:
            continue
        col = 0 if gutter is None or x0 < gutter else 1
        starts.append((col, y0, m[0]))
    starts.sort(key=lambda s: (s[0], s[1]))
    markers = []
    last = carry
    for col, y, number in starts:
        if last is None or number > last:
            markers.append((col, y, number))
            last = number
    return markers, gutter


def _solution_index_from_items(pic, items, page_height, carry_solutions, match_solution):
    """Best-effort solution index for a Picture from OCR reading-order spans."""
    if match_solution is None:
        return 1
    spans = []
    offset = 0
    current = dict(carry_solutions)
    for it in items:
        problem = it["problem"]
        if it["kind"] == "image":
            spans.append((offset, problem, current.get(problem, 1) if problem else 1))
            offset += max(len(it["text"]), 1)
        else:
            for line in it["text"].splitlines():
                if problem is not None and match_solution is not None:
                    sol = match_solution(line)
                    if sol is not None:
                        current[problem] = sol
                spans.append((offset, problem, current.get(problem, 1) if problem else 1))
                offset += max(len(line) + 1, 1)
    if not spans:
        return 1
    yc = (pic["box"][1] + pic["box"][3]) / 2
    target = (yc / page_height) * offset
    solution = 1
    for start, _, sol in spans:
        if start > target:
            break
        solution = sol
    return solution


def _assign_pics_by_markers(
    pics, markers, gutter, carry, items, page_height, carry_solutions, match_solution
):
    """Assign each Picture to the last text-layer marker at or above it.

    Positions compare column-major ((column, y), matching `markers`' reading
    order); a picture above the page's first marker belongs to `carry`.
    """
    assigned = []
    for pic in pics:
        xc = (pic["box"][0] + pic["box"][2]) / 2
        yc = (pic["box"][1] + pic["box"][3]) / 2
        col = 0 if gutter is None or xc < gutter else 1
        number = carry
        for mcol, my, prob in markers:
            if (mcol, my) <= (col, yc + config.Y_TOL):
                number = prob
            else:
                break
        solution = _solution_index_from_items(
            pic, items, page_height, carry_solutions, match_solution
        )
        assigned.append((pic, number, solution))
    return assigned


def _assign_solution_pics(
    pics, items, page_height, carry, carry_solutions=None, match_solution=None
):
    """Pair each DETR Picture on a solution page with a problem number.

    The OCR-only fallback used when the text layer gave no confident markers
    (see _text_layer_markers). `items` is the page's `parse_layout` output
    (already seeded with `carry`, the problem in progress at the top of the
    page); `pics` is top-to-bottom. Three tiers, most reliable first:
      1. the page has no problem-tagged content -> everything belongs to `carry`;
      2. nanonets' inline <img> count matches DETR's picture count -> zip them
         in reading order (each <img> item is already tagged with its problem);
      3. otherwise estimate by position: a picture's y-center fraction of the
         page is looked up in the items' cumulative char-offset spans, and it
         takes the problem of the item its fraction falls in. Rough (figures
         occupy height but few chars, and column layouts break it), but only
         used when everything better doesn't apply.
    Returns [(picture_det, problem_number | None, solution_index), ...].
    """
    carry_solutions = carry_solutions or {}
    spans = []  # (start_offset, problem, solution)
    offset = 0
    current_solutions = dict(carry_solutions)
    for it in items:
        problem = it["problem"]
        if it["kind"] == "image":
            spans.append((offset, problem, current_solutions.get(problem, 1) if problem else 1))
            offset += max(len(it["text"]), 1)
        else:
            for line in it["text"].splitlines():
                if problem is not None and match_solution is not None:
                    sol = match_solution(line)
                    if sol is not None:
                        current_solutions[problem] = sol
                spans.append((offset, problem, current_solutions.get(problem, 1) if problem else 1))
                offset += max(len(line) + 1, 1)

    numbers = {it["problem"] for it in items if it["problem"] is not None}
    if not numbers:
        solution = carry_solutions.get(carry, 1) if carry is not None else 1
        return [(p, carry, solution) for p in pics]

    img_items = [it for it in items if it["kind"] == "image"]
    if len(img_items) == len(pics):
        img_solutions = [
            _solution_index_from_items(p, items, page_height, carry_solutions, match_solution)
            for p in pics
        ]
        return [
            (p, it["problem"], sol)
            for p, it, sol in zip(pics, img_items, img_solutions)
        ]

    assigned = []
    for pic in pics:
        yc = (pic["box"][1] + pic["box"][3]) / 2
        target = (yc / page_height) * offset
        number = carry
        solution = carry_solutions.get(carry, 1) if carry is not None else 1
        for start, prob, sol in spans:
            if start > target:
                break
            if prob is not None:
                number = prob
                solution = sol
        assigned.append((pic, number, solution))
    return assigned



def process_solution_document(
    page_paths,
    client,
    threshold=None,
    match=None,
    cache=None,
    layout=None,
    clean_page=None,
    source_pdf=None,
    match_solution=None,
    figure_floor=None,
    figure_exclusion_regions=None,
    validate_page=None,
):
    """OCR a solution document and crop its figures, assigned to problems.

    Text segmentation stays with the series (`Series.parse_solutions`); this
    returns the raw material for it plus the figure crops the text pipeline
    would lose: ``(pages_md, figures)`` where `pages_md` is the *raw* markdown
    per page and `figures` maps problem number -> {solution index: [PIL crop,
    ...]} from DETR's Picture boxes (blank, nested, and -- per `layout` --
    page-spanning boxes dropped, exactly as on statement pages).

    Problem numbering must agree with what `parse_solutions` derives from the
    same text, so each page is tagged with the same `match` marker matcher and
    the same strictly-increasing guard, carried across pages. `clean_page` is
    the series' per-page markdown cleanup ((page_index, markdown) -> markdown),
    applied before tagging (e.g. Mandelbrot strips its out-of-order answer-key
    box); `pages_md` stays uncleaned so answer parsers still see everything.
    Figures above the first problem (cover art, logos) are dropped.

    `source_pdf` is the PDF the pages were rendered from, when there is one:
    a born-digital PDF's text layer gives exact marker positions (tier 0,
    `_text_layer_markers`), used whenever its problem set agrees with the OCR's
    for the page; otherwise assignment falls back to the OCR-only tiers
    (`_assign_solution_pics`).

    `figure_floor` is the series' `Series.solution_figure_floor` ((pdf_page,
    image) -> rendered y or None): a Picture whose vertical centre falls below
    the returned y is page furniture printed after the last worked solution (a
    back-cover credits box / colophon) and is dropped. It needs the PDF text
    layer, so it runs only when `source_pdf` is a PDF; None (any page without the
    marker) keeps every figure. This is the figure-side partner of `clean_page`'s
    back-cover text stripping.

    `figure_exclusion_regions` is the series' optional
    `Series.solution_figure_exclusion_regions` hook.  It returns narrowly
    identified rendered rectangles (usually anchored in the source PDF text
    layer); Pictures whose centres land in one are solution-page furniture.
    """
    layout = layout or config.LayoutOptions()
    thr = threshold if threshold is not None else config.NANONETS_DETECT_THRESHOLD
    temp = layout.nanonets_temperature
    doc = None
    if source_pdf is not None and Path(source_pdf).suffix.lower() == ".pdf":
        import pymupdf

        doc = pymupdf.open(source_pdf)
    pages_md = []
    figure_items = []
    carry = None  # problem in progress at the top of the next page
    carry_solutions = {}  # problem -> solution index in progress across pages
    for index, path in enumerate(page_paths):
        image = Image.open(path).convert("RGB")
        # Detection runs before OCR (it re-runs every page regardless) so its
        # figure boxes are available to mask a looping region on the final
        # runaway-recovery rung (see _ocr_page).
        solution_eq_filter = (
            layout.solution_equation_text_overlap
            and layout.equation_text_overlap is not None
        )
        scan_thr = min(thr, config.EQUATION_TEXT_MIN_SCORE) if solution_eq_filter else thr
        scanned = detect.detect(image, scan_thr)
        detections = [d for d in scanned if d["score"] >= thr]
        equation_text_boxes = (
            [
                d["box"]
                for d in scanned
                if d["label"] in config.TEXT_LABELS
                and d["score"] >= config.EQUATION_TEXT_MIN_SCORE
            ]
            if solution_eq_filter
            else None
        )
        markdown = _ocr_page(
            client,
            image,
            temp,
            cache=cache,
            cache_key=path,
            mask_boxes=_figure_mask_boxes(detections),
            validate=(
                (lambda text, i=index: validate_page(i, text))
                if validate_page is not None
                else None
            ),
            merge_incomplete=(
                (lambda whole, sliced: _merge_incomplete_problem_blocks(
                    whole, sliced, match or nanonets_mod._match_marker
                ))
                if validate_page is not None
                else None
            ),
        )
        pages_md.append(markdown)
        if clean_page is not None:
            raw_markdown = markdown
            markdown = clean_page(index, markdown)
            # A series cleanup that blanks a solution page has classified the
            # entire page as furniture/non-test material. Suppress its detected
            # figures too; otherwise they fall through under the carried problem
            # even though their corresponding text was deliberately removed.
            if raw_markdown.strip() and not markdown.strip():
                continue
        items = nanonets_mod.parse_layout(
            markdown,
            match,
            split_marker_table_rows=layout.split_marker_table_rows,
            start_problem=carry,
            # ordered_list_markers is a statement-page numbering fix only; a
            # solution document's own "N." markers drive figure assignment, so
            # it stays at the default here (avoid mis-splitting a solution's
            # genuine ordered list into spurious problems).
            point_value_list_markers=layout.point_value_list_markers,
            point_value_marker_consistency=layout.point_value_marker_consistency,
            heading_problem_markers=layout.heading_problem_markers,
            strict_section_restarts=layout.strict_section_restarts,
            consecutive_problem_markers=layout.consecutive_problem_markers,
            page_initial_point_restart=layout.page_initial_point_restart,
            flat_problem_numbering=layout.flat_problem_numbering,
            backreference_problem_markers=layout.backreference_problem_markers,
            split_glued_bare_markers=layout.split_glued_bare_markers,
        )
        # pdf_io names rendered pages "page_<pdf page number>.png".
        pdf_index = int(Path(path).stem.split("_")[1]) - 1 if doc is not None else None
        pics = _sorted_pictures(
            detections,
            image,
            layout.max_picture_area_frac,
            layout.header_picture_frac,
            layout.right_margin_picture_frac,
            layout.footer_picture_frac,
            layout.min_picture_height_frac,
            layout.equation_text_overlap if solution_eq_filter else None,
            equation_text_boxes,
            layout.equation_picture_min_aspect,
            layout.text_layer_equation_coverage,
            doc[pdf_index] if doc is not None else None,
        )
        if pics and doc is not None and layout.solution_answer_box_filter:
            pics = _drop_solution_answer_boxes(
                pics,
                doc[pdf_index],
                image,
                layout.solution_answer_box_max_width_frac,
            )
        if pics and doc is not None and figure_floor is not None:
            floor = figure_floor(doc[pdf_index], image)
            if floor is not None:
                pics = [p for p in pics if (p["box"][1] + p["box"][3]) / 2 < floor]
        if pics and doc is not None and figure_exclusion_regions is not None:
            regions = figure_exclusion_regions(doc[pdf_index], image)
            if regions:
                pics = [
                    p for p in pics
                    if not any(
                        x0 <= (p["box"][0] + p["box"][2]) / 2 <= x1
                        and y0 <= (p["box"][1] + p["box"][3]) / 2 <= y1
                        for x0, y0, x1, y1 in regions
                    )
                ]
        assigned = None
        if pics and doc is not None:
            markers, gutter = _text_layer_markers(doc[pdf_index], image, match, carry)
            new_numbers = {it["problem"] for it in items} - {None, carry}
            if markers and {m[2] for m in markers} == new_numbers:
                assigned = _assign_pics_by_markers(
                    pics,
                    markers,
                    gutter,
                    carry,
                    items,
                    image.height,
                    carry_solutions,
                    match_solution,
                )
        if assigned is None:
            assigned = _assign_solution_pics(
                pics, items, image.height, carry, carry_solutions, match_solution
            )
        for pic, number, solution in assigned:
            if number is not None:
                figure_items.append((number, solution, image.crop(tuple(pic["box"]))))
        if match_solution is not None:
            for it in items:
                number = it["problem"]
                if number is None or it["kind"] != "text":
                    continue
                solution = match_solution(it["text"])
                if solution is not None:
                    carry_solutions[number] = solution
        page_numbers = [it["problem"] for it in items if it["problem"] is not None]
        if page_numbers:
            page_max = max(page_numbers)
            carry = page_max if carry is None else max(carry, page_max)
    if doc is not None:
        doc.close()
    figures = {}
    for number, solution, crop in figure_items:
        figures.setdefault(number, {}).setdefault(solution, []).append(crop)
    return pages_md, figures


def _figure_ref(number, solution, k, path_prefix):
    """Markdown reference to a solution figure crop, as written by output.py."""
    name = f"problem_{number}_solution_{solution}_image_{k}.png"
    return f"![]({path_prefix}{name})"


def _place_figure_refs(text, refs):
    """Substitute figure references for the reading-order placeholders in `text`.

    `refs` is the authoritative list of crop references for this block (from
    DETR), in crop order. Each ``FIGURE_PLACEHOLDER`` left by the OCR marks where
    the model thought a figure sat; we replace them in order:
      * placeholders == refs -> exact 1:1 placement;
      * more placeholders than refs -> fill the first, drop the extras (the model
        over-emitted <img> tags on text-only spans);
      * fewer placeholders than refs -> fill what we have, append the rest at the
        end (the model missed some -- crop is kept, position is best-effort).
    With no refs, any stray placeholders are removed. No crop is ever dropped and
    no reference is ever left dangling.
    """
    segments = text.split(nanonets_mod.FIGURE_PLACEHOLDER)
    n_placeholders = len(segments) - 1
    if not refs:
        return _tidy("".join(segments))
    out = [segments[0]]
    for i in range(1, len(segments)):
        if i - 1 < len(refs):
            out.append(f"\n\n{refs[i - 1]}\n\n")
        out.append(segments[i])
    joined = "".join(out)
    if len(refs) > n_placeholders:
        trailing = "".join(f"\n\n{r}" for r in refs[n_placeholders:])
        joined = joined.rstrip() + trailing
    return _tidy(joined)


def _tidy(text):
    """Collapse the blank-line runs the ref insertions introduce; strip ends."""
    import re

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def inline_problem_figures(problems, path_prefix=""):
    """Reference each statement figure crop from its problem text, in place.

    The statement counterpart to `inline_solution_figures`. Joins the two halves
    of the "where does this figure go" signal: the DETR crops (the image elements
    on each `Problem`, in vertical order, which `write_problems` saves as
    ``problem_<n>_image_<k>.png``) and the reading-order `FIGURE_PLACEHOLDER`
    sentinels `process_image_markdown` left in the text when the series set
    `LayoutOptions.inline_figures`. Each `Problem`'s text elements are collapsed
    into one whose text carries a ``![](<path_prefix>problem_<n>_image_<k>.png)``
    ref (a path rooted at the output dir) for every figure; the image elements
    are kept unchanged and in order so the saved crop names still line up.

    Runs for every series, in both figure modes -- the point is that a problem's
    images are discoverable from problems.json, not only by globbing crop files:
      * inline_figures set   -> placeholders present, refs land at the model's
        <img> positions;
      * inline_figures unset -> no placeholders, so `_place_figure_refs` appends
        all refs at the end (position unknown, but the crop is still referenced).
    Count mismatches degrade exactly as on the solution side
    (`_place_figure_refs`): no crop is dropped, no ref dangles. Returns
    `problems` (mutated in place) for convenience.
    """
    for prob in problems:
        images = [el for el in prob.elements if el.kind == "image" and el.crop is not None]
        refs = [
            f"![]({path_prefix}problem_{prob.number}_image_{k}.png)"
            for k in range(1, len(images) + 1)
        ]
        statement = _place_figure_refs(prob.text, refs)
        prob.elements = (
            [ProblemElement("text", "Text", [], text=statement)] if statement else []
        ) + images
    return problems


def inline_solution_figures(solutions, figures, path_prefix=""):
    """Place each DETR solution-figure crop inline in its solution text.

    Joins the two halves of the "where does this figure go" signal: `figures`
    (from DETR -- the authoritative crops per problem and solution index, keyed
    {number: {solution_index: [crop, ...]}}) and the reading-order placeholders
    the OCR left in `solutions` (see nanonets.FIGURE_PLACEHOLDER). Returns a new
    solutions map of the same shape (a string per problem, or a list of solution
    strings) with each crop referenced as ``![](<path_prefix><crop name>)`` at
    its position. `path_prefix` is prepended to every crop filename (e.g.
    ``"usamts/2012_round1/"`` so paths resolve from the output root).

    A solution with no detected figures is returned unchanged except that stray
    placeholders are stripped. Figure groups whose solution index has no matching
    text block are appended to the problem's last block, so every crop that was
    written to disk is referenced exactly once.
    """
    result = {}
    for number, value in solutions.items():
        per_solution = figures.get(number, {})
        is_list = isinstance(value, (list, tuple))
        blocks = list(value) if is_list else [value]
        used = set()
        new_blocks = []
        for i, text in enumerate(blocks):
            solution = i + 1
            crops = per_solution.get(solution, [])
            refs = [_figure_ref(number, solution, k, path_prefix) for k in range(1, len(crops) + 1)]
            new_blocks.append(_place_figure_refs(text, refs))
            if crops:
                used.add(solution)
        leftover = [
            _figure_ref(number, solution, k, path_prefix)
            for solution, crops in sorted(per_solution.items())
            if solution not in used
            for k in range(1, len(crops) + 1)
        ]
        if leftover:
            tail = "".join(f"\n\n{r}" for r in leftover)
            if new_blocks:
                new_blocks[-1] = _tidy(new_blocks[-1].rstrip() + tail)
            else:
                new_blocks = [_tidy(tail)]
        result[number] = new_blocks if is_list else "\n".join(new_blocks)
    return result
