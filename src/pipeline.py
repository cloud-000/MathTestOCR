"""Orchestration: page image -> structured problems.

Two engines:
  * mlx       -- detect -> OCR each text box -> find anchors -> group -> assemble.
                 No global VLM reasoning; all segmentation is deterministic geometry.
  * nanonets  -- one whole-page OCR pass returns problem-segmented markdown with
                 inline <img> tags; DETR supplies only the image crops, mapped to
                 problems by reading-order ordinal (see process_image_nanonets).
"""

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from . import anchors as anchors_mod
from . import config, detect, grouping
from . import nanonets as nanonets_mod
from .ocr import OCRModel


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
                     equation_text_overlap=None, equation_text_boxes=None):
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
            if h <= 0 or w / h <= config.EQUATION_PICTURE_MIN_ASPECT:
                return False
            return any(_contained_frac(box, t) > equation_text_overlap for t in text_boxes)

        pics = [d for d in pics if not _is_equation(d["box"])]
    pics = _drop_nested_pictures(pics)
    pics.sort(key=lambda d: (d["box"][1], d["box"][0]))
    return pics


def _box_area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


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
                     equation_text_boxes=None):
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

    `items` is the page's parse_layout output. When DETR's problem-start count
    disagrees with nanonets' problem count (so the geometry is untrustworthy),
    the problem-tagged inline <img> positions are used as a fallback -- but only
    when their count matches DETR's pictures and none is appended past the last
    problem's text (see below).
    """
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
    # Geometry is the primary signal, but only trustworthy when DETR found exactly
    # one start per problem: a figure's vertical position then pins it to the
    # problem whose text sits directly above. When the counts disagree the starts
    # have drifted (a statement split into several left-margin boxes, or a missed
    # one), so before falling back to fuzzy geometry try nanonets' inline <img>
    # tags (zip pics top-to-bottom to the tags in reading order). This runs only
    # on a start/problem mismatch -- when the starts agree geometry is preferred,
    # because nanonets' tag order is the less reliable of the two.
    if len(starts) != len(problem_seq):
        # Trust the tags only when their problem-tagged count matches DETR's
        # pictures *and* none is "appended" past its problem: nanonets sometimes
        # dumps a figure's <img> at the very bottom of the page (after the last
        # problem) instead of beside it, mis-tagging it -- e.g. an octagon diagram
        # belonging to problem 4 emitted after problem 5. The tell is that no
        # later, higher-numbered problem's text follows the tag (trailing footer
        # text inherits the last problem's number, so a plain position check is
        # not enough); when that happens we distrust the tags and keep geometry.
        indexed = list(enumerate(items or []))
        img_tags = [
            (i, it["problem"])
            for i, it in indexed
            if it["kind"] == "image" and it["problem"] is not None
        ]

        def _appended(idx, prob):
            return not any(
                it["kind"] == "text" and it["problem"] is not None and it["problem"] > prob
                for j, it in indexed
                if j > idx
            )

        appended = any(_appended(i, p) for i, p in img_tags)
        if img_tags and len(img_tags) == len(pics) and not appended:
            groups = {}
            for pic, (_, number) in zip(pics, img_tags):
                groups.setdefault(number, []).append(pic)
            return groups
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
        return groups
    if len(starts) != len(problem_seq):
        print(
            f"[nanonets] problem-count mismatch: {len(problem_seq)} problem(s) from "
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
    return groups


def _ocr_page(client, image, base_temp, cache=None, cache_key=None, mask_boxes=None):
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
    if cache is not None and cache_key is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit
    temps = [base_temp] + [t for t in config.NANONETS_RETRY_TEMPS if t > base_temp]
    for i, temp in enumerate(temps):
        if i:
            print(f"[nanonets] runaway; retrying at temperature {temp}")
        markdown, runaway = client.parse_page(image, temp)
        if not runaway:
            break
    else:
        if mask_boxes:
            print("[nanonets] runaway persists; masking figures and re-OCRing")
            markdown, runaway = client.parse_page(
                image, temps[-1], mask_boxes=mask_boxes
            )
    if cache is not None and cache_key is not None and not runaway:
        cache.put(cache_key, markdown)
    return markdown


def _figure_mask_boxes(detections):
    """Figure rectangles (DETR Picture/Table) to blank on the masking rung."""
    return [d["box"] for d in detections if d["label"] in config.IMAGE_LABELS]


def process_image_nanonets(
    image_path,
    client,
    threshold=config.NANONETS_DETECT_THRESHOLD,
    match_marker=None,
    cache=None,
    layout=None,
    carry=None,
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
    """
    layout = layout or config.LayoutOptions()
    print("[nanonets] Starting pipeline...")
    image = Image.open(image_path).convert("RGB")
    print("[nanonets] Running layout detection (DETR)...")
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
    print("[nanonets] Layout detection done.")
    print("[nanonets] Running whole-page OCR (Nanonets)...")
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
    )
    print("[nanonets] Nanonets OCR done.")
    items = nanonets_mod.parse_layout(
        markdown,
        match_marker,
        split_marker_table_rows=layout.split_marker_table_rows,
        start_problem=carry,
        ordered_list_markers=layout.ordered_list_markers,
    )

    print("[nanonets] Assembling problems...")
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
        assign_detections, image, geom_seq, layout, items, carry, equation_text_boxes
    )
    for number in sorted(groups):
        prob = problem_for(number)
        for pic in groups[number]:
            box = pic["box"]
            prob.elements.append(
                ProblemElement("image", "Picture", box, crop=image.crop(tuple(box)))
            )

    print("[nanonets] Problem assembly done.")
    return [problems[n] for n in sorted(problems)], detections, groups


def process_test(page_paths, engine, model, threshold=None, match=None, cache=None, layout=None):
    """Parse every page of a multi-page test and merge problems by number.

    A test (e.g. a USAMTS PDF rendered to page PNGs) may span several pages, with
    a single problem's statement and figures split across them. Each page is run
    through the chosen engine, then problems sharing a number are merged (their
    elements concatenated in page order). `engine` is "nanonets" or "mlx"; `model`
    is the matching NanonetsClient / OCRModel. `match` is the series-specific
    marker matcher. `cache` is an optional OCRCache for the nanonets whole-page OCR
    (ignored by the mlx engine, which OCRs per crop). `layout` is the series'
    LayoutOptions (nanonets only). Returns the merged [Problem], number-sorted.

    For the nanonets engine the highest problem number seen so far is carried into
    the next page (`carry`) so that page's leading text and any figure sitting
    above its first problem attach to the problem continued from the previous page
    instead of being dropped (see process_image_nanonets / _assign_pictures).
    """
    merged = {}  # number -> Problem
    carry = None  # nanonets: problem in progress at the top of the next page
    for path in page_paths:
        if engine == "nanonets":
            thr = threshold if threshold is not None else config.NANONETS_DETECT_THRESHOLD
            problems, _, _ = process_image_nanonets(
                path, model, thr, match_marker=match, cache=cache, layout=layout, carry=carry
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
        if problem is not None and it["kind"] == "text":
            solution = match_solution(it["text"])
            if solution is not None:
                current[problem] = solution
        spans.append((offset, problem, current.get(problem, 1) if problem else 1))
        offset += max(len(it["text"]), 1)
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
    item_solutions = []
    current_solutions = dict(carry_solutions)
    for it in items:
        problem = it["problem"]
        if problem is not None and it["kind"] == "text" and match_solution is not None:
            solution = match_solution(it["text"])
            if solution is not None:
                current_solutions[problem] = solution
        item_solutions.append(current_solutions.get(problem, 1) if problem else 1)

    numbers = {it["problem"] for it in items if it["problem"] is not None}
    if not numbers:
        solution = carry_solutions.get(carry, 1) if carry is not None else 1
        return [(p, carry, solution) for p in pics]
    img_items = [it for it in items if it["kind"] == "image"]
    if len(img_items) == len(pics):
        img_solutions = [
            sol for it, sol in zip(items, item_solutions) if it["kind"] == "image"
        ]
        return [
            (p, it["problem"], sol)
            for p, it, sol in zip(pics, img_items, img_solutions)
        ]
    spans = []  # (start_offset, problem, solution) per item, in reading order
    offset = 0
    for it, solution in zip(items, item_solutions):
        spans.append((offset, it["problem"], solution))
        offset += max(len(it["text"]), 1)
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
        detections = detect.detect(image, thr)
        markdown = _ocr_page(
            client,
            image,
            temp,
            cache=cache,
            cache_key=path,
            mask_boxes=_figure_mask_boxes(detections),
        )
        pages_md.append(markdown)
        if clean_page is not None:
            markdown = clean_page(index, markdown)
        items = nanonets_mod.parse_layout(
            markdown,
            match,
            split_marker_table_rows=layout.split_marker_table_rows,
            start_problem=carry,
            # ordered_list_markers is a statement-page numbering fix only; a
            # solution document's own "N." markers drive figure assignment, so
            # it stays at the default here (avoid mis-splitting a solution's
            # genuine ordered list into spurious problems).
        )
        pics = _sorted_pictures(
            detections, image, layout.max_picture_area_frac, layout.header_picture_frac
        )
        assigned = None
        if pics and doc is not None:
            # pdf_io names rendered pages "page_<pdf page number>.png".
            pdf_index = int(Path(path).stem.split("_")[1]) - 1
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
            carry = max(page_numbers)
    if doc is not None:
        doc.close()
    drop = max(layout.drop_trailing_solution_figures, 0)
    if drop:
        figure_items = figure_items[:-drop] if drop < len(figure_items) else []
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
    sentinels `process_image_nanonets` left in the text when the series set
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
