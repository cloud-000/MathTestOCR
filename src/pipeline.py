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


def _sorted_pictures(detections, image, max_area_frac=None):
    """Non-blank DETR Picture detections, top-to-bottom (reading order).

    `max_area_frac` (from a series' LayoutOptions) optionally drops any Picture
    covering more than that fraction of the page -- a whole-page layout
    misclassification. None (the default) keeps every non-blank Picture.
    """
    pics = [
        d
        for d in detections
        if d["label"] == "Picture" and not grouping.is_blank_crop(image, d["box"])
    ]
    if max_area_frac is not None:
        max_area = max_area_frac * image.width * image.height
        pics = [d for d in pics if _box_area(d["box"]) <= max_area]
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
    """
    labels = config.HEADER_LABELS if headers_only else config.TEXT_LABELS
    cand = [
        d
        for d in detections
        if d["label"] in labels and not grouping.is_blank_crop(image, d["box"])
    ]
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


def _assign_pictures(detections, image, problem_seq, layout):
    """Map each non-blank DETR Picture to a problem number by vertical position.

    `problem_seq` is the ordered list of problem numbers (top-to-bottom). Each
    picture is assigned to the problem whose start is the lowest one at or above
    the picture's vertical centre; pictures above the first problem (page logos)
    are dropped. Returns {problem_number: [picture_det, ...]}.

    `layout` is the series' LayoutOptions: it controls the page-spanning Picture
    filter (`max_picture_area_frac`) and whether the gap-based problem-start
    fallback is used when DETR's left-margin count disagrees with nanonets'.
    """
    pics = _sorted_pictures(detections, image, layout.max_picture_area_frac)
    if not pics or not problem_seq:
        return {}
    starts = _problem_start_ys(detections, image, layout.problem_start_from_headers)
    groups = {}
    if not starts:
        # No usable text geometry: keep every figure, on the first problem.
        groups[problem_seq[0]] = list(pics)
        return groups
    if len(starts) != len(problem_seq) and layout.gap_based_picture_fallback:
        starts = _gap_based_starts(detections, image, len(problem_seq)) or starts
    if len(starts) != len(problem_seq):
        print(
            f"[nanonets] problem-count mismatch: {len(problem_seq)} problem(s) from "
            f"text vs {len(starts)} from DETR layout; figure assignment may drift"
        )
    for pic in pics:
        yc = (pic["box"][1] + pic["box"][3]) / 2
        if yc + config.Y_TOL < starts[0]:
            continue  # vertically above the first problem -> page header / logo
        idx = 0
        for i, sy in enumerate(starts):
            if sy <= yc + config.Y_TOL:
                idx = i
            else:
                break
        idx = min(idx, len(problem_seq) - 1)
        groups.setdefault(problem_seq[idx], []).append(pic)
    return groups


def process_image_nanonets(
    image_path,
    client,
    threshold=config.NANONETS_DETECT_THRESHOLD,
    match_marker=None,
    cache=None,
    layout=None,
):
    """Whole-page OCR via the nanonets engine; DETR supplies the image crops.

    Nanonets does all text transcription and problem segmentation. Figures are
    mapped to problems geometrically from DETR (see _assign_pictures) -- Nanonets'
    inline <img> tags are ignored, since the model both invents them on text-only
    problems and omits them on real figures. Returns (problems, detections,
    groups); `groups` maps each problem to its Picture detections (debug overlay).

    `match_marker` is an optional series-specific marker matcher for
    competition-specific numbering quirks. `cache` is an optional OCRCache: when
    given, the (slow) whole-page OCR is served from / written to it, while DETR
    detection still runs every time. `layout` is the series' LayoutOptions (its
    table/figure heuristic knobs); None uses the conservative defaults.
    """
    layout = layout or config.LayoutOptions()
    print("[nanonets] Starting pipeline...")
    image = Image.open(image_path).convert("RGB")
    print("[nanonets] Running layout detection (DETR)...")
    detections = detect.detect(image, threshold)
    print("[nanonets] Layout detection done.")
    print("[nanonets] Running whole-page OCR (Nanonets)...")
    temp = layout.nanonets_temperature
    if cache is not None:
        markdown = cache.page_markdown(image_path, lambda: client.parse_page(image, temp))
    else:
        markdown = client.parse_page(image, temp)
    print("[nanonets] Nanonets OCR done.")
    items = nanonets_mod.parse_layout(
        markdown, match_marker, split_marker_table_rows=layout.split_marker_table_rows
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
            continue  # <img> descriptions are unreliable; figures come from DETR
        prob = problem_for(number)
        lines = [ln for ln in it["text"].splitlines() if not grouping.is_footer_text(ln)]
        text = "\n".join(lines).strip()
        if text:
            prob.elements.append(ProblemElement("text", "Text", [], text=text))

    # Geometric figure assignment from DETR.
    groups = _assign_pictures(detections, image, problem_seq, layout)
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
    through the chosen engine independently, then problems sharing a number are
    merged (their elements concatenated in page order). `engine` is "nanonets" or
    "mlx"; `model` is the matching NanonetsClient / OCRModel. `match` is the
    series-specific marker matcher. `cache` is an optional OCRCache for the
    nanonets whole-page OCR (ignored by the mlx engine, which OCRs per crop).
    `layout` is the series' LayoutOptions (nanonets only). Returns the merged
    [Problem], number-sorted.
    """
    merged = {}  # number -> Problem
    for path in page_paths:
        if engine == "nanonets":
            thr = threshold if threshold is not None else config.NANONETS_DETECT_THRESHOLD
            problems, _, _ = process_image_nanonets(
                path, model, thr, match_marker=match, cache=cache, layout=layout
            )
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
    temp = layout.nanonets_temperature
    parts = []
    for path in page_paths:
        if cache is not None:
            markdown = cache.page_markdown(
                path, lambda p=path: client.parse_page(Image.open(p).convert("RGB"), temp)
            )
        else:
            markdown = client.parse_page(Image.open(path).convert("RGB"), temp)
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


def _assign_pics_by_markers(pics, markers, gutter, carry):
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
        assigned.append((pic, number))
    return assigned


def _assign_solution_pics(pics, items, page_height, carry):
    """Pair each DETR Picture on a solution page with a problem number.

    The OCR-only fallback used when the text layer gave no confident markers
    (see _text_layer_markers). `items` is the page's `parse_layout` output
    (already seeded with `carry`, the problem in progress at the top of the
    page); `pics` is top-to-bottom. Three tiers, most reliable first:
      1. the page starts no new problem -> everything belongs to `carry`;
      2. nanonets' inline <img> count matches DETR's picture count -> zip them
         in reading order (each <img> item is already tagged with its problem);
      3. otherwise estimate by position: a picture's y-center fraction of the
         page is looked up in the items' cumulative char-offset spans, and it
         takes the problem of the item its fraction falls in. Rough (figures
         occupy height but few chars, and column layouts break it), but only
         used when everything better doesn't apply.
    Returns [(picture_det, problem_number | None), ...].
    """
    numbers = {it["problem"] for it in items if it["problem"] is not None}
    if not numbers or numbers == {carry}:
        return [(p, carry) for p in pics]
    img_items = [it for it in items if it["kind"] == "image"]
    if len(img_items) == len(pics):
        return [(p, it["problem"]) for p, it in zip(pics, img_items)]
    spans = []  # (start_offset, problem) per item, in reading order
    offset = 0
    for it in items:
        spans.append((offset, it["problem"]))
        offset += max(len(it["text"]), 1)
    assigned = []
    for pic in pics:
        yc = (pic["box"][1] + pic["box"][3]) / 2
        target = (yc / page_height) * offset
        number = carry
        for start, prob in spans:
            if start > target:
                break
            if prob is not None:
                number = prob
        assigned.append((pic, number))
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
):
    """OCR a solution document and crop its figures, assigned to problems.

    Text segmentation stays with the series (`Series.parse_solutions`); this
    returns the raw material for it plus the figure crops the text pipeline
    would lose: ``(pages_md, figures)`` where `pages_md` is the *raw* markdown
    per page and `figures` maps problem number -> [PIL crop, ...] from DETR's
    Picture boxes (blank, nested, and -- per `layout` -- page-spanning boxes
    dropped, exactly as on statement pages).

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
    for index, path in enumerate(page_paths):
        image = Image.open(path).convert("RGB")
        if cache is not None:
            markdown = cache.page_markdown(path, lambda: client.parse_page(image, temp))
        else:
            markdown = client.parse_page(image, temp)
        pages_md.append(markdown)
        if clean_page is not None:
            markdown = clean_page(index, markdown)
        items = nanonets_mod.parse_layout(
            markdown,
            match,
            split_marker_table_rows=layout.split_marker_table_rows,
            start_problem=carry,
        )
        detections = detect.detect(image, thr)
        pics = _sorted_pictures(detections, image, layout.max_picture_area_frac)
        assigned = None
        if pics and doc is not None:
            # pdf_io names rendered pages "page_<pdf page number>.png".
            pdf_index = int(Path(path).stem.split("_")[1]) - 1
            markers, gutter = _text_layer_markers(doc[pdf_index], image, match, carry)
            new_numbers = {it["problem"] for it in items} - {None, carry}
            if markers and {m[2] for m in markers} == new_numbers:
                assigned = _assign_pics_by_markers(pics, markers, gutter, carry)
        if assigned is None:
            assigned = _assign_solution_pics(pics, items, image.height, carry)
        for pic, number in assigned:
            if number is not None:
                figure_items.append((number, image.crop(tuple(pic["box"]))))
        page_numbers = [it["problem"] for it in items if it["problem"] is not None]
        if page_numbers:
            carry = max(page_numbers)
    if doc is not None:
        doc.close()
    drop = max(layout.drop_trailing_solution_figures, 0)
    if drop:
        figure_items = figure_items[:-drop] if drop < len(figure_items) else []
    figures = {}
    for number, crop in figure_items:
        figures.setdefault(number, []).append(crop)
    return pages_md, figures
