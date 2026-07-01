"""Orchestration: page image -> structured problems.

Two engines:
  * mlx       -- detect -> OCR each text box -> find anchors -> group -> assemble.
                 No global VLM reasoning; all segmentation is deterministic geometry.
  * nanonets  -- one whole-page OCR pass returns problem-segmented markdown with
                 inline <img> tags; DETR supplies only the image crops, mapped to
                 problems by reading-order ordinal (see process_image_nanonets).
"""

from dataclasses import dataclass, field

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
    pics.sort(key=lambda d: (d["box"][1], d["box"][0]))
    return pics


def _box_area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def _problem_start_ys(detections, image):
    """Vertical positions where problems start, from DETR's left-margin text boxes.

    A content text box whose left edge sits at the page's left text margin begins
    a problem (its statement/number). Centered headers and indented continuation
    lines start further right and are excluded; boxes on the same row are merged.
    Returns the start y_top values sorted top-to-bottom.
    """
    cand = [
        d
        for d in detections
        if d["label"] in config.TEXT_LABELS and not grouping.is_blank_crop(image, d["box"])
    ]
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
    starts = _problem_start_ys(detections, image)
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


def ocr_pages_markdown(page_paths, client, cache=None, layout=None):
    """Whole-page OCR of every page to markdown, concatenated in reading order.

    Unlike `process_test`, this runs no layout detection and does no problem
    segmentation -- it just returns the raw Nanonets markdown for the whole
    document. Used by the series solution parser (`Series.parse_solutions`),
    which segments the concatenated text itself. `client` is a NanonetsClient.
    `cache` is an optional OCRCache serving / storing each page's markdown.
    `layout` is the series' LayoutOptions (only its OCR temperature is used here).
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
    return "\n\n".join(parts)
