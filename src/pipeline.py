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


def _sorted_pictures(detections, image):
    """Non-blank DETR Picture detections, sorted top-to-bottom (reading order)."""
    pics = [
        d
        for d in detections
        if d["label"] == "Picture" and not grouping.is_blank_crop(image, d["box"])
    ]
    pics.sort(key=lambda d: (d["box"][1], d["box"][0]))
    return pics


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


def _assign_pictures(detections, image, problem_seq):
    """Map each non-blank DETR Picture to a problem number by vertical position.

    `problem_seq` is the ordered list of problem numbers (top-to-bottom). Each
    picture is assigned to the problem whose start is the lowest one at or above
    the picture's vertical centre; pictures above the first problem (page logos)
    are dropped. Returns {problem_number: [picture_det, ...]}.
    """
    pics = _sorted_pictures(detections, image)
    if not pics or not problem_seq:
        return {}
    starts = _problem_start_ys(detections, image)
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
    image_path, client, threshold=config.NANONETS_DETECT_THRESHOLD, match_marker=None
):
    """Whole-page OCR via the nanonets engine; DETR supplies the image crops.

    Nanonets does all text transcription and problem segmentation. Figures are
    mapped to problems geometrically from DETR (see _assign_pictures) -- Nanonets'
    inline <img> tags are ignored, since the model both invents them on text-only
    problems and omits them on real figures. Returns (problems, detections,
    groups); `groups` maps each problem to its Picture detections (debug overlay).

    `match_marker` is an optional series-specific marker matcher for
    competition-specific numbering quirks.
    """
    print("[nanonets] Starting pipeline...")
    image = Image.open(image_path).convert("RGB")
    print("[nanonets] Running layout detection (DETR)...")
    detections = detect.detect(image, threshold)
    print("[nanonets] Layout detection done.")
    print("[nanonets] Running whole-page OCR (Nanonets)...")
    markdown = client.parse_page(image)
    print("[nanonets] Nanonets OCR done.")
    items = nanonets_mod.parse_layout(markdown, match_marker)

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
    groups = _assign_pictures(detections, image, problem_seq)
    for number in sorted(groups):
        prob = problem_for(number)
        for pic in groups[number]:
            box = pic["box"]
            prob.elements.append(
                ProblemElement("image", "Picture", box, crop=image.crop(tuple(box)))
            )

    print("[nanonets] Problem assembly done.")
    return [problems[n] for n in sorted(problems)], detections, groups


def process_test(page_paths, engine, model, threshold=None, match=None):
    """Parse every page of a multi-page test and merge problems by number.

    A test (e.g. a USAMTS PDF rendered to page PNGs) may span several pages, with
    a single problem's statement and figures split across them. Each page is run
    through the chosen engine independently, then problems sharing a number are
    merged (their elements concatenated in page order). `engine` is "nanonets" or
    "mlx"; `model` is the matching NanonetsClient / OCRModel. `match` is the
    series-specific marker matcher. Returns the merged [Problem], number-sorted.
    """
    merged = {}  # number -> Problem
    for path in page_paths:
        if engine == "nanonets":
            thr = threshold if threshold is not None else config.NANONETS_DETECT_THRESHOLD
            problems, _, _ = process_image_nanonets(path, model, thr, match_marker=match)
        else:
            thr = threshold if threshold is not None else config.DETECT_THRESHOLD
            problems, _, _ = process_image(path, model, thr, match=match)
        for p in problems:
            if p.number not in merged:
                merged[p.number] = Problem(number=p.number)
            merged[p.number].elements.extend(p.elements)
    return [merged[n] for n in sorted(merged)]
