"""Orchestration: page image -> structured problems.

Two engines:
  * mlx       -- detect -> OCR each text box -> find anchors -> group -> assemble.
                 No global VLM reasoning; all segmentation is deterministic geometry.
  * nanonets  -- one whole-page OCR pass returns problem-segmented markdown with
                 inline <img> tags; DETR supplies only the image crops, mapped to
                 problems by reading-order ordinal (see process_image_nanonets).
"""

import sys
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


def _assemble(groups, image, ocr: OCRModel):
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
                    latex = anchors_mod.strip_leading_marker(latex)
                elements.append(ProblemElement("text", d["label"], box, text=latex))
        problems.append(Problem(number=number, elements=elements))
    return problems


def process_image(image_path, ocr: OCRModel, threshold=config.DETECT_THRESHOLD):
    """Run the full pipeline on one page. Returns (problems, detections, groups)."""
    image = Image.open(image_path).convert("RGB")
    detections = detect.detect(image, threshold)
    if not detections:
        return [], [], {}

    _ocr_text_boxes(detections, image, ocr)
    anchors = anchors_mod.detect_anchors(detections, image.width)

    if anchors:
        groups = grouping.group_by_anchors(detections, anchors, image)
    else:
        groups = grouping.fallback_group_by_gaps(detections, image, image.height)

    problems = _assemble(groups, image, ocr)
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


def process_image_nanonets(image_path, client, threshold=config.NANONETS_DETECT_THRESHOLD):
    """Whole-page OCR via the nanonets engine; DETR supplies the image crops.

    Returns (problems, detections, groups) to match process_image. `groups` maps
    each problem to the Picture detections assigned to it (for the debug overlay).
    """
    image = Image.open(image_path).convert("RGB")
    detections = detect.detect(image, threshold)
    markdown = client.parse_page(image)
    items = nanonets_mod.parse_layout(markdown)

    # Nanonets reports the header/content split itself: any <img> before problem 1
    # (problem is None) is a page logo/banner. Drop that many Picture crops from the
    # top so both sides start at the first content figure and the ordinal zip lines up.
    pictures = _sorted_pictures(detections, image)
    n_header = sum(1 for it in items if it["kind"] == "image" and it["problem"] is None)
    pictures = pictures[n_header:]
    img_items = [it for it in items if it["kind"] == "image" and it["problem"] is not None]

    # Reading-order ordinal mapping: the i-th in-body <img> tag <-> the i-th DETR
    # Picture from the top. Both sides have the header dropped, so they align.
    for i, it in enumerate(img_items):
        it["_pic"] = pictures[i] if i < len(pictures) else None
    leftover = pictures[len(img_items):]
    if len(pictures) != len(img_items):
        print(
            f"[nanonets] image-count mismatch: {len(img_items)} <img> tag(s) vs "
            f"{len(pictures)} DETR picture(s)",
            file=sys.stderr,
        )

    problems = {}  # number -> Problem, insertion-ordered (numbers increase)
    groups = {}

    def problem_for(number):
        return problems.setdefault(number, Problem(number=number))

    for it in items:
        number = it["problem"]
        if number is None:  # page header (title/logo before problem 1)
            continue
        prob = problem_for(number)
        if it["kind"] == "text":
            lines = [ln for ln in it["text"].splitlines() if not grouping.is_footer_text(ln)]
            text = "\n".join(lines).strip()
            if text:
                prob.elements.append(ProblemElement("text", "Text", [], text=text))
        else:
            pic = it.get("_pic")
            if pic is not None:
                box = pic["box"]
                prob.elements.append(
                    ProblemElement("image", "Picture", box, crop=image.crop(tuple(box)))
                )
                groups.setdefault(number, []).append(pic)
            else:
                # Nanonets saw a figure DETR did not detect: record it, no crop.
                prob.elements.append(ProblemElement("image", "Picture", [], text=it["text"]))

    # DETR found more figures than Nanonets tagged: attach the extras to the last
    # problem so no detected image is silently dropped.
    if leftover and problems:
        last = problems[max(problems)]
        for pic in leftover:
            box = pic["box"]
            last.elements.append(
                ProblemElement("image", "Picture", box, crop=image.crop(tuple(box)))
            )
            groups.setdefault(last.number, []).append(pic)

    return [problems[n] for n in sorted(problems)], detections, groups
