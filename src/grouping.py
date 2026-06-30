"""Assign detections to problems by geometry, and filter page furniture.

This module replaces the global-reasoning VLM call entirely. Grouping is just:
each box belongs to the most recent anchor above it.
"""

import re

from PIL import Image, ImageStat

from . import config

_FOOTER_RE = re.compile(
    r"copyright|©|all rights reserved|www\.|https?://"
    r"|^\s*page\s*\d+\s*$|^\s*score\s*:?\s*$",
    re.IGNORECASE,
)


def is_blank_crop(image: Image.Image, box) -> bool:
    """True if a crop is essentially empty (answer blank, score box)."""
    crop = image.crop(tuple(box)).convert("L")
    if crop.width == 0 or crop.height == 0:
        return True
    # Fraction of pixels darker than the ink threshold.
    hist = crop.histogram()
    dark = sum(hist[: config.DARK_PIXEL_THRESHOLD])
    total = crop.width * crop.height
    return (dark / total) < config.BLANK_INK_RATIO


def is_footer_text(text: str) -> bool:
    return bool(text) and bool(_FOOTER_RE.search(text))


def _vertical_center(box):
    return (box[1] + box[3]) / 2


def classify(d, image):
    """Decide what a detection is, or that it should be dropped.

    Returns "image", "text", or None. Uses a strict whitelist: only known text
    and image classes survive. Container classes (Form, Key-Value Region,
    Document Index, …) and junk are dropped — these are what produce giant
    page-spanning crops that wreck OCR.
    """
    label = d["label"]
    if label in config.JUNK_LABELS:
        return None

    box = d["box"]
    if label in config.IMAGE_LABELS:
        return None if is_blank_crop(image, box) else "image"

    if label in config.TEXT_LABELS:
        text = (d.get("text") or "").strip()
        if is_footer_text(text):
            return None
        # Standalone anchor markers (bare "26" / "Problem 19" header) carry no
        # statement text; keep them out of content.
        if (
            d.get("is_anchor")
            and not d.get("marker_inline", False)
            and not (d.get("text_clean") or "").strip()
        ):
            return None
        if is_blank_crop(image, box):
            return None
        return "text"

    return None  # Form, Code, Checkbox, Key-Value Region, Document Index, …


def group_by_anchors(detections, anchors, image):
    """Return {problem_number: [detections]} using anchor segmentation + rules.

    Boxes above the first anchor are dropped (page header/title). Junk labels,
    footers, and blank crops are dropped.
    """
    groups = {a["problem"]: [] for a in anchors}
    first_anchor_y = anchors[0]["y"]

    for d in detections:
        yc = _vertical_center(d["box"])

        # Above the first problem -> header / title region.
        if yc + config.Y_TOL < first_anchor_y:
            continue

        if classify(d, image) is None:
            continue

        owner = None
        for a in anchors:
            if a["y"] <= yc + config.Y_TOL:
                owner = a["problem"]
            else:
                break
        if owner is not None:
            groups[owner].append(d)

    return groups


def fallback_group_by_gaps(detections, image, page_height):
    """No printed numbers: split into problems on large vertical gaps."""
    keep = [d for d in detections if classify(d, image) is not None]
    keep.sort(key=lambda d: d["box"][1])
    gap = config.FALLBACK_GAP_FRAC * page_height
    groups = {}
    problem = 1
    last_bottom = None
    for d in keep:
        if last_bottom is not None and d["box"][1] - last_bottom >= gap:
            problem += 1
        groups.setdefault(problem, []).append(d)
        last_bottom = d["box"][3]
    return groups
