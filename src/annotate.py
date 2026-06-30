"""Debug-only overlay: draw detected boxes + indices for visual inspection.

The pipeline no longer depends on this — it exists purely so you can eyeball
detections and grouping while tuning thresholds.
"""

from math import ceil

from PIL import Image, ImageDraw, ImageFont

from . import config

_PALETTE = [
    "red", "blue", "green", "purple", "orange",
    "brown", "magenta", "teal", "olive", "navy",
]


def _font():
    try:
        return ImageFont.truetype(config.FONT_PATH, size=config.FONT_SIZE)
    except OSError:
        return ImageFont.load_default()


def draw_detections(image: Image.Image, detections, groups=None) -> Image.Image:
    """Overlay boxes. If `groups` is given, color each box by its problem."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font()

    box_to_problem = {}
    if groups:
        for problem, dets in groups.items():
            for d in dets:
                box_to_problem[id(d)] = problem

    for i, d in enumerate(detections):
        x0, y0, x1, y1 = d["box"]
        if id(d) in box_to_problem:
            color = _PALETTE[box_to_problem[id(d)] % len(_PALETTE)]
            label = f"{i}:P{box_to_problem[id(d)]}"
        else:
            color = "gray"
            label = str(i)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=ceil(2))
        draw.text((x0 + 2, y0 + 2), label, fill=color, font=font)

    return Image.alpha_composite(base, overlay)
