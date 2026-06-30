"""DETR layout detection. Model loads lazily so importing the package is cheap."""

import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from . import config

CLASSES_MAP = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
    11: "Document Index",
    12: "Code",
    13: "Checkbox-Selected",
    14: "Checkbox-Unselected",
    15: "Form",
    16: "Key-Value Region",
}

_processor = None
_model = None


def _ensure_loaded():
    global _processor, _model
    if _model is None:
        _processor = RTDetrImageProcessor.from_pretrained(config.LAYOUT_MODEL)
        _model = RTDetrV2ForObjectDetection.from_pretrained(config.LAYOUT_MODEL)
    return _processor, _model


def detect(image: Image.Image, threshold: float = config.DETECT_THRESHOLD):
    """Detect document layout elements.

    Returns a list of dicts: {"label", "score", "box": [x0, y0, x1, y1]}.
    """
    processor, model = _ensure_loaded()
    image = image.convert("RGB")

    inputs = processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image.size[::-1]]),  # pyright: ignore[reportArgumentType]
        threshold=threshold,
    )

    detections = []
    for result in results:
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            detections.append(
                {
                    "label": CLASSES_MAP[label_id.item()],
                    "score": round(score.item(), 2),
                    "box": [round(x, 2) for x in box.tolist()],
                }
            )
    return detections
