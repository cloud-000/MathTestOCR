"""Nanonets engine: whole-page OCR via a local OpenAI-compatible endpoint.

Unlike the mlx engine, the model sees the whole page once and returns structured
markdown: LaTeX for math, HTML for tables, and an ``<img>...</img>`` tag at each
figure's location in reading order. We never ask it for coordinates -- the DETR
layout model still supplies the actual image crops (see pipeline.py). This module
only talks to the endpoint and turns its markdown into an ordered list of items
tagged with the problem each belongs to.
"""

import base64
import io
import re

from PIL import Image

from . import config
from .anchors import _match_marker

# <img ...>desc</img>  or self-closing <img .../>
_IMG_RE = re.compile(r"<img\b[^>]*>(.*?)</img>|<img\b[^>]*/?>", re.IGNORECASE | re.DOTALL)
# Structural HTML that wraps layout/data tables -- flattened to line breaks.
_TABLE_TAG_RE = re.compile(r"</?(?:table|thead|tbody|tr|td|th)\b[^>]*>", re.IGNORECASE)
# Page furniture the prompt asks the model to tag.
_FURNITURE_RE = re.compile(
    r"<(?:watermark|page_number)\b[^>]*>.*?</(?:watermark|page_number)>",
    re.IGNORECASE | re.DOTALL,
)
_POINTS_ONLY_RE = re.compile(r"^\(\s*\d+\s*\)$")  # "(1)" point-value cells
_BLANK_RUN_RE = re.compile(r"_{2,}")  # answer blanks: "________"


class NanonetsClient:
    """Wrapper around the OpenAI-compatible Nanonets-OCR endpoint."""

    def __init__(self, base_url: str = config.NANONETS_BASE_URL, model=config.NANONETS_MODEL):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key="not-needed")
        self._model = model

    @property
    def model(self) -> str:
        """Resolved model id; auto-detected from /v1/models when not configured."""
        if self._model is None:
            self._model = self._client.models.list().data[0].id
        return self._model

    def parse_page(self, image: Image.Image) -> str:
        """Return the raw markdown transcription of a whole page image."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": config.NANONETS_PROMPT},
                    ],
                }
            ],
        )
        return resp.choices[0].message.content or ""


def _clean_text_line(line: str) -> str:
    return _BLANK_RUN_RE.sub("", line).strip()


def parse_layout(markdown: str):
    """Turn Nanonets markdown into an ordered list of items.

    Each item is a dict ``{"kind": "text"|"image", "problem": int|None, "text": str}``
    in reading order. ``problem`` is the number of the most recent problem marker
    seen above the item (``None`` for page-header content before problem 1).
    ``text`` is the statement text (markers stripped) or the image description.
    """
    markdown = _FURNITURE_RE.sub("", markdown)

    # Split into alternating text / image tokens, preserving order.
    tokens = []  # ("text", str) | ("image", description)
    pos = 0
    for m in _IMG_RE.finditer(markdown):
        if m.start() > pos:
            tokens.append(("text", markdown[pos : m.start()]))
        tokens.append(("image", (m.group(1) or "").strip()))
        pos = m.end()
    if pos < len(markdown):
        tokens.append(("text", markdown[pos:]))

    items = []
    buf: list[str] = []
    current = None  # current problem number
    last = None  # highest problem number accepted so far

    def flush():
        if not buf:
            return
        text = "\n".join(buf).strip()
        buf.clear()
        if text:
            items.append({"kind": "text", "problem": current, "text": text})

    for kind, payload in tokens:
        if kind == "image":
            flush()
            items.append({"kind": "image", "problem": current, "text": payload})
            continue

        text = _TABLE_TAG_RE.sub("\n", payload)
        for raw in text.splitlines():
            line = _clean_text_line(raw)
            if not line or _POINTS_ONLY_RE.match(line):
                continue
            match = _match_marker(line)
            if match is not None and (last is None or match[0] > last):
                # New problem starts here; flush the previous one first.
                flush()
                current = last = match[0]
                line = line[match[1] :].strip()  # drop the printed marker
                if not line:
                    continue
            buf.append(line)
    flush()
    return items
