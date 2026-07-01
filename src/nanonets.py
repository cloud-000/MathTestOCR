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
_IMG_RE = re.compile(
    r"<img\b[^>]*>(.*?)</img>|<img\b[^>]*/?>", re.IGNORECASE | re.DOTALL
)
# Structural HTML that wraps layout/data tables -- flattened to line breaks.
_TABLE_TAG_RE = re.compile(r"</?(?:table|thead|tbody|tr|td|th)\b[^>]*>", re.IGNORECASE)
# Page furniture the prompt asks the model to tag.
_FURNITURE_RE = re.compile(
    r"<(?:watermark|page_number)\b[^>]*>.*?</(?:watermark|page_number)>",
    re.IGNORECASE | re.DOTALL,
)
_POINTS_ONLY_RE = re.compile(r"^\(\s*\d+\s*\)$")  # "(1)" point-value cells
_BLANK_RUN_RE = re.compile(r"_{2,}")  # answer blanks: "________"
# MATHCOUNTS answer line: after the problem number comes a blank to fill plus an
# optional unit ("26. ____ cm In the figure..."). Strip the blank and any
# lowercase unit words so they don't lead the statement (which starts uppercase).
_ANSWER_BLANK_RE = re.compile(r"^_{2,}\s*(?:[a-z]+\s+)*")

# Chars that legitimately repeat in layout (answer blanks, dotted leaders, rules).
# A tail made only of these is filler, not a generation loop.
_FILLER_CHARS = set("_ .-—–·•\t\n")


def _is_runaway(text: str) -> bool:
    """True if the tail of `text` is a verbatim loop (degenerate generation).

    The Nanonets model can get stuck endlessly re-describing a figure
    ("The numbers 24, A-1 ... are written in the dodecagons." x N). We catch it
    by checking whether the final probe-sized slice recurs several times within
    the recent window. Filler-only tails (rows of underscores/dots) are ignored.
    """
    tail = text[-config.NANONETS_REPEAT_WINDOW :]
    probe = tail[-config.NANONETS_REPEAT_PROBE :]
    if len(probe) < config.NANONETS_REPEAT_PROBE or set(probe) <= _FILLER_CHARS:
        return False
    return tail.count(probe) >= config.NANONETS_REPEAT_COUNT


def _close_dangling_img(text: str) -> str:
    """Repair an ``<img>`` left unterminated by a truncated/aborted stream.

    The image description is never used (DETR supplies crops by reading-order
    ordinal), so we drop the partial text and leave a clean empty
    ``<img></img>`` -- the positional marker survives and still maps to its crop.
    """
    idx = text.rfind("<img")
    if idx == -1:
        return text
    tail = text[idx:]
    if "</img>" in tail.lower() or re.match(r"<img\b[^>]*/>", tail, re.IGNORECASE):
        return text  # properly closed or self-closing
    return text[:idx] + "<img></img>"


class NanonetsClient:
    """Wrapper around the OpenAI-compatible Nanonets-OCR endpoint."""

    def __init__(
        self, base_url: str = config.NANONETS_BASE_URL, model=config.NANONETS_MODEL
    ):
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
            temperature=0,  # greedy: deterministic, faithful transcription
            max_tokens=config.NANONETS_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": config.NANONETS_PROMPT},
                    ],
                }
            ],
            stream=True,
        )
        print("[nanonets] Streaming response:")
        full_content = []
        total = 0  # running char count, to throttle the runaway check
        for chunk in resp:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_content.append(content)
                total += len(content)
                # Cheap to check; only meaningful once we have a window's worth.
                if total >= config.NANONETS_REPEAT_WINDOW and _is_runaway(
                    "".join(full_content)
                ):
                    print("\n[nanonets] runaway repetition detected; aborting stream")
                    resp.close()
                    break
        print("\n[nanonets] Images processed / streaming complete")
        return _close_dangling_img("".join(full_content))


def _clean_text_line(line: str) -> str:
    return _BLANK_RUN_RE.sub("", line).strip()


def parse_layout(markdown: str, match_marker=None):
    """Turn Nanonets markdown into an ordered list of items.

    Each item is a dict ``{"kind": "text"|"image", "problem": int|None, "text": str}``
    in reading order. ``problem`` is the number of the most recent problem marker
    seen above the item (``None`` for page-header content before problem 1).
    ``text`` is the statement text (markers stripped) or the image description.

    `match_marker` is an optional series-specific marker matcher (see
    anchors._match_marker); it defaults to the built-in pattern set.
    """
    match_marker = match_marker or _match_marker
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
            line = raw.strip()
            if not line:
                continue
            # Match the marker before blank runs are scrubbed, so an answer line's
            # "N. ____ unit" can be told apart from a real statement.
            match = match_marker(line)
            if match is not None and (last is None or match[0] > last):
                # New problem starts here; flush the previous one first.
                flush()
                current = last = match[0]
                line = _ANSWER_BLANK_RE.sub("", line[match[1] :].lstrip(), count=1)
            line = _clean_text_line(line)
            if not line or _POINTS_ONLY_RE.match(line):
                continue
            buf.append(line)
    flush()
    return items
