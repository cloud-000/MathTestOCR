"""Find problem-start anchors from OCR'd detections.

A problem almost always prints its own number ("Problem 19", "26.", "1/3/37.").
We detect those markers deterministically instead of asking a VLM to reason
about the page. Each detection that carries a marker becomes an anchor.
"""

import re

from . import config

# Ordered by priority. Each returns the problem number from group(1)/last group.
_PATTERNS = [
    re.compile(r"^\s*(?:Problem|Question|Prob\.?)\s+(\d+)", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*/\s*(\d+)\s*\."),  # USAMTS "1/3/37."
    re.compile(r"^\s*(\d+)\s*[.)]"),  # "26." or "26)"
]

# How much non-marker text must remain for the marker to be "inline" (part of
# the statement) rather than a standalone header/number box.
_INLINE_MIN_REMAINDER = 12


def strip_leading_marker(text: str) -> str:
    """Remove a leading problem marker ("1.", "Problem 19", "1/3/37.") from OCR
    output so it doesn't leak into the statement text."""
    result = _match_marker(text)
    if result is None:
        return text
    _, end = result
    return text[end:].lstrip()


def _match_marker(text: str):
    """Return (problem_number, match_end) for the first matching pattern, or None."""
    for pat in _PATTERNS:
        m = pat.match(text)
        if m:
            return int(m.group(m.lastindex)), m.end()
    return None


def detect_anchors(detections, page_width):
    """Annotate detections in place and return the list of anchors.

    Each detection may gain:
      - "is_anchor": bool
      - "problem": int             (when anchor)
      - "marker_inline": bool      (marker shares the box with statement text)
      - "text_clean": str          (text with a standalone marker stripped)

    An anchor dict is {"problem", "y", "det_index"}.
    """
    if not detections:
        return []

    left_edge = min(d["box"][0] for d in detections)
    left_col_cut = left_edge + config.LEFT_MARGIN_FRAC * page_width
    narrow_cut = config.LEFT_COL_MAX_WIDTH_FRAC * page_width

    anchors = []
    for idx, d in enumerate(detections):
        text = (d.get("text") or "").strip()
        d["is_anchor"] = False
        d["text_clean"] = text

        result = _match_marker(text)
        problem = None

        if result is not None:
            problem, end = result
            remainder = text[end:].strip()
            inline = len(remainder) >= _INLINE_MIN_REMAINDER
            d["marker_inline"] = inline
            d["text_clean"] = remainder if inline else ""
        else:
            # Secondary: a narrow box in the left margin whose text is just a number.
            x0, _, x1, _ = d["box"]
            width = x1 - x0
            if (
                x0 <= left_col_cut
                and width <= narrow_cut
                and re.fullmatch(r"\d+", text)
            ):
                problem = int(text)
                d["marker_inline"] = False
                d["text_clean"] = ""

        if problem is not None:
            d["is_anchor"] = True
            d["problem"] = problem
            d["_full_text"] = text  # kept in case we later reject this anchor
            anchors.append({"problem": problem, "y": d["box"][1], "det_index": idx})

    anchors.sort(key=lambda a: a["y"])
    return _enforce_increasing(anchors, detections)


def _enforce_increasing(anchors, detections):
    """Drop false anchors so problem numbers strictly increase down the page.

    A list item like "1." inside problem 19 matches the "N." pattern but breaks
    the increasing sequence, so it is demoted back to ordinary content.
    """
    kept = []
    last = None
    for a in anchors:
        if last is None or a["problem"] > last:
            kept.append(a)
            last = a["problem"]
        else:
            d = detections[a["det_index"]]
            d["is_anchor"] = False
            d["text_clean"] = d.pop("_full_text", d.get("text_clean", ""))
            d.pop("problem", None)
    for a in kept:
        detections[a["det_index"]].pop("_full_text", None)
    return kept
