"""Central configuration. No more hardcoded paths scattered across modules."""

from pathlib import Path

# --- Models ---
LAYOUT_MODEL = "docling-project/docling-layout-heron"
OCR_MODEL = "mlx-community/gemma-4-E4B-it-qat-4bit"

# --- Detection ---
DETECT_THRESHOLD = 0.6

# --- Grouping geometry ---
# Vertical tolerance (px) when deciding which anchor a box sits under. Lets a
# statement that starts a hair above its problem number still bind correctly.
Y_TOL = 12

# A box is in the left "number column" if its left edge is within this fraction
# of the page width from the leftmost detection, and it is narrow.
LEFT_MARGIN_FRAC = 0.15
LEFT_COL_MAX_WIDTH_FRAC = 0.2

# Fallback grouping (no printed numbers): a vertical gap larger than this
# fraction of page height starts a new problem.
FALLBACK_GAP_FRAC = 0.06

# --- Blank / junk filtering ---
# Crops with a smaller dark-pixel ratio than this are treated as empty
# (answer blanks, score boxes) and dropped.
BLANK_INK_RATIO = 0.004
DARK_PIXEL_THRESHOLD = 200  # grayscale value below this counts as "ink"

# DETR classes that are page furniture, never problem content.
JUNK_LABELS = {"Page-header", "Page-footer", "Footnote"}

# DETR classes we treat as text to OCR (everything else is kept as an image crop).
TEXT_LABELS = {"Text", "Formula", "List-item", "Section-header", "Title", "Caption"}

# Picture-like classes kept as image crops.
IMAGE_LABELS = {"Picture", "Table"}

# --- Debug overlay ---
FONT_PATH = "/Users/cloud/Library/Fonts/FiraCodeNerdFont-Bold.ttf"
FONT_SIZE = 24

# --- IO ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "m0"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "m0" / "tmp"
