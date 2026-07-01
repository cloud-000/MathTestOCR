"""Central configuration. No more hardcoded paths scattered across modules."""

from pathlib import Path

# --- Models ---
LAYOUT_MODEL = "docling-project/docling-layout-heron"
OCR_MODEL = "mlx-community/gemma-4-E4B-it-qat-4bit"  # MLX engine (legacy path)

# --- Engines ---
# "nanonets": whole-page OCR via the local OpenAI-compatible endpoint (default).
# "mlx": the per-crop Gemma OCR + anchor/grouping pipeline.
DEFAULT_ENGINE = "nanonets"

# --- Nanonets engine (OpenAI-compatible endpoint) ---
NANONETS_BASE_URL = "http://127.0.0.1:8080/v1"
NANONETS_MODEL = None  # None -> auto-detect via GET /v1/models (first id)
# The detection threshold the nanonets engine uses for the DETR crops it pulls.
# Lower than the mlx default: figures are faint and 0.6 drops some (e.g. a
# grid diagram), while 0.5 catches them without admitting page-spanning junk.
NANONETS_DETECT_THRESHOLD = 0.5

# Standard Nanonets-OCR prompt. The <img></img> tags it emits at each figure's
# location (in reading order) are what lets us map DETR crops to problems, so do
# not drop the image-description instruction.
NANONETS_PROMPT = (
    "Extract the text from the above document as if you were reading it "
    "naturally. Return the tables in html format. Return the equations in LaTeX "
    "representation. If there is an image in the document, output an empty "
    "<img> tag; do not describe the image or add captions. "
    "Watermarks should "
    "be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page "
    "numbers should be wrapped in brackets. Ex: <page_number>14</page_number> "
    "or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for "
    "check boxes."
# "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the  tag; otherwise, add the image caption inside . Watermarks should be wrapped in brackets. Ex: OFFICIAL COPY. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."
)

# Hard cap on tokens generated per page. A backstop against runaway loops
# (the model can degenerate into an infinite description on a dense figure);
# generous enough for a fully packed competition page of real content.
NANONETS_MAX_TOKENS = 16384
# Runaway-loop guard. While streaming, if the last NANONETS_REPEAT_PROBE chars
# of the tail recur NANONETS_REPEAT_COUNT times within the last
# NANONETS_REPEAT_WINDOW chars, the model is stuck in a verbatim loop -- abort
# the stream. Probes made only of layout-filler chars (underscores, dots) are
# ignored so long answer-blanks / leader lines don't trip the guard.
NANONETS_REPEAT_WINDOW = 1200
NANONETS_REPEAT_PROBE = 48
NANONETS_REPEAT_COUNT = 4

# Picture->problem mapping is geometric: each non-blank DETR Picture is assigned
# to the problem whose statement it vertically sits in. Nanonets' inline <img>
# tags are NOT trusted for this (the model both hallucinates them on text-only
# problems and omits them on real figures). Problem bands come from DETR's
# left-margin text boxes: a content box whose left edge is within this fraction
# of the page width of the leftmost content box starts a problem. Tight enough
# to exclude centered headers ("Each problem is worth 5 points.") and footers.
NANONETS_START_X_TOL_FRAC = 0.02

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

# --- Series (batch parsing of whole competitions) ---
# Default output root; per-series results go to <DEFAULT_OUT_DIR>/<series>/<test>/.
DEFAULT_OUT_DIR = "out"

# Optional default --data-dir per series, so the source path can be omitted on
# the CLI. Leave a series out (or None) to require --data-dir explicitly. These
# are external paths specific to the user's machine; override on the CLI anytime.
_MATHTESTS_ROOT = Path("/Users/cloud/MathTests")
SERIES_DATA_DIRS = {
    "usamts": _MATHTESTS_ROOT / "USAMTS" / "out",
    "purplecomet": _MATHTESTS_ROOT / "PurpleComet" / "out",
    "mandelbrot": _MATHTESTS_ROOT / "Mandelbrot" / "out",
    "mathcounts": _MATHTESTS_ROOT / "Mathcounts" / "out",
}

# A Mathcounts <year>/<level> folder mixes problem rounds with answer/solution
# PDFs. Only these round stems are parseable tests; everything else (answers,
# solutions, stray year booklets) is skipped by discovery.
MATHCOUNTS_TEST_ROUNDS = {
    "sprint", "target", "team", "countdown", "cdr",
    "warmups", "workouts", "masters",
}
