"""Central configuration. No more hardcoded paths scattered across modules."""

from dataclasses import dataclass
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

# Sampling temperature for the whole-page OCR. 0.0 is greedy/deterministic and
# the right default for faithful transcription. A series may nudge this up via
# its LayoutOptions (see `LayoutOptions.nanonets_temperature`) when greedy
# decoding degenerates on its pages -- e.g. Mandelbrot grids where the model
# gets stuck repeating a <table> row. Keep it as low as still works: higher
# temperatures trade transcription fidelity for loop-breaking.
NANONETS_TEMPERATURE = 0.0

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

# A DETR "Picture" covering more than this fraction of the page area is almost
# certainly a whole-page layout misclassification, not a real figure (seen on
# dense MATHCOUNTS text pages: a low-confidence box spanning nearly the entire
# page). This is *not* applied globally -- USAMTS problems can have legitimate
# page-filling diagrams -- so it is the value MATHCOUNTS opts into via its
# LayoutOptions (see MathcountsSeries.layout_options); other series keep every
# Picture (max_picture_area_frac=None).
NANONETS_MAX_PICTURE_AREA_FRAC = 0.5

# When DETR detects both a figure group and its individual panels, a Picture
# whose area is at least this fraction covered by a larger Picture is a nested
# duplicate and dropped, keeping only the enclosing crop (e.g. Purple Comet
# problem 29's "four patterns" strip: one wide box plus one box per panel).
# Applied for every series -- a figure and its own sub-panels are never both
# wanted as separate crops.
NESTED_PICTURE_FRAC = 0.9

# --- Solution-figure assignment (pipeline.process_solution_document) ---
# Tier 0 reads problem-marker positions from the solution PDF's embedded text
# layer. A text block only counts as a problem start when at least this much
# text follows its marker -- a real paragraph ("1. It is possible to fit...").
# This keeps short marker-shaped furniture out: answer-key cells ("4. 12") and
# bare number headers, whose out-of-order numbers would poison the assignment.
SOLUTION_MARKER_MIN_CHARS = 40
# Two-column solution sheets (Mandelbrot's landscape pages) are detected by a
# vertical gutter no text block crosses: at least this fraction of the page
# wide, with its center inside the middle band below. Blocks wider than
# SOLUTION_COLUMN_MAX_SPAN_FRAC (banners, footers) are ignored by the search.
SOLUTION_GUTTER_MIN_FRAC = 0.03
SOLUTION_GUTTER_BAND = (0.3, 0.7)
SOLUTION_COLUMN_MAX_SPAN_FRAC = 0.6


@dataclass(frozen=True)
class LayoutOptions:
    """Series-scoped tuning for the nanonets layout / figure heuristics.

    The defaults here are the conservative, series-agnostic behavior. MATHCOUNTS
    pages (dense answer-blank tables, faint inset figures, and a recurring
    whole-page false-positive Picture box) opt into the extra heuristics via
    ``MathcountsSeries.layout_options``. Threaded from the CLI through
    ``process_test`` / ``process_image_nanonets`` so a series' quirks stay in the
    series, not baked into the shared pipeline (mirrors ``match_marker`` /
    ``skip_page``).
    """

    # Drop any DETR Picture covering more than this fraction of the page area
    # (a whole-page layout misclassification). None -> keep every Picture.
    max_picture_area_frac: float | None = None
    # When DETR's left-margin problem-start count disagrees with nanonets'
    # problem count, fall back to splitting page content at the largest vertical
    # gaps (see pipeline._gap_based_starts). Off by default: on pages with
    # unusual margins it can scramble otherwise-correct figure assignment.
    gap_based_picture_fallback: bool = False
    # Split a <table> block row-by-row, rewriting each row whose leading cell is
    # a problem marker into plain statement text (MATHCOUNTS packs many problems
    # into one answer-blank table). Off by default: other series' tables are
    # real tabular data, kept verbatim as HTML.
    split_marker_table_rows: bool = False
    # Take each problem's start position from its left-margin heading box alone
    # (see config.HEADER_LABELS), ignoring the statement/body text below it. On
    # by default a problem whose number sits on its own heading line (e.g. Purple
    # Comet's "Problem N") counts twice in the left-margin scan -- once for the
    # heading, once for the statement -- doubling the start count and drifting
    # figure assignment. Off by default: series whose numbers are inline with the
    # statement have no separate heading and rely on the body text as the start.
    problem_start_from_headers: bool = False
    # Sampling temperature for the whole-page OCR pass. Defaults to the greedy
    # NANONETS_TEMPERATURE (0.0); a series raises it slightly only when greedy
    # decoding loops on its pages (e.g. Mandelbrot grids that make the model
    # repeat a <table> row). Higher values trade transcription fidelity for
    # loop-breaking, so keep it as low as still works.
    nanonets_temperature: float = NANONETS_TEMPERATURE

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

# DETR classes that are a problem's heading line (its number), used as the
# problem-start signal when LayoutOptions.problem_start_from_headers is set.
HEADER_LABELS = {"Section-header", "Title"}

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
