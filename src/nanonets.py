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

# Sentinel marking "a figure sits here in reading order", used only while
# assembling solution text (see pipeline.inline_solution_figures). The model's
# inline <img> position is unreliable for *which* problem a figure belongs to
# (DETR decides that geometrically), but it is a useful hint for *where within*
# an already-assigned solution's text the figure goes. We normalize every <img>
# tag to this single char so the placement step can count and replace them
# against DETR's authoritative crop count. U+FFFC (object replacement char)
# never occurs in real transcription.
FIGURE_PLACEHOLDER = "\ufffc"


def normalize_img_placeholders(text: str) -> str:
    """Replace every ``<img>`` tag (open/close or self-closing) with the sentinel.

    Used on solution text so a later step can align the model's reading-order
    figure positions with DETR's actual crops (see FIGURE_PLACEHOLDER).
    """
    return _IMG_RE.sub(FIGURE_PLACEHOLDER, text)
# A whole <table>...</table> block. Kept verbatim so tabular data survives as
# HTML in the output (both problem statements and solutions). Lazy so adjacent
# tables don't merge; DOTALL so a table spanning several lines is one match.
_TABLE_BLOCK_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
# A single row, and its cells, within a table block -- used to look for a
# problem marker in each row's leading cell (see consume_table in parse_layout).
_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
# Structural table tags. Only applied to text *outside* a recognized table block
# (a stray or unclosed tag) -- there it is flattened to a line break so raw
# markup doesn't leak into the statement.
_TABLE_TAG_RE = re.compile(r"</?(?:table|thead|tbody|tr|td|th)\b[^>]*>", re.IGNORECASE)
# Page furniture the prompt asks the model to tag.
_FURNITURE_RE = re.compile(
    r"<(?:watermark|page_number)\b[^>]*>.*?</(?:watermark|page_number)>",
    re.IGNORECASE | re.DOTALL,
)
# Point-value cells the model emits beside each problem: a parenthesized number
# ("(1)"), or -- in Mandelbrot's two-column problem tables -- a single circled
# digit ("①"..) or ballot box ("☐"). None is problem content; dropped as
# furniture wherever a line/cell is nothing but one of these.
_POINTS_ONLY_RE = re.compile(
    r"^\(\s*\d+\s*\)$"
    r"|^[①-⑳❶-❿➀-➉⓪☐-☒■□◻◼]+$"
)
_BLANK_RUN_RE = re.compile(r"_{2,}")  # answer blanks: "________"
# A display-math close (``$$`` or ``\]``) and any trailing spaces. The next
# problem's marker is sometimes emitted on the same line, right after the prior
# problem's closing display equation ("...\boxed{5}.$$ **Problem 14** ..."); the
# line-anchored marker matcher then misses it and swallows the whole next
# problem. See _split_glued_markers.
_MATH_CLOSE_RE = re.compile(r"(?:\$\$|\\\])[ \t]*")
# Ordered-list scaffolding (see parse_layout's ordered_list_markers): a line
# that *opens* a list item starts the next problem; all list/line-break tags are
# flattened to spaces so none leak into the statement.
_LI_OPEN_RE = re.compile(r"<li\b", re.IGNORECASE)
_LIST_TAG_RE = re.compile(r"</?(?:ol|ul|li)\b[^>]*>|<br\s*/?>", re.IGNORECASE)
# The unit that labels an answer blank, once the blank rule itself is gone:
# a leading run of currency signs / short lowercase-or-number tokens up to
# where the real statement starts (a capital letter, an opening paren, or the
# end of the fragment). Catches "$", "cm$^2$", "units$^2$", "units 2", "base 8".
# The capital/paren guard keeps it from eating into a real statement, which in
# these tests always begins with a capitalized word.
_ANSWER_UNIT_RE = re.compile(
    r"^(?:\$|[a-z][\w°²³.^${}/-]*|\d+)(?:\s+(?:\$|[a-z][\w°²³.^${}/-]*|\d+))*"
    r"\s*(?=[A-Z(]|$)"
)

# Chars that legitimately repeat in layout (answer blanks, dotted leaders, rules).
# A tail made only of these is filler, not a generation loop.
_FILLER_CHARS = set("_ .-—–·•\t\n")

# Collapse an opening tag's attributes to just its name: `<span style="...">` ->
# `<span>`, `<td colspan="2">` -> `<td>`. Used only inside the runaway guard so
# a long identical style attribute emitted once per answer-key cell can't fill
# the repeat probe by itself (see _is_runaway). Tag *names* survive, so a real
# tag loop (`<td><td><td>...`) is still caught.
_TAG_ATTR_RE = re.compile(r"<([a-zA-Z][\w:-]*)\b[^>]*?(/?)>")


def _is_runaway(text: str) -> bool:
    """True if the tail of `text` is a verbatim loop (degenerate generation).

    The Nanonets model can get stuck endlessly re-describing a figure
    ("The numbers 24, A-1 ... are written in the dodecagons." x N). We catch it
    by checking whether the final probe-sized slice recurs several times in a
    tight cluster near the end of the recent window. Filler-only tails (rows of
    underscores/dots/dashes) are ignored unless the unbroken filler run grows
    past `config.NANONETS_FILLER_MAX_RUN` -- a real rule/leader spans one row, a
    loop stuck on filler ("- - - - -") does not.

    Tag *attributes* are collapsed first (`<span style="...">` -> `<span>`) so a
    long identical style string emitted once per cell -- e.g. an answer key's
    ``<span style="border: 1px solid black; padding: 2px;">`` (53 chars) around
    every answer, dozens of them a few chars apart -- can't fill the 48-char
    probe by itself and masquerade as a loop; the varying cell numbers/answers
    between the tags then break the repeat. Tag *names* are kept, so a genuine
    tag loop (`<td><td><td>...`) and any repeated content still trip the guard.
    We normalize a slice wider than the window (stripping shrinks it) so the
    post-normalization tail is full-length.
    """
    text = _TAG_ATTR_RE.sub(r"<\1\2>", text[-2 * config.NANONETS_REPEAT_WINDOW :])
    tail = text[-config.NANONETS_REPEAT_WINDOW :]
    probe = tail[-config.NANONETS_REPEAT_PROBE :]
    if len(probe) < config.NANONETS_REPEAT_PROBE:
        return False
    if set(probe) <= _FILLER_CHARS:
        # Filler-only tail: normally a legitimate rule/leader/answer-blank, so
        # ignored -- but those span one row. A generation loop stuck on filler
        # ("- - - - -", "____...") emits an unbounded run, so an over-long
        # unbroken trailing filler run is itself a runaway. (Filler chars are
        # untouched by the tag-attr collapse above, so measuring on `tail` is
        # exact.)
        run = 0
        for ch in reversed(tail):
            if ch in _FILLER_CHARS:
                run += 1
            else:
                break
        return run >= config.NANONETS_FILLER_MAX_RUN

    positions = []
    start = 0
    while True:
        idx = tail.find(probe, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    if len(positions) >= config.NANONETS_REPEAT_COUNT:
        recent = positions[-config.NANONETS_REPEAT_COUNT :]
        if all(
            b - a <= config.NANONETS_REPEAT_MAX_GAP
            for a, b in zip(recent, recent[1:])
        ):
            return True

    # The probe cluster above only spans loops whose period fits in
    # NANONETS_REPEAT_MAX_GAP. A model stuck re-describing a figure loops on a
    # far longer unit (a whole paragraph, ~300+ chars) that recurs too few times,
    # too far apart, to register there. Fall through to a tandem-repeat check.
    return _tandem_loop(text)


def _tandem_loop(text: str) -> bool:
    """True if the tail is a long block repeated near-verbatim back-to-back.

    Targets the runaway the probe-cluster guard misses: the model re-emitting the
    same figure-description paragraph dozens of times (period ~300-400 chars).
    We anchor a probe-sized slice just *before* the tail tip -- these loops drift
    slightly at the very end, so the last chars are the least reliable -- find
    its previous occurrence to read off the period, then confirm that several
    consecutive periods across the window match. The min-period floor and repeat
    count keep legitimate bounded repetition (a summation's similar terms, a
    short table) from tripping it; see the config constants for the reasoning.
    """
    probe = config.NANONETS_REPEAT_PROBE
    w = text[-config.NANONETS_LOOP_WINDOW :]
    n = len(w)
    if n < config.NANONETS_LOOP_MIN_PERIOD * config.NANONETS_LOOP_MIN_REPEATS:
        return False
    anchor_end = n - probe
    anchor = w[anchor_end - probe : anchor_end]
    if len(anchor) < probe:
        return False
    prev = w.rfind(anchor, 0, anchor_end - 1)
    if prev == -1:
        return False
    period = (anchor_end - probe) - prev
    if not (
        config.NANONETS_LOOP_MIN_PERIOD <= period <= config.NANONETS_LOOP_MAX_PERIOD
    ):
        return False
    reps = n // period
    if reps < config.NANONETS_LOOP_MIN_REPEATS:
        return False
    matches = 0
    for k in range(reps - 1):
        b1 = w[n - period * (k + 1) : n - period * k]
        b2 = w[n - period * (k + 2) : n - period * (k + 1)]
        if len(b1) == period == len(b2):
            same = sum(1 for x, y in zip(b1, b2) if x == y) / period
            if same >= config.NANONETS_LOOP_MATCH:
                matches += 1
    return matches >= config.NANONETS_LOOP_MIN_REPEATS - 1


def _split_glued_markers(text: str, match_marker) -> str:
    """Break a line before a problem marker glued onto a display-math close.

    Nanonets occasionally emits the next problem's marker on the same line as
    the previous problem's closing display equation
    ("...\\boxed{5}.$$ **Problem 14** The three roots..."). `consume_lines`'s
    matcher only anchors at line start, so such a marker is invisible and the
    whole following problem is swallowed. We insert a newline before it so the
    existing per-line logic sees it normally.

    Splitting is gated two ways so it can't misfire:

    * on a *display-math close* (``$$`` or ``\\]``) -- a real new problem never
      opens mid-sentence, but a display equation routinely ends the problem
      before it (an in-prose "as shown in Problem 3" is thus never touched);
    * only on a *worded* marker ("Problem"/"Question"), i.e. one whose text
      begins with a letter. A ``$$`` is ambiguous (it also *opens* display math),
      and a bare-number or ``1/3/37.``-style marker right after one is almost
      always a numeric math literal ("``$$469234692346...4685.$$``"), which the
      permissive matcher would otherwise read as a spurious problem start. The
      word "Problem" never appears inside a formula, so a worded marker is safe.

    Whether the freed marker actually starts a new problem (vs. repeats an
    already-seen number) is still decided by consume_lines -- this only makes it
    visible.
    """
    def repl(m):
        probe = text[m.end():].lstrip("*_# ")
        if probe[:1].isalpha() and match_marker(probe) is not None:
            return m.group(0).rstrip() + "\n"
        return m.group(0)

    return _MATH_CLOSE_RE.sub(repl, text)


def _close_dangling_img(text: str) -> str:
    """Repair an ``<img>`` the model never closed with ``</img>``.

    Two different situations look the same (an opening ``<img`` with no
    matching close): a genuinely aborted/truncated stream, where nothing
    follows it and the rest of the page is lost; or the model simply
    forgetting the closing tag while continuing to transcribe the rest of the
    page perfectly fine (observed on MATHCOUNTS tables: it wrote a full image
    description straight into a `<td>` and never closed `<img>` before
    `</td>`). Truncating unconditionally at the dangling tag -- the old
    behavior -- silently dropped every problem after it in the second case.
    We only truncate when nothing follows; otherwise we splice in `</img>`
    right after the opening tag (dropping the never-used description, same as
    the truncation case) and keep going.
    """
    idx = text.rfind("<img")
    if idx == -1:
        return text
    tail = text[idx:]
    if "</img>" in tail.lower() or re.match(r"<img\b[^>]*/>", tail, re.IGNORECASE):
        return text  # properly closed or self-closing
    m = re.match(r"<img\b[^>]*>", tail, re.IGNORECASE)
    if not m:
        return text[:idx] + "<img></img>"  # opening tag itself got cut off
    content_start = idx + m.end()
    # The (always-discarded) description runs until real page content resumes:
    # the next tag, or a blank-line paragraph break -- whichever comes first.
    # Everything from there on is kept. Anchoring only on the next tag drops the
    # rest of the page when the following problem is plain text/LaTeX with no
    # markup (e.g. the model wrote "<img>\n\n**Problem 30** ...$\sqrt{m}-n$").
    rest = text[content_start:]
    tag_at = rest.find("<")
    para = re.search(r"\n[ \t]*\n", rest)
    para_at = para.start() if para else -1
    cuts = [p for p in (tag_at, para_at) if p != -1]
    if not cuts:
        return text[:idx] + "<img></img>"  # nothing follows -- genuine truncation
    return text[:content_start] + "</img>" + rest[min(cuts):]


def _select_model(ids):
    """Pick the OCR/vision model from the endpoint's advertised ids.

    Returns the first id containing any `config.NANONETS_MODEL_PREFER` keyword
    (case-insensitive), else the first id. Empty list raises, same as before.
    """
    for keyword in config.NANONETS_MODEL_PREFER:
        for model_id in ids:
            if keyword in model_id.lower():
                return model_id
    return ids[0]


class NanonetsClient:
    """Wrapper around the OpenAI-compatible Nanonets-OCR endpoint."""

    name = "nanonets"  # engine label for the shared pipeline's logging

    def __init__(
        self, base_url: str = config.NANONETS_BASE_URL, model=config.NANONETS_MODEL
    ):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key="not-needed")
        self._model = model

    @property
    def model(self) -> str:
        """Resolved model id; auto-detected from /v1/models when not configured.

        The endpoint can serve more than one model (e.g. a text-only chat model
        listed *before* the OCR one), so we don't blindly take the first id --
        that silently drives the wrong, non-vision model. Prefer an id matching
        `config.NANONETS_MODEL_PREFER`, falling back to the first id, and log the
        pick so a mis-selection is visible rather than surfacing later as an
        opaque 500/empty transcription.
        """
        if self._model is None:
            ids = [m.id for m in self._client.models.list().data]
            self._model = _select_model(ids)
            print(f"[nanonets] using model: {self._model}")
        return self._model

    def parse_page(
        self,
        image: Image.Image,
        temperature: float = config.NANONETS_TEMPERATURE,
        mask_boxes=None,
    ) -> tuple[str, bool]:
        """OCR a whole page image; return ``(markdown, runaway)``.

        `temperature` defaults to the greedy `config.NANONETS_TEMPERATURE` (0.0)
        for deterministic, faithful transcription; a series can raise it via its
        LayoutOptions when greedy decoding loops on its pages (see
        `config.LayoutOptions.nanonets_temperature`), and `pipeline._ocr_page`
        escalates it further to recover a looping page.

        `mask_boxes` is an optional list of ``(x0, y0, x1, y1)`` rectangles to
        blank (fill `config.NANONETS_MASK_FILL`) before transcription -- the
        figure-masking rung of the runaway ladder passes DETR's figure boxes so
        the looping region is gone. The returned bool is True when the stream was
        aborted by the runaway guard (a truncated, incomplete transcription); the
        caller uses it to decide whether to retry and whether to cache.
        """
        image = image.convert("RGB")
        if mask_boxes:
            from PIL import ImageDraw

            image = image.copy()
            draw = ImageDraw.Draw(image)
            for box in mask_boxes:
                draw.rectangle(tuple(box), fill=config.NANONETS_MASK_FILL)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
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
        runaway = False
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
                    runaway = True
                    break
        print("\n[nanonets] Images processed / streaming complete")
        return _close_dangling_img("".join(full_content)), runaway


def _clean_text_line(line: str) -> str:
    return _BLANK_RUN_RE.sub("", line).strip()


def parse_layout(markdown: str, match_marker=None, split_marker_table_rows=False,
                 start_problem=None, ordered_list_markers=False):
    """Turn Nanonets markdown into an ordered list of items.

    Each item is a dict ``{"kind": "text"|"image", "problem": int|None, "text": str}``
    in reading order. ``problem`` is the number of the most recent problem marker
    seen above the item (``None`` for page-header content before problem 1).
    ``text`` is the statement text (markers stripped) or the image description.

    `match_marker` is an optional series-specific marker matcher (see
    anchors._match_marker); it defaults to the built-in pattern set.

    `start_problem` seeds the "current problem" state, for parsing one page of a
    multi-page document: content before the page's first marker binds to the
    problem carried in from the previous page instead of ``None``, and the
    strictly-increasing marker guard continues from it.

    `split_marker_table_rows` is a series-scoped opt-in (see
    config.LayoutOptions): when True, a ``<table>`` block is folded in row by row
    and any row whose leading cell is a problem marker is rewritten to plain
    statement text (MATHCOUNTS answer-blank tables). When False (the default),
    every table block is kept verbatim as HTML -- the right choice for series
    whose tables are genuine tabular data, not packed problem lists.

    `ordered_list_markers` is another series-scoped opt-in: when True, a line
    opening an ``<ol>/<li>`` list item begins the next sequential problem, for
    rounds that number problems by list position rather than a literal ``N.``
    marker (see config.LayoutOptions).
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
    current = start_problem  # current effective problem number
    last_raw = start_problem  # highest raw problem number in current section
    max_raw = start_problem or 0  # max raw problem number in current section
    offset = 0  # cumulative problem offset across section restarts
    saw_heading = False

    _HEADING_RE = re.compile(
        r"^\s*(?:#+|\*{1,2})\s*(?:[A-Z0-9].*?)(?:\*{1,2})?\s*$", re.IGNORECASE
    )

    def flush():
        if not buf:
            return
        text = "\n".join(buf).strip()
        buf.clear()
        if text:
            items.append({"kind": "text", "problem": current, "text": text})

    def check_marker(raw_num: int) -> bool:
        """Check if `raw_num` starts a new problem or a new section.

        Flushes `buf` under the previous problem before updating state.
        Returns True if accepted (updating state), False if rejected."""
        nonlocal current, last_raw, max_raw, offset, saw_heading
        is_restart = False
        if last_raw is not None and raw_num <= last_raw:
            if saw_heading or raw_num == 1 or (start_problem is not None and raw_num < start_problem):
                is_restart = True

        if is_restart:
            flush()
            offset += max_raw
            last_raw = raw_num
            max_raw = raw_num
            current = raw_num + offset
            saw_heading = False
            return True
        elif last_raw is None or raw_num > last_raw:
            flush()
            last_raw = raw_num
            max_raw = max(max_raw, raw_num)
            current = raw_num + offset
            saw_heading = False
            return True
        return False

    def consume_lines(chunk):
        """Fold a run of plain text (no whole table block) into buf.

        Handles marker detection and per-line cleanup; any stray/unclosed table
        tag left in `chunk` is flattened to a line break.
        """
        nonlocal current, last_raw, saw_heading
        text = _TABLE_TAG_RE.sub("\n", chunk)
        text = _split_glued_markers(text, match_marker)
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or (
                line.startswith("**")
                and (
                    "[" in line
                    or "Section" in line
                    or "Round" in line
                    or "Division" in line
                    or "Part" in line
                )
            ):
                saw_heading = True
            if ordered_list_markers:
                # A line opening an <li> begins the next problem (numbered by
                # list position; the printed "N." OCR'd into a separate graphic
                # column). Flatten the list/line-break scaffolding either way so
                # no <ol>/<li>/<br> tag leaks into the statement.
                if _LI_OPEN_RE.search(line):
                    raw_num = (last_raw or 0) + 1
                    check_marker(raw_num)
                line = _LIST_TAG_RE.sub(" ", line).strip()
                if not line:
                    continue
            # Markers can arrive markdown-emphasized ("**1/1/12.**", "*26.*",
            # "### 5."); strip leading emphasis/heading chars so the matcher sees
            # the bare marker. Match before blank runs are scrubbed, so an answer
            # line's "N. ____ unit" can be told apart from a real statement.
            probe = line.lstrip("*_# ")
            match = match_marker(probe)
            if match is not None and check_marker(match[0]):
                # Drop the marker and any emphasis closer it left behind ("**").
                line = probe[match[1] :].lstrip("*_ ")
            elif (
                match is not None
                and last_raw is not None
                and match[0] <= last_raw
                and not _clean_text_line(probe[match[1] :].lstrip("*_ "))
            ):
                # A marker repeating an already-seen number with nothing but a
                # blank rule after it ("60. ____") is the problem's printed
                # answer line, not a statement -- older MATHCOUNTS rounds print
                # one under every problem. It is not a new marker (the number
                # doesn't increase), so drop the whole line instead of letting
                # the bare "60." dangle onto the end of the statement.
                continue
            line = _clean_text_line(line)
            if not line or _POINTS_ONLY_RE.match(line):
                continue
            buf.append(line)

    def consume_table(html):
        """Fold a <table>...</table> block in row by row, checking each row's
        leading cell for a marker.

        MATHCOUNTS answer-blank tables pack many problems into one table
        ("1. ____ | <statement>", "2. ____ | <statement>", ...), so the marker
        lives inside a cell rather than before the table. Scanning the whole
        block as one opaque unit (the old behavior) meant every row silently
        inherited whatever `current` was before the table -- usually None,
        which drops the entire table's problems.

        Nanonets renders this exact layout as a table on some pages and as
        plain "N. ____ statement" lines on others (same content, no visible
        difference on the page), so a row recognized as a marker row is
        treated the same as the plain-text case: the marker/blank cell is
        dropped and only the remaining cells' text is kept, tags stripped --
        not the raw `<table>` markup, which would otherwise leak into the
        statement inconsistently depending on which format the model picked.
        A row whose leading cell does *not* parse as a marker is assumed to be
        real tabular data belonging to the current problem's statement (e.g. a
        table of values the problem itself refers to), and is kept verbatim as
        its own single-row table so that formatting survives.
        """
        nonlocal current, last_raw
        rows = _ROW_RE.findall(html)
        if not rows:
            row_html = re.sub(r"\s+", " ", html).strip()
            if row_html:
                buf.append(row_html)
            return
        for row in rows:
            cells = _CELL_RE.findall(row)
            first_cell = _TAG_RE.sub(" ", cells[0]).strip() if cells else ""
            probe = first_cell.lstrip("*_# ")
            m = match_marker(probe)
            if m is not None and check_marker(m[0]):
                # The statement and the answer blank each live in their own
                # cell, but which column holds which varies by year: some print
                # "N. ____ unit | <statement>", older rounds print the reverse,
                # "<statement> | N. ____". So strip a leading marker from *every*
                # cell (not just the first) and keep whatever text survives. A
                # cell holding the answer blank is recognized by the blank rule
                # it contains; the blank *and the unit that labels it* ("cm$^2$",
                # a lone "$") are dropped, so only a real statement survives --
                # whichever side it was on.
                parts = []
                for c in cells:
                    cell = re.sub(r"\s+", " ", _TAG_RE.sub(" ", c)).strip()
                    # Note the blank rule *before* stripping the marker: doing so
                    # lstrips the underscores away too, so this is the last point
                    # the answer-blank cell can be told apart from a statement.
                    is_answer = bool(_BLANK_RUN_RE.search(cell))
                    probe_cell = cell.lstrip("*_# ")
                    cm = match_marker(probe_cell)
                    if cm is not None:
                        cell = probe_cell[cm[1] :].lstrip("*_ ")
                    if is_answer:
                        # Drop the unit that labeled the blank ("cm$^2$", a lone
                        # "$"). Anything after it is a statement sharing the cell
                        # (rare) and is kept.
                        cell = _ANSWER_UNIT_RE.sub("", _clean_text_line(cell), count=1)
                    cell = _clean_text_line(cell)
                    if cell and not _POINTS_ONLY_RE.match(cell):
                        parts.append(cell)
                statement = " ".join(parts).strip()
                if statement:
                    buf.append(statement)
                continue
            # A non-marker row is real tabular data belonging to the current
            # problem -- kept verbatim -- unless it is nothing but a point-value
            # cell ("<td>①</td>"), a standalone furniture row emitted beside the
            # problems, which is dropped.
            row_text = _TAG_RE.sub(" ", row).strip()
            if row_text and _POINTS_ONLY_RE.match(row_text):
                continue
            buf.append(re.sub(r"\s+", " ", f"<table>{row}</table>").strip())

    for kind, payload in tokens:
        if kind == "image":
            flush()
            items.append({"kind": "image", "problem": current, "text": payload})
            continue

        # Keep each <table>...</table> block as HTML; only the text around the
        # tables is scanned for problem markers. When a series opts into
        # `split_marker_table_rows`, each row's leading cell is scanned too and
        # marker rows are unpacked into plain statements (see consume_table);
        # otherwise the whole block is kept verbatim.
        pos = 0
        for tm in _TABLE_BLOCK_RE.finditer(payload):
            consume_lines(payload[pos : tm.start()])
            if split_marker_table_rows:
                consume_table(tm.group(0))
            else:
                html = re.sub(r"\s+", " ", tm.group(0)).strip()
                if html:
                    buf.append(html)
            pos = tm.end()
        consume_lines(payload[pos:])
    flush()
    return items
