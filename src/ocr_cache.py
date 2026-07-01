"""Persistent per-page OCR cache.

The Nanonets whole-page OCR pass (``NanonetsClient.parse_page``) is the slow part
of the pipeline; DETR layout detection is comparatively cheap and is *never*
cached (it re-runs every time). When a cache is enabled, each page's markdown
transcription is stored in a JSON file inside the test's output directory, keyed
by page filename. On a hit the model call is skipped and the stored markdown is
returned; on a miss the OCR runs and its result is written back.

Keying by page *filename* (not absolute path) makes the cache survive across
runs, even though each run renders the PDF into a fresh temp dir -- ``pdf_io``
emits deterministic page names, so ``page-1.png`` maps to the same page every
time. A cache file is scoped to one test *and one document role* (statements vs.
solutions render to colliding page names), so callers pass a role-specific
filename.
"""

import json
from pathlib import Path

PARSE_CACHE = "ocr_cache.json"
SOLUTION_CACHE = "ocr_cache_solutions.json"


class OCRCache:
    """A JSON-backed {page_filename: markdown} cache for one test document.

    When `enabled` is False every lookup is a miss that runs the OCR and stores
    nothing -- so callers can construct one unconditionally and let the flag
    decide. Writes are flushed to disk after each new page so a long run that is
    interrupted still keeps the pages it already OCR'd.
    """

    def __init__(self, path, enabled=False):
        self.path = Path(path)
        self.enabled = enabled
        self._data = {}
        if enabled and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}  # corrupt/unreadable cache -> just re-OCR

    def page_markdown(self, page_path, ocr_fn):
        """Return cached markdown for `page_path`, else run `ocr_fn` and cache it.

        `ocr_fn` is a zero-argument callable that performs the actual OCR. It is
        only invoked on a miss.
        """
        key = Path(page_path).name
        if self.enabled and key in self._data:
            print(f"[cache] hit {key} ({self.path.name})")
            return self._data[key]
        markdown = ocr_fn()
        if self.enabled:
            self._data[key] = markdown
            self._flush()
        return markdown

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=0))
