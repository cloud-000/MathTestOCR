"""Per-crop OCR via a local MLX VLM.

The VLM is used ONLY for the narrow task it handles well: read the text in a
small crop. It never reasons about whole-page layout.
"""

import gc

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from PIL import Image

from . import config

_LATEX_PROMPT = (
    "Convert the problem to LaTeX format. The delimiters are $ for inline and "
    "$$ for centered/display equations. Normal text stays as normal text, "
    "outside the LaTeX delimiters. Output only the converted text, nothing else."
)


class OCRModel:
    """Thin wrapper around an MLX VLM, used for crop OCR."""

    def __init__(self, name: str = config.OCR_MODEL):
        self.name = name
        self.model = None
        self.processor = None
        self.config = None

    def load(self):
        if self.model is None:
            self.model, self.processor = load(self.name)
            self.config = self.model.config
        return self

    def _generate(self, prompt: str, image: Image.Image, think: bool = False) -> str:
        self.load()
        formatted = apply_chat_template(
            self.processor, self.config, prompt, num_images=1, enable_thinking=think
        )
        return generate(
            self.model,  # type: ignore[reportArgumentType]
            self.processor,  # type: ignore[reportArgumentType]
            formatted,  # type: ignore[reportArgumentType]
            [image],  # type: ignore[reportArgumentType]
            verbose=False,
        ).text

    def read_text(self, crop: Image.Image) -> str:
        """Plain transcription of a crop (used for anchor detection)."""
        return self._generate(
            "Transcribe the text in this image exactly. Output only the text.", crop
        ).strip()

    def latex_ocr(self, crop: Image.Image) -> str:
        """Transcription with math rendered as LaTeX (used for final output)."""
        return self._generate(_LATEX_PROMPT, crop).strip()

    def unload(self):
        self.model = None
        self.processor = None
        self.config = None
        gc.collect()
        mx.clear_cache()
