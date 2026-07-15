"""Llama engine: whole-page OCR via the hosted LlamaCloud parsing API.

A cloud alternative to the local nanonets endpoint. Like nanonets, it sees a
whole page once and returns problem-segmented markdown (LaTeX for math, HTML for
tables); the DETR layout model still supplies the actual figure crops and all
segmentation stays deterministic geometry (see pipeline.py). Because it produces
the same whole-page markdown, it is interchangeable with the nanonets engine --
it exposes the identical ``parse_page(image, temperature, mask_boxes) ->
(markdown, runaway)`` contract, so ``pipeline.process_image_markdown`` and the
solution/answer paths drive it unchanged (both are listed in
``config.MARKDOWN_ENGINES``). This module only talks to the API and turns a page
image into markdown; the engine-agnostic ``nanonets.parse_layout`` handles it
downstream.

Requires a LlamaCloud API key: set ``LLAMA_CLOUD_API_KEY`` (or
``LLAMA_PARSE_API_KEY``) in the environment, or pin ``config.LLAMA_CLOUD_API_KEY``.
"""

import io

from PIL import Image

from . import config


def _result_markdown(result) -> str:
    """Pull the page markdown out of a ParsingGetResponse (expand=["markdown"]).

    The ``markdown`` field is a ``Markdown`` object whose ``pages`` each carry a
    ``.markdown`` string; we feed one page image but join defensively in case the
    service splits it. Falls back to the plain ``text`` field (same page shape)
    so a result parsed without markdown is never silently dropped.
    """
    for field in ("markdown", "text"):
        obj = getattr(result, field, None)
        pages = getattr(obj, "pages", None)
        if pages:
            parts = [getattr(p, field, None) for p in pages]
            joined = "\n\n".join(p for p in parts if p)
            if joined:
                return joined
    return ""


class LlamaClient:
    """Wrapper around the hosted LlamaCloud parsing API (matches NanonetsClient).

    Deliberately mirrors ``NanonetsClient.parse_page`` so the two engines are
    drop-in interchangeable in the pipeline. Differences the contract papers
    over: ``temperature`` is accepted for signature parity and ignored (a hosted
    service, not a local sampler), and the returned ``runaway`` flag is always
    False -- there is no local greedy-decoding loop to guard against, so the
    pipeline's runaway-recovery ladder (retry temps / figure masking) simply
    never triggers for this engine.
    """

    name = "llama"  # engine label for the shared pipeline's logging

    def __init__(
        self,
        api_key: str | None = None,
        tier: str = config.LLAMA_TIER,
        version: str = config.LLAMA_VERSION,
        prompt: str | None = config.LLAMA_PROMPT,
    ):
        from llama_cloud import LlamaCloud

        # api_key=None lets the SDK read LLAMA_CLOUD_API_KEY / LLAMA_PARSE_API_KEY
        # from the environment; config.LLAMA_CLOUD_API_KEY pins it explicitly.
        self._client = LlamaCloud(api_key=api_key or config.LLAMA_CLOUD_API_KEY)
        self._tier = tier
        self._version = version
        # Custom parsing instructions (agentic_options.custom_prompt) -- used to
        # make the parser emit inline <img> figure markers (see config.LLAMA_PROMPT).
        # Only the AI (non-fast) tiers accept it, so it is dropped on the fast tier.
        self._prompt = prompt

    def parse_page(
        self,
        image: Image.Image,
        temperature: float = config.NANONETS_TEMPERATURE,
        mask_boxes=None,
    ) -> tuple[str, bool]:
        """OCR a whole page image; return ``(markdown, runaway)``.

        `temperature` is ignored (see the class docstring). `mask_boxes` is an
        optional list of ``(x0, y0, x1, y1)`` rectangles blanked
        (``config.NANONETS_MASK_FILL``) before upload, honored for parity with
        NanonetsClient -- though the runaway ladder that supplies them never
        fires here, since `runaway` is always False. The page is uploaded and
        the parse job polled to completion by the SDK; the resulting page
        markdown is returned.
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
        # The parse() convenience method uploads, polls to completion, and fetches
        # the result in one call; `expand` selects which content the fetch returns
        # (it raises on an empty sequence), so ask for markdown explicitly. The
        # custom prompt (inline <img> markers) rides agentic_options and is only
        # valid on the AI tiers -- omit it on "fast" so the request still succeeds.
        kwargs = {}
        if self._prompt and self._tier != "fast":
            kwargs["agentic_options"] = {"custom_prompt": self._prompt}
        result = self._client.parsing.parse(
            tier=self._tier,
            version=self._version,
            upload_file=("page.png", buf.getvalue(), "image/png"),
            expand=["markdown"],
            **kwargs,
        )
        return _result_markdown(result), False
