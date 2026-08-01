"""Last-resort answer extraction from solution prose.

Deterministic answer parsing (a series' `parse_answers`) keys the final answer
off a printed marker -- "Answer:", a ``\\boxed{...}``, PUMaC's "(ANS: ...)".
Older material states no such marker: the answer sits mid-sentence in the worked
solution with nothing to anchor a regex on. When a series' own parsing comes up
empty, it may fall back to `extract`, which reads the answer out of the
statement+solution text with a text LLM.

By default this reuses the OCR engine's OpenAI-compatible endpoint
(`config.ANSWER_LLM_*`) -- the OCR VLM served there doubles as a competent text
extractor, so the pipeline stays local -- and is swappable to any stronger chat
model. It is deliberately failure-tolerant: any endpoint or parse problem yields
``None`` (the problem is left out of the answer key) rather than raising. An
answer key is allowed to be partial; it must never be wrong-by-crash, nor block
the rest of a run because a local endpoint is down.
"""

import re

from . import config

_client = None
_model = None
_warned = False  # endpoint failures are logged once per run, not per problem


def _ensure_client():
    global _client, _model
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(base_url=config.ANSWER_LLM_BASE_URL, api_key="not-needed")
        _model = config.ANSWER_LLM_MODEL
        if _model is None:
            _model = _client.models.list().data[0].id
            print(f"[answer-llm] using model: {_model}")
    return _client, _model


def _unwrap_boxed(value):
    r"""Unwrap a reply that echoed the source's ``\boxed{...}``, or return it as is.

    Models routinely copy the box along with the answer. Only an exact wrapper is
    an echo: trailing text means the box was merely the reply's first clause, so
    the value is left alone rather than silently truncated. Braces are matched by
    depth so a nested ``\frac{a}{b}`` survives.
    """
    opener = r"\boxed{"
    if not value.startswith(opener):
        return value
    depth = 0
    for index in range(len(opener) - 1, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                if index != len(value) - 1:
                    return value
                return value[len(opener) : index]
    return value


def _clean(reply):
    """Reduce a raw LLM reply to a bare answer string, or None.

    Strips code fences, LaTeX delimiters, an echoed answer box, and surrounding
    emphasis; keeps only the first line (a multi-line reply is prose, not an
    answer); and treats an empty reply or an "UNKNOWN" refusal as no answer.
    """
    if not reply:
        return None
    out = reply.strip().splitlines()[0].strip() if reply.strip() else ""
    out = out.strip("`").strip()
    out = re.sub(r"^\$+|\$+$", "", out).strip()
    out = re.sub(r"^\\[(\[]|\\[)\]]$", "", out).strip()
    out = re.sub(r"^\*+|\*+$", "", out).strip()
    out = _unwrap_boxed(out).strip()
    out = re.sub(r"^\$+|\$+$", "", out).strip()
    if not out or "UNKNOWN" in out.upper():
        return None
    return out


def extract(problem_block):
    """Return the final answer read from a statement+solution block, or None.

    `problem_block` is the full text for one problem (its restated statement and
    worked solution). Returns None when the fallback is disabled, the text is
    empty, the endpoint is unreachable, or the model declines to commit to an
    answer -- in every such case the caller simply omits the problem.
    """
    if not config.ANSWER_LLM_ENABLED:
        return None
    text = (problem_block or "").strip()
    if not text:
        return None
    global _warned
    try:
        client, model = _ensure_client()
        resp = client.chat.completions.create(
            model=model,
            temperature=config.ANSWER_LLM_TEMPERATURE,
            max_tokens=config.ANSWER_LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": config.ANSWER_LLM_PROMPT + text}],
        )
    except Exception as exc:  # endpoint down, model error, network -- non-fatal
        if not _warned:
            print(f"[answer-llm] fallback disabled for this run: {exc}")
            _warned = True
        return None
    return _clean(resp.choices[0].message.content)
