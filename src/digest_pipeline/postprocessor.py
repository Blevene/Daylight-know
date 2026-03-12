"""LLM-powered post-processing: practical implications and critiques.

Provides additional analysis layers on top of the base summarization.
Each function makes a separate LLM call with a specialised system prompt,
reusing the shared backoff and token-limit logic from ``llm_utils``.

Returns dicts mapping paper keys (``paper_1``, ``paper_2``, …) to analysis
strings so the pipeline can display each paper's analysis grouped together.

Uses litellm as the LLM adapter, enabling any provider (OpenAI, Anthropic,
Cohere, local models, etc.) via the ``LLM_MODEL`` setting.
"""

from __future__ import annotations

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper
from digest_pipeline.llm_utils import _normalize_markdown_bullets, llm_call
from digest_pipeline.prompts import load_prompt

IMPLICATIONS_SYSTEM_PROMPT = load_prompt("implications")
CRITIQUES_SYSTEM_PROMPT = load_prompt("critiques")


def extract_implications(papers: list[Paper], settings: Settings) -> dict[str, str]:
    """Generate per-paper practical implications.

    Returns a dict mapping ``paper_N`` keys to implication strings.
    Enforces ``settings.llm_max_tokens`` and retries with exponential
    backoff on rate-limit errors, matching the summarizer contract.
    """
    raw = llm_call(
        papers,
        IMPLICATIONS_SYSTEM_PROMPT,
        settings,
        label="implications",
        schema_name="paper_implications",
    )
    return {k: _normalize_markdown_bullets(v) for k, v in raw.items()}


def generate_critiques(papers: list[Paper], settings: Settings) -> dict[str, str]:
    """Generate per-paper critical analysis.

    Returns a dict mapping ``paper_N`` keys to critique strings.
    Enforces ``settings.llm_max_tokens`` and retries with exponential
    backoff on rate-limit errors, matching the summarizer contract.
    """
    raw = llm_call(
        papers,
        CRITIQUES_SYSTEM_PROMPT,
        settings,
        label="critiques",
        schema_name="paper_critiques",
    )
    return {k: _normalize_markdown_bullets(v) for k, v in raw.items()}
