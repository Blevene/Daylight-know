"""LLM-powered summarization with token limits and exponential backoff.

Uses litellm as the LLM adapter, enabling any provider (OpenAI, Anthropic,
Cohere, local models, etc.) via the ``LLM_MODEL`` setting.

Returns a dict mapping paper keys (``paper_1``, ``paper_2``, …) to summary
strings so the pipeline can display each paper's analysis grouped together.

EARS coverage
─────────────
- Event 2.2-3: pass paper abstracts to LLM after chunking/storage is complete.
- State 2.3-1: enforce strict max-token limit during LLM queries.
- Unwanted 2.4-4: exponential backoff on rate-limit errors.
"""

from __future__ import annotations

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper
from digest_pipeline.llm_utils import llm_call
from digest_pipeline.prompts import load_prompt

SYSTEM_PROMPT = load_prompt("summarizer")


def summarize(
    papers: list[Paper],
    settings: Settings,
    github_section: str = "",
) -> dict[str, str]:
    """Generate per-paper LLM summaries.

    Returns a dict mapping ``paper_N`` keys to summary strings.
    Enforces ``settings.llm_max_tokens`` (EARS 2.3-1) and retries with
    exponential backoff on rate-limit errors (EARS 2.4-4).
    """
    return llm_call(
        papers,
        SYSTEM_PROMPT,
        settings,
        label="summarization",
        schema_name="paper_summaries",
        github_section=github_section,
    )
