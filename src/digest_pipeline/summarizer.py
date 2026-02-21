"""LLM-powered summarization with token limits and exponential backoff.

Uses litellm as the LLM adapter, enabling any provider (OpenAI, Anthropic,
Cohere, local models, etc.) via the ``LLM_MODEL`` setting.

EARS coverage
─────────────
- Event 2.2-3: pass paper abstracts to LLM after chunking/storage is complete.
- State 2.3-1: enforce strict max-token limit during LLM queries.
- Unwanted 2.4-4: exponential backoff on rate-limit errors.
"""

from __future__ import annotations

import logging
import time

import litellm

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert research assistant. Given a set of academic paper "
    "abstracts, produce a concise daily digest highlighting the key findings, "
    "novel contributions, and practical implications. Group related papers "
    "together when possible."
)


def _build_user_prompt(papers: list[Paper], github_section: str = "") -> str:
    parts: list[str] = []
    for i, p in enumerate(papers, 1):
        parts.append(
            f"### Paper {i}: {p.title}\n"
            f"**Authors:** {', '.join(p.authors)}\n"
            f"**URL:** {p.url}\n\n"
            f"{p.abstract}\n"
        )
    prompt = "\n---\n".join(parts)
    if github_section:
        prompt += f"\n\n---\n## Trending GitHub Repositories\n{github_section}"
    return prompt


def summarize(
    papers: list[Paper],
    settings: Settings,
    github_section: str = "",
) -> str:
    """Generate an LLM summary of the paper abstracts.

    Enforces ``settings.llm_max_tokens`` (EARS 2.3-1) and retries with
    exponential backoff on rate-limit errors (EARS 2.4-4).
    """
    user_prompt = _build_user_prompt(papers, github_section)

    max_backoff_attempts = 5
    for attempt in range(1, max_backoff_attempts + 1):
        try:
            response = litellm.completion(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.llm_api_key,
                api_base=settings.llm_api_base,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            summary = response.choices[0].message.content or ""
            logger.info("LLM summarization complete (%d chars).", len(summary))
            return summary
        except litellm.RateLimitError as exc:
            wait = 2 ** attempt
            logger.warning(
                "Rate-limited (attempt %d/%d). Retrying in %ds: %s",
                attempt, max_backoff_attempts, wait, exc,
            )
            if attempt == max_backoff_attempts:
                raise
            time.sleep(wait)

    return ""  # unreachable, but satisfies type checkers
