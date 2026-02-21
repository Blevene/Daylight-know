"""LLM-powered post-processing: practical implications and critiques.

Provides additional analysis layers on top of the base summarization.
Each function makes a separate LLM call with a specialised system prompt,
reusing the same backoff and token-limit patterns as the summarizer.
"""

from __future__ import annotations

import logging
import time

from openai import OpenAI, RateLimitError

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)

IMPLICATIONS_SYSTEM_PROMPT = (
    "You are an expert research analyst specialising in translating academic "
    "findings into actionable insights. Given a set of academic paper "
    "abstracts, identify concrete practical implications for practitioners, "
    "engineers, and industry professionals. For each paper (or group of "
    "related papers), describe: (1) who would benefit from these findings, "
    "(2) how the results could be applied in real-world settings, and "
    "(3) any prerequisites or limitations that affect practical adoption. "
    "Be specific and grounded — avoid vague generalities."
)

CRITIQUES_SYSTEM_PROMPT = (
    "You are a rigorous peer reviewer with broad expertise across machine "
    "learning, artificial intelligence, and related fields. Given a set of "
    "academic paper abstracts, provide a balanced critical analysis. For each "
    "paper (or group of related papers), address: (1) methodological "
    "strengths, (2) potential weaknesses or gaps (e.g. limited evaluation, "
    "strong assumptions, missing baselines), and (3) open questions that "
    "future work should tackle. Be fair but honest — highlight genuine "
    "contributions while noting areas for improvement."
)


def _build_user_prompt(papers: list[Paper]) -> str:
    parts: list[str] = []
    for i, p in enumerate(papers, 1):
        parts.append(
            f"### Paper {i}: {p.title}\n"
            f"**Authors:** {', '.join(p.authors)}\n"
            f"**URL:** {p.url}\n\n"
            f"{p.abstract}\n"
        )
    return "\n---\n".join(parts)


def _llm_call(
    papers: list[Paper],
    system_prompt: str,
    settings: Settings,
    label: str,
) -> str:
    """Shared LLM call with token limit and exponential backoff."""
    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    user_prompt = _build_user_prompt(papers)

    max_backoff_attempts = 5
    for attempt in range(1, max_backoff_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            logger.info("LLM %s complete (%d chars).", label, len(content))
            return content
        except RateLimitError as exc:
            wait = 2 ** attempt
            logger.warning(
                "Rate-limited during %s (attempt %d/%d). Retrying in %ds: %s",
                label, attempt, max_backoff_attempts, wait, exc,
            )
            if attempt == max_backoff_attempts:
                raise
            time.sleep(wait)

    return ""  # unreachable, but satisfies type checkers


def extract_implications(papers: list[Paper], settings: Settings) -> str:
    """Generate practical implications for the given papers.

    Enforces ``settings.llm_max_tokens`` and retries with exponential
    backoff on rate-limit errors, matching the summarizer contract.
    """
    return _llm_call(papers, IMPLICATIONS_SYSTEM_PROMPT, settings, "implications")


def generate_critiques(papers: list[Paper], settings: Settings) -> str:
    """Generate critical analysis of the given papers.

    Enforces ``settings.llm_max_tokens`` and retries with exponential
    backoff on rate-limit errors, matching the summarizer contract.
    """
    return _llm_call(papers, CRITIQUES_SYSTEM_PROMPT, settings, "critiques")
