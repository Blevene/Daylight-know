"""Shared LLM call utilities: response format, user prompt building, and retry logic.

Centralises the backoff loop, structured output schema construction, and
JSON parsing so that ``summarizer`` and ``postprocessor`` stay thin wrappers.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import litellm

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)


def build_response_format(name: str, num_papers: int) -> dict[str, Any]:
    """Build a ``response_format`` dict with explicit ``paper_N`` keys.

    Generates a strict JSON schema that enumerates every expected key so that
    providers supporting structured output (OpenAI strict mode, etc.) can
    enforce the shape at the API level.
    """
    properties = {
        f"paper_{i}": {"type": "string"} for i in range(1, num_papers + 1)
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
                "additionalProperties": False,
            },
        },
    }


def build_user_prompt(papers: list[Paper], github_section: str = "") -> str:
    """Format papers (and optional GitHub section) into the LLM user message."""
    parts: list[str] = []
    for i, p in enumerate(papers, 1):
        lines = [
            f"### Paper {i}: {p.title}\n",
            f"**Source:** {p.source}\n",
            f"**Authors:** {', '.join(p.authors)}\n",
        ]
        if p.categories:
            lines.append(f"**Categories:** {', '.join(p.categories)}\n")
        if p.fields_of_study:
            lines.append(f"**Fields of Study:** {', '.join(p.fields_of_study)}\n")
        if p.upvotes:
            lines.append(f"**Community Upvotes:** {p.upvotes}\n")
        lines.append(f"**URL:** {p.url}\n\n")
        lines.append(f"{p.abstract}\n")
        parts.append("".join(lines))
    prompt = "\n---\n".join(parts)
    if github_section:
        prompt += f"\n\n---\n## Trending GitHub Repositories\n{github_section}"
    return prompt


def llm_call(
    papers: list[Paper],
    system_prompt: str,
    settings: Settings,
    label: str,
    schema_name: str = "paper_analysis",
    github_section: str = "",
) -> dict[str, str]:
    """Perform an LLM completion with structured output, token limit, and backoff.

    Returns a dict mapping ``paper_N`` keys to analysis strings.
    Enforces ``settings.llm_max_tokens`` and retries with exponential
    backoff on rate-limit errors.
    """
    user_prompt = build_user_prompt(papers, github_section)
    response_format = build_response_format(schema_name, len(papers))

    max_backoff_attempts = 5
    for attempt in range(1, max_backoff_attempts + 1):
        try:
            response = litellm.completion(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.llm_api_key,
                api_base=settings.llm_api_base,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
            )
            raw = response.choices[0].message.content or ""
            logger.info("LLM %s complete (%d chars).", label, len(raw))
            if not raw:
                return {}
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                logger.error(
                    "LLM %s returned invalid JSON: %.200s", label, raw,
                )
                return {}
            if not isinstance(parsed, dict):
                logger.error(
                    "LLM %s returned non-object JSON: %s", label, type(parsed),
                )
                return {}
            return {k: str(v) for k, v in parsed.items()}
        except litellm.RateLimitError as exc:
            wait = 2 ** attempt
            logger.warning(
                "Rate-limited during %s (attempt %d/%d). Retrying in %ds: %s",
                label, attempt, max_backoff_attempts, wait, exc,
            )
            if attempt == max_backoff_attempts:
                raise
            time.sleep(wait)

    return {}  # unreachable, but satisfies type checkers
