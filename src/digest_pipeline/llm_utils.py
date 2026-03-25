"""Shared LLM call utilities: response format, user prompt building, and retry logic.

Centralises the backoff loop, structured output schema construction, and
JSON parsing so that ``summarizer`` and ``postprocessor`` stay thin wrappers.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import litellm

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)

LLM_BATCH_SIZE = 7  # max papers per LLM call to avoid output truncation

_FENCE_RE = re.compile(r"^```\w*\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```$")
_BOLD_HEADER_RE = re.compile(r"(?<!\n)(\*\*[^*]+:\*\*)")
_BULLET_RE = re.compile(r"(?<!\n)([•\-] )")


def _normalize_markdown_bullets(text: str) -> str:
    """Ensure markdown bullet points and bold headers have proper newlines.

    LLMs (notably Gemini) often compress structured markdown into a single
    line within JSON string values, e.g.::

        "**Strengths:** - point 1 - point 2 **Weaknesses:** - point 3"

    or using Unicode bullets::

        "**Strengths:** • point 1 • point 2 **Weaknesses:** • point 3"

    This function inserts newlines so that mistune can parse bullet lists
    and bold headers correctly.
    """
    # Normalize Unicode bullets to markdown dashes
    text = text.replace("• ", "- ")
    # Ensure bold section headers (e.g. **Strengths:**) start on a new line
    text = _BOLD_HEADER_RE.sub(r"\n\n\1", text)
    # Ensure each bullet item starts on a new line
    text = _BULLET_RE.sub(r"\n\1", text)
    return text.strip()


def parse_llm_json(raw: str) -> Any:
    """Parse JSON from an LLM response, stripping markdown code fences if present.

    Many LLM providers (notably Gemini) wrap JSON output in markdown fences
    like ````` ```json ... ``` `````.  This helper strips those before parsing.
    """
    cleaned = _FENCE_RE.sub("", raw.strip())
    cleaned = _FENCE_END_RE.sub("", cleaned)
    return json.loads(cleaned)


def build_response_format(name: str, num_papers: int) -> dict[str, Any]:
    """Build a ``response_format`` dict with explicit ``paper_N`` keys.

    Generates a strict JSON schema that enumerates every expected key so that
    providers supporting structured output (OpenAI strict mode, etc.) can
    enforce the shape at the API level.
    """
    properties = {f"paper_{i}": {"type": "string"} for i in range(1, num_papers + 1)}
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


def build_user_prompt(papers: list[Paper]) -> str:
    """Format papers into the LLM user message."""
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
    return "\n---\n".join(parts)


def _llm_call_single(
    papers: list[Paper],
    system_prompt: str,
    settings: Settings,
    label: str,
    schema_name: str = "paper_analysis",
) -> dict[str, str]:
    """Perform a single LLM completion for one batch of papers.

    Returns a dict mapping ``paper_N`` keys to analysis strings.
    Enforces ``settings.llm_max_tokens`` and retries with exponential
    backoff on rate-limit errors and malformed responses.
    """
    user_prompt = build_user_prompt(papers)
    response_format = build_response_format(schema_name, len(papers))
    expected_keys = {f"paper_{i}" for i in range(1, len(papers) + 1)}

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
                logger.warning(
                    "LLM %s returned empty content (attempt %d/%d).",
                    label,
                    attempt,
                    max_backoff_attempts,
                )
                if attempt < max_backoff_attempts:
                    time.sleep(2**attempt)
                    continue
                return {}
            try:
                parsed = parse_llm_json(raw)
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "LLM %s returned invalid JSON (attempt %d/%d): %.200s",
                    label,
                    attempt,
                    max_backoff_attempts,
                    raw,
                )
                if attempt < max_backoff_attempts:
                    time.sleep(2**attempt)
                    continue
                return {}
            if not isinstance(parsed, dict):
                logger.warning(
                    "LLM %s returned non-object JSON (attempt %d/%d): %s",
                    label,
                    attempt,
                    max_backoff_attempts,
                    type(parsed),
                )
                if attempt < max_backoff_attempts:
                    time.sleep(2**attempt)
                    continue
                return {}
            result = {k: str(v) for k, v in parsed.items()}
            missing = expected_keys - result.keys()
            if missing:
                logger.warning(
                    "LLM %s missing keys %s (attempt %d/%d).",
                    label,
                    sorted(missing),
                    attempt,
                    max_backoff_attempts,
                )
            return result
        except litellm.RateLimitError as exc:
            wait = 2**attempt
            logger.warning(
                "Rate-limited during %s (attempt %d/%d). Retrying in %ds: %s",
                label,
                attempt,
                max_backoff_attempts,
                wait,
                exc,
            )
            if attempt == max_backoff_attempts:
                logger.error(
                    "LLM %s rate-limited after %d attempts; skipping.", label, max_backoff_attempts
                )
                return {}
            time.sleep(wait)
        except Exception:
            logger.exception(
                "LLM %s failed on attempt %d/%d.",
                label,
                attempt,
                max_backoff_attempts,
            )
            if attempt == max_backoff_attempts:
                return {}
            time.sleep(2**attempt)

    return {}  # unreachable, but satisfies type checkers


def llm_call(
    papers: list[Paper],
    system_prompt: str,
    settings: Settings,
    label: str,
    schema_name: str = "paper_analysis",
) -> dict[str, str]:
    """Perform LLM completion with automatic batching, structured output, and retry.

    Splits papers into batches of ``LLM_BATCH_SIZE`` to avoid output
    truncation, calls the LLM for each batch, and merges results with
    globally consistent ``paper_N`` keys.
    """
    if len(papers) <= LLM_BATCH_SIZE:
        return _llm_call_single(papers, system_prompt, settings, label, schema_name)

    merged: dict[str, str] = {}
    for batch_start in range(0, len(papers), LLM_BATCH_SIZE):
        batch = papers[batch_start : batch_start + LLM_BATCH_SIZE]
        batch_num = batch_start // LLM_BATCH_SIZE + 1
        batch_label = f"{label} batch {batch_num}"
        batch_result = _llm_call_single(
            batch, system_prompt, settings, batch_label, schema_name
        )
        # Re-key from batch-local paper_N to global paper_N
        for local_key, value in batch_result.items():
            # local_key is "paper_1", "paper_2", etc. within the batch
            local_idx = int(local_key.split("_")[1])
            global_idx = batch_start + local_idx
            merged[f"paper_{global_idx}"] = value

    return merged
