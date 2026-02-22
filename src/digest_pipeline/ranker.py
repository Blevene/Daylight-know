"""Interest-based paper ranking with keyword boosting and LLM scoring.

Scores papers against a user-defined interest profile to select the most
relevant papers from a larger fetch pool. Used between OpenAlex fetching
and the main digest pipeline.
"""

from __future__ import annotations

import json
import logging

import litellm

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper
from digest_pipeline.prompts import load_prompt

logger = logging.getLogger(__name__)

KEYWORD_BOOST = 2
BATCH_SIZE = 20


def compute_keyword_scores(papers: list[Paper], keywords: list[str]) -> list[int]:
    """Score each paper by keyword matches in title and abstract.

    Returns a list of integer scores (one per paper). Each keyword match
    in the title adds KEYWORD_BOOST points, and each match in the
    abstract adds KEYWORD_BOOST points.
    """
    if not keywords:
        return [0] * len(papers)

    lower_keywords = [kw.lower() for kw in keywords]
    scores: list[int] = []
    for paper in papers:
        score = 0
        title_lower = paper.title.lower()
        abstract_lower = paper.abstract.lower()
        for kw in lower_keywords:
            if kw in title_lower:
                score += KEYWORD_BOOST
            if kw in abstract_lower:
                score += KEYWORD_BOOST
        scores.append(score)
    return scores


def score_batch_with_llm(papers: list[Paper], settings: Settings) -> list[int]:
    """Score a batch of papers using the LLM.

    Sends paper titles and abstracts to the configured LLM with the
    interest profile. Returns a list of integer scores (1-10) per paper.
    On failure, returns zeros for graceful degradation.
    """
    if not papers:
        return []

    system_prompt = load_prompt("ranker").replace(
        "{interest_profile}", settings.openalex_interest_profile
    )

    # Build user message with paper titles and abstracts
    parts: list[str] = []
    for i, p in enumerate(papers, 1):
        parts.append(f"paper_{i}: {p.title}\n{p.abstract[:500]}")
    user_prompt = "\n---\n".join(parts)

    # Build response format expecting integer scores
    properties = {
        f"paper_{i}": {"type": "integer"} for i in range(1, len(papers) + 1)
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "relevance_scores",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
                "additionalProperties": False,
            },
        },
    }

    try:
        response = litellm.completion(
            model=settings.llm_model,
            max_tokens=256,
            api_key=settings.llm_api_key,
            api_base=settings.llm_api_base,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        return [
            max(0, min(10, int(parsed.get(f"paper_{i}", 0))))
            for i in range(1, len(papers) + 1)
        ]
    except Exception:
        logger.warning("LLM scoring failed — falling back to zero scores.", exc_info=True)
        return [0] * len(papers)


def rank_papers(papers: list[Paper], settings: Settings) -> list[Paper]:
    """Rank papers by relevance and return the top N.

    Combines keyword boost scores with LLM-based relevance scoring.
    If no interest profile or keywords are configured, returns papers
    unchanged (backward compatible).
    """
    if not settings.openalex_interest_profile and not settings.openalex_interest_keywords:
        return papers

    max_results = settings.openalex_max_results
    if len(papers) <= max_results:
        return papers

    # 1. Keyword scores
    keyword_scores = compute_keyword_scores(papers, settings.openalex_interest_keywords)

    # 2. LLM scores (in batches)
    llm_scores = [0] * len(papers)
    if settings.openalex_interest_profile:
        for start in range(0, len(papers), BATCH_SIZE):
            batch = papers[start : start + BATCH_SIZE]
            batch_scores = score_batch_with_llm(batch, settings)
            for j, score in enumerate(batch_scores):
                llm_scores[start + j] = score

    # 3. Combine and sort
    combined = [
        (kw + llm, i, paper)
        for i, (paper, kw, llm) in enumerate(zip(papers, keyword_scores, llm_scores))
    ]
    combined.sort(key=lambda x: (-x[0], x[1]))  # highest score first, stable order

    ranked = [paper for _, _, paper in combined[:max_results]]
    logger.info(
        "Ranked %d papers -> top %d (score range: %d-%d).",
        len(papers),
        len(ranked),
        combined[-1][0],
        combined[0][0],
    )
    return ranked
