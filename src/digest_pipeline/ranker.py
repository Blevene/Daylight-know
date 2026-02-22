"""Interest-based paper ranking with keyword boosting and LLM scoring.

Scores papers against a user-defined interest profile to select the most
relevant papers from a larger fetch pool. Used between OpenAlex fetching
and the main digest pipeline.
"""

from __future__ import annotations

import logging

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)

KEYWORD_BOOST = 2


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
