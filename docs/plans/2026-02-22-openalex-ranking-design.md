# OpenAlex Hybrid Fetch-Rank-Filter

**Date:** 2026-02-22
**Status:** Approved

## Problem

OpenAlex results are either too narrow (search term cuts 95% of papers) or too broad (field-level filtering includes irrelevant work). No quality or relevance ranking exists — every paper that has a title and abstract gets included.

## Solution

Fetch a wide pool of papers using subfield filters (no search term), then rank them against a user-defined interest profile using LLM batch scoring + keyword boosting. Keep only the top N.

## Flow

```
OpenAlex API (100 papers, subfield-filtered, sorted by date)
    |
ranker.rank_papers(papers, settings)
    |
  1. Keyword boost: +2 per keyword match in title/abstract
  2. LLM batch scoring: batches of 20, score 1-10 against interest profile
  3. Combine: LLM score + keyword boost, sort descending
    |
Top 20 papers -> existing pipeline (summarize -> implications -> critiques)
```

## New Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `OPENALEX_INTEREST_PROFILE` | str | `""` | Natural language description of research interests |
| `OPENALEX_INTEREST_KEYWORDS` | str | `""` | Comma-separated boost keywords |
| `OPENALEX_FETCH_POOL` | int | `100` | How many papers to fetch before ranking |

`OPENALEX_MAX_RESULTS` keeps its existing meaning: how many papers survive ranking into the digest.

## New Module: `src/digest_pipeline/ranker.py`

### `rank_papers(papers, settings) -> list[Paper]`

1. If no interest profile and no keywords configured, return papers unchanged (backward compatible).
2. Compute keyword boost score per paper: +2 per keyword found in title or abstract (case-insensitive).
3. LLM batch scoring: send batches of 20 papers to the configured `LLM_MODEL` with a prompt containing the interest profile and paper titles/abstracts. Returns 1-10 relevance scores.
4. Final score = LLM score + keyword boost. Sort descending, return top `OPENALEX_MAX_RESULTS`.
5. **Graceful degradation:** if LLM scoring fails, fall back to keyword-only ranking.

### Scoring Prompt

System prompt instructs the LLM to rate each paper's relevance (1-10) against the provided research interest profile. Returns structured JSON `{"paper_1": 7, "paper_2": 3, ...}`.

## Changes to Existing Code

- **`openalex_fetcher.py`**: Remove `search` parameter from API call. Use `OPENALEX_FETCH_POOL` as `per_page` instead of `OPENALEX_MAX_RESULTS`.
- **`pipeline.py`**: After `fetch_openalex_papers()`, call `rank_papers()` before extending the main papers list.
- **`config.py`**: Add `openalex_interest_profile`, `openalex_interest_keywords`, `openalex_fetch_pool` fields.
- **`.env.example`**: Document new fields.
- **`setup.py`**: Add interest profile and keywords prompts to the setup wizard.

## Cost Impact

~5 extra LLM calls per run with short abstracts (title + abstract only, not full papers). Uses the already-configured `LLM_MODEL` via litellm.

## Graceful Degradation

If the LLM scoring step fails (API error, timeout), the ranker falls back to keyword-only scoring. If neither profile nor keywords are configured, ranking is skipped entirely — full backward compatibility.
