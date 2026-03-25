# Pipeline Parallelization — Design Spec

**Date:** 2026-03-25

## Problem

The pipeline dry run takes ~50 minutes. The extract/chunk/store phase (~72 min for 200 papers) is 90% of runtime and runs sequentially. Post-processing LLM calls (implications, critiques, ELI5) are independent but also run sequentially (~7 min total).

## Changes

### 1. Parallel extract/chunk/store

Extract the per-paper loop body (pipeline.py lines 182-211) into a helper `_process_paper(paper, settings) -> Paper | None`. Run papers through `concurrent.futures.ThreadPoolExecutor(max_workers=settings.pipeline_ingest_workers)`.

- Returns the Paper if successfully processed, None if unparseable/skipped.
- On `VectorStoreError`: log error and return None (don't crash the pool — other workers may succeed). After the pool completes, if zero papers were processed, halt the pipeline.
- Warm the chunker cache before the pool by calling `chunk_text("")` to ensure the singleton model is loaded (avoids race on first call).
- Collect results with `executor.map()` or `as_completed()`, filter None values to build `processed_papers`.

**Config:** `pipeline_ingest_workers: int = Field(default=4)` — number of concurrent workers. Set to 1 for sequential behavior.

### 2. Parallel post-processing LLM calls

Run `extract_implications()`, `generate_critiques()`, and `generate_eli5()` concurrently using `concurrent.futures.ThreadPoolExecutor(max_workers=3)`. The summarizer stays sequential (runs first as the baseline). Each post-processor is submitted as a future, results collected after all complete.

Error logging for empty results is applied after collection, same as current behavior.

**Config:** `pipeline_postprocess_parallel: bool = Field(default=True)` — toggle. When False, runs sequentially (current behavior).

### 3. Config additions — `config.py`

```python
# ── Pipeline performance ─────────────────────────────────────
pipeline_ingest_workers: int = Field(default=4)
pipeline_postprocess_parallel: bool = Field(default=True)
```

### 4. `.env.example`

Add new section:
```
# ── Pipeline Performance ─────────────────────────────────────
PIPELINE_INGEST_WORKERS="4"
PIPELINE_POSTPROCESS_PARALLEL="true"
```

### 5. Setup wizard — `setup.py`

Add ingest workers prompt and postprocess parallel toggle to `_collect_optional_settings()`. Add both keys to a new "Pipeline Performance" group in `_write_env_file()`.

### 6. Tests

- Test `_process_paper()` helper: success, unparseable, no-PDF/no-abstract, VectorStoreError.
- Test parallel ingest with workers=2 (verify all papers processed).
- Test parallel post-processing (verify all three called and results collected).
- Test sequential fallback (workers=1, parallel=False).
- Update existing `run()` tests if signatures change.

### 7. Module docstring

Update pipeline.py docstring to reflect parallelized ingest and post-processing.

## Out of Scope

- No changes to the LLM batching logic in `llm_utils.py`
- No changes to the chunker or vectorstore internals
- No async/await migration — `ThreadPoolExecutor` is sufficient for I/O-bound work

## Expected Impact

- Ingest: ~72 min → ~20 min (4 workers)
- Post-processing: ~7 min → ~3 min (bounded by slowest call)
- Total: ~50 min → ~25 min
