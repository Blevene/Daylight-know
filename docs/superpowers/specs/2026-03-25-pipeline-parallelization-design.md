# Pipeline Parallelization — Design Spec

**Date:** 2026-03-25

## Problem

The pipeline takes ~83 minutes end-to-end (200 papers). The extract/chunk/store phase (~72 min) is ~87% of runtime and runs sequentially. Post-processing LLM calls (implications, critiques, ELI5) are independent but also run sequentially (~8 min total).

## Changes

### 1. Parallel extract/chunk/store

Extract the per-paper loop body (pipeline.py lines 182-211) into a helper `_process_paper(paper, collection, settings) -> Paper | None`. Run papers through `concurrent.futures.ThreadPoolExecutor(max_workers=settings.pipeline_ingest_workers)`.

- Returns the Paper if successfully processed, None if unparseable/skipped.
- Uses `executor.submit()` + `as_completed()` (not `executor.map()`) to allow collecting partial results when individual papers fail.
- On `VectorStoreError` or other exceptions: log error and return None. After the pool completes, if zero papers were processed, halt the pipeline.

**Thread safety precautions:**

- **ChromaDB shared collection:** Create a single `PersistentClient` and collection handle *before* the pool starts via `_get_collection(settings)`. Pass the collection into `_process_paper()` so all workers share it. ChromaDB collection objects are thread-safe for concurrent `upsert()` when sharing a single client (avoids SQLite `database is locked` from multiple `PersistentClient` instances). The `_process_paper` helper calls `collection.upsert()` directly instead of going through `store_chunks()`.
- **Chunker singleton lock:** Add `threading.Lock` to `_get_chunker()` in `chunker.py` to guard the lazy initialization. Additionally, warm the cache before the pool by calling `chunk_text("warmup")` (non-empty string to avoid edge cases). The lock is defense-in-depth.
- **No shared mutable state:** Each paper is dispatched to exactly one worker. Workers return `Paper | None` — results are collected by the main thread. No shared list mutation.

**Fail-fast on systemic failures:** Track consecutive `VectorStoreError` count. If 5 consecutive papers fail with `VectorStoreError`, cancel remaining futures and halt the pipeline (ChromaDB is likely down, not a per-paper issue). Log the systemic failure clearly.

**Config:** `pipeline_ingest_workers: int = Field(default=4)` — number of concurrent workers. Set to 1 for sequential behavior.

### 2. Parallel post-processing LLM calls

Run **only enabled** post-processors concurrently using `concurrent.futures.ThreadPoolExecutor`. Only submit futures for post-processors whose config toggle is True (e.g., skip `generate_critiques` if `postprocessing_critiques=False`). Use `executor.submit()` + individual `future.result()` with try/except so one failure doesn't lose the others.

The summarizer stays sequential (runs first as baseline). Error logging for empty results is applied after collection, preserving the current asymmetry (critiques and ELI5 log errors on empty; implications does not).

**Config:** `pipeline_postprocess_parallel: bool = Field(default=True)` — toggle. When False, runs sequentially (current behavior).

### 3. Extract helper functions from `run()`

To keep `run()` manageable (currently 180 lines), extract two helpers:

- `_ingest_papers(papers, settings) -> list[Paper]` — handles the extract/chunk/store phase (parallel or sequential based on config). Returns processed papers.
- `_postprocess(papers, settings, github_section) -> tuple[dict, dict, dict, dict]` — handles summarization + parallel/sequential post-processing. Returns `(summaries, implications, critiques, eli5)`.

### 4. Chunker thread safety — `chunker.py`

Add `threading.Lock` around the singleton initialization:

```python
import threading

_chunker_lock = threading.Lock()
_chunker: SemanticChunker | None = None

def _get_chunker() -> SemanticChunker:
    global _chunker
    if _chunker is not None:
        return _chunker
    with _chunker_lock:
        if _chunker is None:  # double-checked locking
            static_model = StaticModel.from_pretrained(...)
            embeddings = Model2VecEmbeddings(model=static_model)
            _chunker = SemanticChunker(...)
    return _chunker
```

This is the only change to `chunker.py`.

### 5. Config additions — `config.py`

```python
# ── Pipeline performance ─────────────────────────────────────
pipeline_ingest_workers: int = Field(default=4)
pipeline_postprocess_parallel: bool = Field(default=True)
```

### 6. `.env.example`

Add new section:
```
# ── Pipeline Performance ─────────────────────────────────────
PIPELINE_INGEST_WORKERS="4"
PIPELINE_POSTPROCESS_PARALLEL="true"
```

### 7. Setup wizard — `setup.py`

Add ingest workers prompt and postprocess parallel toggle to `_collect_optional_settings()`. Add both keys to a new "Pipeline Performance" group in `_write_env_file()`.

### 8. Tests

- **`_process_paper()` helper:** success, unparseable, no-PDF/no-abstract, VectorStoreError returns None.
- **Parallel ingest:** workers=2, verify all papers processed and results collected.
- **Fail-fast:** 5 consecutive VectorStoreError triggers early halt.
- **Parallel post-processing:** verify only enabled post-processors are submitted, results collected correctly.
- **Sequential fallback:** workers=1 and parallel=False produce identical results.
- **Post-processing partial failure:** one LLM call raises, other results still collected.
- **Chunker lock:** concurrent calls to `_get_chunker()` produce the same instance.
- **Update existing `run()` tests** for new helper signatures.

### 9. Module docstring

Update pipeline.py docstring to reflect parallelized ingest and post-processing.

## Out of Scope

- No changes to the LLM batching logic in `llm_utils.py`
- No async/await migration — `ThreadPoolExecutor` is sufficient for I/O-bound work

## Expected Impact

- Ingest: ~72 min → ~20-30 min (4 workers, accounting for SQLite write contention)
- Post-processing: ~8 min → ~3 min (bounded by slowest call)
- Total: ~83 min → ~25-35 min
