# Pipeline Parallelization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the extract/chunk/store ingest phase and post-processing LLM calls to reduce pipeline runtime from ~83 min to ~25-35 min.

**Architecture:** ThreadPoolExecutor for both parallelization targets. Ingest workers share a single ChromaDB collection handle. Post-processors run concurrently via submit()+as_completed(). Both are configurable with sensible defaults and sequential fallbacks.

**Tech Stack:** Python concurrent.futures, threading, ChromaDB, litellm, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-pipeline-parallelization-design.md`

---

### Task 1: Add threading.Lock to chunker singleton

**Files:**
- Modify: `src/digest_pipeline/chunker.py:22-38`
- Test: `tests/unit/test_chunker.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_chunker.py`:

```python
"""Tests for chunker thread safety."""

import threading
from unittest.mock import patch, MagicMock

from digest_pipeline.chunker import _get_chunker


@patch("digest_pipeline.chunker.StaticModel.from_pretrained")
@patch("digest_pipeline.chunker.SemanticChunker")
def test_get_chunker_concurrent_calls_create_single_instance(
    mock_chunker_cls, mock_model,
):
    """Multiple threads calling _get_chunker() should create only one instance."""
    import digest_pipeline.chunker as mod
    mod._chunker = None  # reset singleton

    mock_model.return_value = MagicMock()
    mock_instance = MagicMock()
    mock_chunker_cls.return_value = mock_instance

    results = []
    def call_get_chunker():
        results.append(_get_chunker())

    threads = [threading.Thread(target=call_get_chunker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 10
    # All threads should get the same instance
    assert all(r is results[0] for r in results)
    # Model should only be loaded once
    mock_model.assert_called_once()

    mod._chunker = None  # cleanup
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_chunker.py -v`
Expected: FAIL (race condition — `mock_model` called multiple times)

- [ ] **Step 3: Add threading.Lock to `_get_chunker()`**

In `src/digest_pipeline/chunker.py`, add `import threading` and replace the singleton pattern:

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
            static_model = StaticModel.from_pretrained(_EMBEDDING_MODEL, force_download=False)
            embeddings = Model2VecEmbeddings(model=static_model)
            _chunker = SemanticChunker(
                embedding_model=embeddings,
                threshold=0.8,
                chunk_size=2048,
                similarity_window=3,
                skip_window=0,
            )
    return _chunker
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/digest_pipeline/chunker.py tests/unit/test_chunker.py
git commit -m "fix: add threading.Lock to chunker singleton

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add optional collection parameter to vectorstore functions

**Files:**
- Modify: `src/digest_pipeline/vectorstore.py:54-68,112-114`
- Test: `tests/unit/test_vectorstore.py` (if exists, else new)

- [ ] **Step 1: Write the failing test**

Create or add to `tests/unit/test_vectorstore.py`:

```python
"""Tests for vectorstore collection parameter passthrough."""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from digest_pipeline.chunker import TextChunk
from digest_pipeline.fetcher import Paper
from digest_pipeline.vectorstore import store_chunks, store_unparseable


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        paper_id="2401.00001",
        title="Test",
        authors=["Alice"],
        abstract="Abstract.",
        url="https://arxiv.org/abs/2401.00001",
        published=datetime(2025, 1, 15, tzinfo=timezone.utc),
        source="arxiv",
        pdf_path=None,
    )
    defaults.update(overrides)
    return Paper(**defaults)


@patch("digest_pipeline.vectorstore._get_collection")
def test_store_chunks_uses_passed_collection(mock_get_coll, make_settings):
    """When a collection is passed, _get_collection should NOT be called."""
    mock_coll = MagicMock()
    chunks = [TextChunk(text="hello", chunk_index=0)]

    store_chunks(_make_paper(), chunks, make_settings(), collection=mock_coll)

    mock_get_coll.assert_not_called()
    mock_coll.upsert.assert_called_once()


@patch("digest_pipeline.vectorstore._get_collection")
def test_store_chunks_falls_back_to_get_collection(mock_get_coll, make_settings):
    """Without a collection param, _get_collection is called as before."""
    mock_coll = MagicMock()
    mock_get_coll.return_value = mock_coll
    chunks = [TextChunk(text="hello", chunk_index=0)]

    store_chunks(_make_paper(), chunks, make_settings())

    mock_get_coll.assert_called_once()
    mock_coll.upsert.assert_called_once()


@patch("digest_pipeline.vectorstore._get_collection")
def test_store_unparseable_uses_passed_collection(mock_get_coll, make_settings):
    """When a collection is passed, _get_collection should NOT be called."""
    mock_coll = MagicMock()

    store_unparseable(_make_paper(), make_settings(), collection=mock_coll)

    mock_get_coll.assert_not_called()
    mock_coll.upsert.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_vectorstore.py -v`
Expected: FAIL (`store_chunks()` does not accept `collection` kwarg)

- [ ] **Step 3: Add optional collection parameter**

In `src/digest_pipeline/vectorstore.py`, update both functions:

For `store_chunks` (line 54-58), change signature to:
```python
def store_chunks(
    paper: Paper,
    chunks: list[TextChunk],
    settings: Settings,
    *,
    collection: chromadb.Collection | None = None,
) -> list[StoredChunk]:
```

And change line 68 from:
```python
    collection = _get_collection(settings)
```
to:
```python
    if collection is None:
        collection = _get_collection(settings)
```

For `store_unparseable` (line 112), change signature to:
```python
def store_unparseable(paper: Paper, settings: Settings, *, collection: chromadb.Collection | None = None) -> None:
```

And add before line 114:
```python
    if collection is None:
        collection = _get_collection(settings)
```
Remove the existing `collection = _get_collection(settings)` on current line 114.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_vectorstore.py -v`
Expected: PASS

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All pass (existing callers don't pass `collection`, so default `None` triggers fallback)

- [ ] **Step 6: Commit**

```bash
git add src/digest_pipeline/vectorstore.py tests/unit/test_vectorstore.py
git commit -m "feat: add optional collection parameter to vectorstore functions

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add config fields

**Files:**
- Modify: `src/digest_pipeline/config.py:73-74`

- [ ] **Step 1: Add config fields**

After the post-processing section (line 74), add:

```python
    # ── Pipeline performance ─────────────────────────────────────
    pipeline_ingest_workers: int = Field(default=4)
    pipeline_postprocess_parallel: bool = Field(default=True)
```

- [ ] **Step 2: Verify**

Run: `python3 -c "import sys; sys.path.insert(0,'src'); from digest_pipeline.config import Settings; s=Settings(_env_file=None,llm_api_key='x'); print(s.pipeline_ingest_workers, s.pipeline_postprocess_parallel)"`
Expected: `4 True`

- [ ] **Step 3: Commit**

```bash
git add src/digest_pipeline/config.py
git commit -m "feat: add pipeline performance config fields

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Extract `_process_paper()` helper and `_ingest_papers()`

**Files:**
- Modify: `src/digest_pipeline/pipeline.py:16-45,178-224`
- Test: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Write tests for `_process_paper()`**

Add to `tests/unit/test_pipeline.py`. First update imports:

```python
from digest_pipeline.pipeline import PaperAnalysis, _build_analyses, _process_paper, run
```

Then add tests:

```python
@patch("digest_pipeline.pipeline.store_chunks")
@patch("digest_pipeline.pipeline.chunk_text", return_value=[])
@patch(
    "digest_pipeline.pipeline.extract_text",
    return_value=ExtractionResult(paper_id="2401.00001", text="content", parseable=True),
)
def test_process_paper_success(mock_extract, mock_chunk, mock_store):
    paper = _make_paper()
    mock_coll = MagicMock()
    settings = Settings(_env_file=None, llm_api_key="x")

    result = _process_paper(paper, mock_coll, settings)

    assert result is paper
    mock_extract.assert_called_once()
    mock_chunk.assert_called_once_with("content")
    mock_store.assert_called_once()


@patch("digest_pipeline.pipeline.store_unparseable")
@patch(
    "digest_pipeline.pipeline.extract_text",
    return_value=ExtractionResult(paper_id="2401.00001", text="", parseable=False),
)
def test_process_paper_unparseable(mock_extract, mock_store_unparse):
    paper = _make_paper()
    mock_coll = MagicMock()
    settings = Settings(_env_file=None, llm_api_key="x")

    result = _process_paper(paper, mock_coll, settings)

    assert result is None
    mock_store_unparse.assert_called_once()


def test_process_paper_no_pdf_no_abstract():
    paper = _make_paper(pdf_path=None, abstract="")
    mock_coll = MagicMock()
    settings = Settings(_env_file=None, llm_api_key="x")

    result = _process_paper(paper, mock_coll, settings)

    assert result is None


@patch("digest_pipeline.pipeline.store_chunks", side_effect=VectorStoreError("down"))
@patch("digest_pipeline.pipeline.chunk_text", return_value=[])
@patch(
    "digest_pipeline.pipeline.extract_text",
    return_value=ExtractionResult(paper_id="2401.00001", text="content", parseable=True),
)
def test_process_paper_vectorstore_error_returns_none(mock_extract, mock_chunk, mock_store):
    paper = _make_paper()
    mock_coll = MagicMock()
    settings = Settings(_env_file=None, llm_api_key="x")

    result = _process_paper(paper, mock_coll, settings)

    assert result is None
```

Also add tests for `_ingest_papers()`:

```python
from digest_pipeline.pipeline import _ingest_papers


@patch("digest_pipeline.pipeline._process_paper")
@patch("digest_pipeline.pipeline._get_collection")
@patch("digest_pipeline.pipeline.chunk_text")
def test_ingest_papers_parallel(mock_chunk, mock_get_coll, mock_process, make_settings):
    """Parallel ingest with workers=2 processes all papers."""
    papers = [_make_paper(paper_id=f"p{i}") for i in range(4)]
    mock_get_coll.return_value = MagicMock()
    mock_process.side_effect = lambda p, c, s: p  # return each paper

    result = _ingest_papers(papers, make_settings(pipeline_ingest_workers=2))

    assert len(result) == 4
    assert mock_process.call_count == 4


@patch("digest_pipeline.pipeline._process_paper")
@patch("digest_pipeline.pipeline._get_collection")
@patch("digest_pipeline.pipeline.chunk_text")
def test_ingest_papers_sequential_fallback(mock_chunk, mock_get_coll, mock_process, make_settings):
    """workers=1 uses sequential path."""
    papers = [_make_paper(paper_id=f"p{i}") for i in range(3)]
    mock_get_coll.return_value = MagicMock()
    mock_process.side_effect = lambda p, c, s: p

    result = _ingest_papers(papers, make_settings(pipeline_ingest_workers=1))

    assert len(result) == 3


@patch("digest_pipeline.pipeline._process_paper", return_value=None)
@patch("digest_pipeline.pipeline._get_collection")
@patch("digest_pipeline.pipeline.chunk_text")
def test_ingest_papers_fail_fast(mock_chunk, mock_get_coll, mock_process, caplog, make_settings):
    """5 consecutive failures triggers early halt."""
    papers = [_make_paper(paper_id=f"p{i}") for i in range(20)]
    mock_get_coll.return_value = MagicMock()

    with caplog.at_level(logging.CRITICAL, logger="digest_pipeline.pipeline"):
        result = _ingest_papers(papers, make_settings(pipeline_ingest_workers=2))

    assert len(result) == 0
    assert any("5 consecutive" in r.message for r in caplog.records)
```

Also add import for `Settings`, `VectorStoreError`, and `MagicMock` at top of test file:

```python
from unittest.mock import patch, MagicMock
from digest_pipeline.config import Settings
from digest_pipeline.vectorstore import VectorStoreError
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_pipeline.py::test_process_paper_success -v`
Expected: FAIL (no `_process_paper` function)

- [ ] **Step 3: Implement `_process_paper()` and `_ingest_papers()`**

Add imports at top of `pipeline.py`:

```python
import concurrent.futures
import chromadb
```

Also add `_get_collection` to the vectorstore import line (line 45):

```python
from digest_pipeline.vectorstore import VectorStoreError, _get_collection, store_chunks, store_unparseable
```

Add the `_process_paper` helper after `_cleanup_pdf_dirs`:

```python
def _process_paper(
    paper: Paper,
    collection: chromadb.Collection,
    settings: Settings,
) -> Paper | None:
    """Process a single paper: extract text, chunk, store. Thread-safe.

    Returns the Paper on success, None if unparseable/skipped/error.
    """
    if paper.pdf_path is not None:
        extraction = extract_text(paper.pdf_path, paper.paper_id)

        if not extraction.parseable:
            try:
                store_unparseable(paper, settings, collection=collection)
            except VectorStoreError:
                logger.error("ChromaDB store failed for unparseable paper %s.", paper.paper_id)
                return None
            return None

        text = extraction.text
    else:
        text = paper.abstract
        if not text:
            logger.warning("Paper %s has no PDF and no abstract — skipping.", paper.paper_id)
            return None

    chunks = chunk_text(text)

    try:
        store_chunks(paper, chunks, settings, collection=collection)
    except VectorStoreError:
        logger.error("ChromaDB store failed for paper %s.", paper.paper_id)
        return None

    return paper
```

Add `_ingest_papers` helper:

```python
def _ingest_papers(papers: list[Paper], settings: Settings) -> list[Paper]:
    """Extract, chunk, and store all papers. Parallel when workers > 1."""

    # Warm chunker cache before pool starts (avoids race on first call).
    chunk_text("warmup")

    # Create shared ChromaDB collection handle for all workers.
    try:
        collection = _get_collection(settings)
    except VectorStoreError:
        logger.critical("ChromaDB connection failed — halting pipeline.")
        sys.exit(1)

    workers = settings.pipeline_ingest_workers

    if workers <= 1:
        # Sequential fallback
        return [p for paper in papers if (p := _process_paper(paper, collection, settings)) is not None]

    processed: list[Paper] = []
    consecutive_failures = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_paper = {
            executor.submit(_process_paper, paper, collection, settings): paper
            for paper in papers
        }

        for future in concurrent.futures.as_completed(future_to_paper):
            try:
                result = future.result()
            except Exception:
                paper = future_to_paper[future]
                logger.exception("Unexpected error processing paper %s.", paper.paper_id)
                result = None

            if result is not None:
                processed.append(result)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    logger.critical(
                        "5 consecutive paper failures — likely systemic. Cancelling remaining."
                    )
                    for f in future_to_paper:
                        f.cancel()
                    break

    return processed
```

Now replace the ingest loop in `run()` (lines 178-211) with:

```python
    # ── Step 2-4: Extract → Chunk → Store (all papers, parallel) ──
    processed_papers = _ingest_papers(papers, settings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_pipeline.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/digest_pipeline/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: extract _process_paper and _ingest_papers with parallel support

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Extract `_postprocess()` helper with parallel support

**Files:**
- Modify: `src/digest_pipeline/pipeline.py:264-286`
- Test: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Write tests**

Add to `tests/unit/test_pipeline.py`:

```python
from digest_pipeline.pipeline import _postprocess


@patch("digest_pipeline.pipeline.generate_eli5", return_value={"paper_1": "ELI5"})
@patch("digest_pipeline.pipeline.generate_critiques", return_value={"paper_1": "Crit"})
@patch("digest_pipeline.pipeline.extract_implications", return_value={"paper_1": "Impl"})
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "Sum"})
def test_postprocess_parallel(mock_sum, mock_impl, mock_crit, mock_eli5, make_paper, make_settings):
    papers = [make_paper()]
    settings = make_settings(pipeline_postprocess_parallel=True)

    summaries, implications, critiques, eli5 = _postprocess(papers, settings, "")

    assert summaries == {"paper_1": "Sum"}
    assert implications == {"paper_1": "Impl"}
    assert critiques == {"paper_1": "Crit"}
    assert eli5 == {"paper_1": "ELI5"}


@patch("digest_pipeline.pipeline.generate_eli5", return_value={"paper_1": "ELI5"})
@patch("digest_pipeline.pipeline.generate_critiques", return_value={"paper_1": "Crit"})
@patch("digest_pipeline.pipeline.extract_implications", return_value={"paper_1": "Impl"})
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "Sum"})
def test_postprocess_sequential(mock_sum, mock_impl, mock_crit, mock_eli5, make_paper, make_settings):
    papers = [make_paper()]
    settings = make_settings(pipeline_postprocess_parallel=False)

    summaries, implications, critiques, eli5 = _postprocess(papers, settings, "")

    assert summaries == {"paper_1": "Sum"}
    assert implications == {"paper_1": "Impl"}


@patch("digest_pipeline.pipeline.generate_eli5")
@patch("digest_pipeline.pipeline.generate_critiques")
@patch("digest_pipeline.pipeline.extract_implications")
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "Sum"})
def test_postprocess_skips_disabled(mock_sum, mock_impl, mock_crit, mock_eli5, make_paper, make_settings):
    papers = [make_paper()]
    settings = make_settings(
        postprocessing_implications=False,
        postprocessing_critiques=False,
        postprocessing_eli5=False,
    )

    summaries, implications, critiques, eli5 = _postprocess(papers, settings, "")

    mock_impl.assert_not_called()
    mock_crit.assert_not_called()
    mock_eli5.assert_not_called()
    assert implications == {}
    assert critiques == {}
    assert eli5 == {}


@patch("digest_pipeline.pipeline.generate_eli5", side_effect=Exception("LLM down"))
@patch("digest_pipeline.pipeline.generate_critiques", return_value={"paper_1": "Crit"})
@patch("digest_pipeline.pipeline.extract_implications", return_value={"paper_1": "Impl"})
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "Sum"})
def test_postprocess_partial_failure(mock_sum, mock_impl, mock_crit, mock_eli5, make_paper, make_settings):
    """One post-processor failing should not lose the others' results."""
    papers = [make_paper()]
    settings = make_settings(pipeline_postprocess_parallel=True)

    summaries, implications, critiques, eli5 = _postprocess(papers, settings, "")

    assert implications == {"paper_1": "Impl"}
    assert critiques == {"paper_1": "Crit"}
    assert eli5 == {}  # failed, returns empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_pipeline.py::test_postprocess_parallel -v`
Expected: FAIL (no `_postprocess` function)

- [ ] **Step 3: Implement `_postprocess()`**

Add to `pipeline.py`:

```python
def _postprocess(
    papers: list[Paper],
    settings: Settings,
    github_section: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Run summarization then post-processing (parallel or sequential).

    Returns (summaries, implications, critiques, eli5).
    """
    summaries = summarize(papers, settings, github_section=github_section)

    implications: dict[str, str] = {}
    critiques: dict[str, str] = {}
    eli5_results: dict[str, str] = {}

    # Build list of enabled post-processors.
    tasks: list[tuple[str, object]] = []
    if settings.postprocessing_implications:
        tasks.append(("implications", extract_implications))
    if settings.postprocessing_critiques:
        tasks.append(("critiques", generate_critiques))
    if settings.postprocessing_eli5:
        tasks.append(("eli5", generate_eli5))

    if not tasks:
        return summaries, implications, critiques, eli5_results

    if settings.pipeline_postprocess_parallel and len(tasks) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_name = {
                executor.submit(fn, papers, settings): name
                for name, fn in tasks
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                except Exception:
                    logger.exception("Post-processing '%s' failed.", name)
                    result = {}
                if name == "implications":
                    implications = result
                elif name == "critiques":
                    critiques = result
                elif name == "eli5":
                    eli5_results = result
    else:
        # Sequential fallback
        for name, fn in tasks:
            try:
                result = fn(papers, settings)
            except Exception:
                logger.exception("Post-processing '%s' failed.", name)
                result = {}
            if name == "implications":
                implications = result
            elif name == "critiques":
                critiques = result
            elif name == "eli5":
                eli5_results = result

    # Error logging for empty results (preserve existing asymmetry).
    if settings.postprocessing_critiques and not critiques:
        logger.error("Critique generation returned no results for %d papers.", len(papers))
    if settings.postprocessing_eli5 and not eli5_results:
        logger.error("ELI5 generation returned no results for %d papers.", len(papers))

    return summaries, implications, critiques, eli5_results
```

Now replace the post-processing block in `run()` (lines 264-286) and the `_build_analyses` call with:

```python
    # ── Step 7: Post-processing (parallel or sequential) ─────────
    summaries, implications, critiques, eli5_results = _postprocess(
        processed_papers, settings, github_section
    )

    # ── Step 8: Assemble & send ─────────────────────────────────
    analyses = _build_analyses(processed_papers, summaries, implications, critiques, eli5=eli5_results)
```

Also remove the now-unused direct summarize call (line 265).

- [ ] **Step 4: Update existing `run()` tests**

The existing tests that mock `summarize`, `extract_implications`, `generate_critiques`, `generate_eli5` still work because `_postprocess` calls them from the same module. However, tests that assert on `mock_summarize.assert_called_once()` etc. should continue to pass since the mocks are at the module level.

Verify: `python3 -m pytest tests/unit/test_pipeline.py -v`

- [ ] **Step 5: Run all tests**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/digest_pipeline/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: extract _postprocess with parallel LLM call support

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Update .env.example, setup wizard, and module docstring

**Files:**
- Modify: `.env.example`
- Modify: `src/digest_pipeline/setup.py:389-390,532`
- Modify: `src/digest_pipeline/pipeline.py:1-14`

- [ ] **Step 1: Update `.env.example`**

Add before the HuggingFace section:

```
# ── Pipeline Performance ─────────────────────────────────────
PIPELINE_INGEST_WORKERS="4"
PIPELINE_POSTPROCESS_PARALLEL="true"
```

- [ ] **Step 2: Update setup wizard**

In `setup.py`, after the PDF archive prompt in `_collect_optional_settings()` (around line 389), add:

```python
    config["PIPELINE_INGEST_WORKERS"] = _prompt("Parallel ingest workers", "4")
    config["PIPELINE_POSTPROCESS_PARALLEL"] = (
        "true" if _prompt_bool("Parallelize post-processing LLM calls?", default=True) else "false"
    )
```

In `_write_env_file()`, add a new section tuple after "PDF Archive":

```python
        (
            "Pipeline Performance",
            ["PIPELINE_INGEST_WORKERS", "PIPELINE_POSTPROCESS_PARALLEL"],
        ),
```

- [ ] **Step 3: Update module docstring**

Update pipeline.py docstring line 4 to reflect parallelization:

```
  4. Chunk text semantically (Chonkie) and store in ChromaDB (ALL papers, parallel)
```

And line 8:

```
  8. Post-process: ELI5, implications & critiques (per-paper JSON, parallel)
```

- [ ] **Step 4: Commit**

```bash
git add .env.example src/digest_pipeline/setup.py src/digest_pipeline/pipeline.py
git commit -m "feat: add pipeline performance config to .env and setup wizard

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `python3 -m pytest tests/unit/ -v`
Expected: All tests pass

- [ ] **Step 2: Run linting**

Run: `ruff check src/digest_pipeline/pipeline.py src/digest_pipeline/chunker.py src/digest_pipeline/vectorstore.py src/digest_pipeline/config.py`
Expected: No errors

- [ ] **Step 3: Dry-run the pipeline** (optional, requires API key)

Run: `digest-pipeline --dry-run`
Expected: Pipeline completes with parallel ingest and post-processing log messages. Faster than previous run.
