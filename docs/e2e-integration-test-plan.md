# Integration & E2E Test Plan

## 1. Executive Summary

The existing test suite (9 test files, ~35 test functions) relies **exclusively** on
`unittest.mock` patches. Every external boundary — arXiv API, PDF downloads,
PyMuPDF, Chonkie, ChromaDB, litellm (LLM), SMTP — is mocked out. Two modules
(`chunker` and `vectorstore`) have **zero test coverage** of any kind.

This document catalogues every unverified code path and proposes a phased plan
to add integration and end-to-end tests that exercise real behaviour.

---

## 2. Current Coverage Audit

### 2.1 Modules With Zero Tests

| Module | Key Functions | Risk |
|---|---|---|
| `chunker.py` | `chunk_text()` | Chonkie `SemanticChunker` is never instantiated; embedding model `all-MiniLM-L6-v2` is never loaded; chunking logic is entirely untested. |
| `vectorstore.py` | `store_chunks()`, `store_unparseable()`, `_get_collection()` | ChromaDB `PersistentClient` is never created; upsert calls, metadata schema, and `VectorStoreError` propagation are all unverified. |

### 2.2 Modules With Mock-Only Tests

| Module | What's Mocked | What's Never Verified |
|---|---|---|
| `fetcher.py` | `arxiv.Client`, `arxiv.Search`, `requests.get` | Real arXiv query construction, real HTTP PDF download, retry backoff timing, actual `Paper` dataclass population from live API responses. |
| `extractor.py` | `fitz.open` | Real PyMuPDF extraction on an actual PDF, multi-page concatenation, handling of scanned/image-only PDFs. |
| `summarizer.py` | `litellm.completion` | Real LLM API call via litellm, rate-limit backoff loop, `max_tokens` enforcement, `None` content handling under real conditions, multi-provider model string routing. |
| `postprocessor.py` | `litellm.completion` | Real LLM API call via litellm, both system prompts with real model, rate-limit retry logic (5 attempts then re-raise). |
| `emailer.py` | `smtplib.SMTP_SSL` | Real SMTP connection, TLS handshake, authentication, actual email delivery. |
| `github_trending.py` | `requests.get` | Real GitHub Search API call, `RequestException` from real network failure, pagination behaviour. |
| `pipeline.py` | All sub-modules | Full end-to-end flow, `main()` CLI entry point, `sys.exit(1)` on `VectorStoreError`, mixed parseable/unparseable papers. |
| `config.py` | N/A (no mocks) | Loading from a real `.env` file, environment variable overrides via `os.environ`. |

---

## 3. Unverified Code Paths — Detailed Breakdown

### 3.1 `chunker.py` — ZERO coverage

| # | Code Path | Lines | Description |
|---|---|---|---|
| C-1 | `chunk_text()` happy path | 27-44 | Instantiate `SemanticChunker` with `all-MiniLM-L6-v2`, chunk a multi-paragraph text, verify `TextChunk` list returned. |
| C-2 | Empty input | 27-44 | Pass empty string `""` — verify graceful return (empty list or single empty chunk). |
| C-3 | Very short input | 27-44 | Single sentence shorter than `chunk_size=512` — verify it returns one chunk. |
| C-4 | Very long input | 27-44 | Multi-page text (>10 KB) — verify multiple chunks are produced and no data loss. |

### 3.2 `vectorstore.py` — ZERO coverage

| # | Code Path | Lines | Description |
|---|---|---|---|
| V-1 | `_get_collection()` happy path | 40-48 | Create a `PersistentClient` with a temp directory, get-or-create a collection. |
| V-2 | `_get_collection()` failure | 40-48 | Pass an invalid/unwritable path — verify `VectorStoreError` is raised with the original exception chained. |
| V-3 | `store_chunks()` happy path | 51-99 | Store 3 chunks for a paper, then query ChromaDB to verify documents, IDs (`<arxiv_id>_chunk_<index>`), and all 6 metadata fields match the schema in EARS §3. |
| V-4 | `store_chunks()` upsert dedup | 51-99 | Call `store_chunks()` twice with the same paper — verify no duplicate documents. |
| V-5 | `store_unparseable()` | 102-121 | Store an unparseable paper, verify the `[unparseable]` document text, `chunk_index=-1`, and `unparseable=True` metadata flag. |
| V-6 | `store_chunks()` empty list | 51-99 | Pass `chunks=[]` — verify no crash and empty `StoredChunk` list returned. |

### 3.3 `fetcher.py` — mock-only coverage

| # | Code Path | Lines | Description |
|---|---|---|---|
| F-1 | `fetch_papers()` real arXiv | 65-110 | Hit the live arXiv API for a narrow topic (e.g. `cs.DL`) with `max_results=2`. Verify `Paper` objects have valid `arxiv_id`, `title`, `authors`, `published`, and that `pdf_path` exists on disk. |
| F-2 | `_download_pdf()` real HTTP | 47-62 | Download an actual small PDF from arXiv. Verify file is non-empty and starts with `%PDF`. |
| F-3 | `_download_pdf()` retry timing | 47-62 | Point at a failing URL with `max_retries=2`. Verify the function returns `False` and that elapsed time indicates backoff was applied (~2s + ~4s). |
| F-4 | Query construction | 72 | Verify that `" OR ".join(...)` with multiple topics produces the correct arXiv query string. |

### 3.4 `extractor.py` — mock-only coverage

| # | Code Path | Lines | Description |
|---|---|---|---|
| E-1 | Real PDF extraction | 27-46 | Create a real PDF (via `reportlab` or a fixture file), pass to `extract_text()`, verify `parseable=True` and text content matches. |
| E-2 | Multi-page PDF | 27-46 | Create a 3-page PDF, verify extracted text contains content from all pages joined by `\n`. |
| E-3 | Corrupt/empty PDF | 27-46 | Pass a file containing `b"not a pdf"` — verify `parseable=False`. |
| E-4 | Image-only PDF | 27-46 | If feasible, pass a scanned-image PDF — verify `parseable=False` (empty text). |

### 3.5 `summarizer.py` — mock-only coverage

| # | Code Path | Lines | Description |
|---|---|---|---|
| S-1 | Real LLM call via litellm | 55-82 | Against a local/stub OpenAI-compatible server, call `litellm.completion()` with `settings.llm_model` and verify a non-empty string is returned. |
| S-2 | Rate-limit backoff | 59-80 | Stub server returns 429 twice then 200 — verify the function retries and eventually succeeds (`litellm.RateLimitError`). |
| S-3 | Rate-limit exhaustion | 59-80 | Stub server returns 429 for all 5 attempts — verify `litellm.RateLimitError` is re-raised. |
| S-4 | `max_tokens` enforcement | 61 | Verify the `max_tokens` kwarg passed to `litellm.completion()` matches `settings.llm_max_tokens`. |
| S-5 | `None` content response | 69 | Stub server returns `choices[0].message.content = null` — verify empty string `""` returned. |
| S-6 | Multi-provider model routing | 55-82 | Verify `litellm.completion()` receives the full `provider/model` string (e.g. `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-20250514`) and `api_base` is passed correctly. |

### 3.6 `postprocessor.py` — mock-only coverage

| # | Code Path | Lines | Description |
|---|---|---|---|
| P-1 | `_llm_call()` real request via litellm | 62-89 | Same as S-1 but verifying both `IMPLICATIONS_SYSTEM_PROMPT` and `CRITIQUES_SYSTEM_PROMPT` are sent correctly via `litellm.completion()`. |
| P-2 | Rate-limit backoff | 66-87 | Same pattern as S-2/S-3 for the shared `_llm_call()` retry loop using `litellm.RateLimitError`. |

### 3.7 `emailer.py` — partially verified

| # | Code Path | Lines | Description |
|---|---|---|---|
| EM-1 | Real SMTP send | 140-144 | Against a local SMTP server (e.g. `aiosmtpd`), send a real email and verify it was received with correct headers and body. |
| EM-2 | Jinja2 template edge cases | 24-76 | Render templates with special characters (`<`, `&`, unicode), empty `implications`/`critiques`, and very long summaries. |
| EM-3 | SSL/TLS connection | 141 | Verify `SMTP_SSL` connects with TLS to a local TLS-enabled SMTP stub. |

### 3.8 `github_trending.py` — mock-only coverage

| # | Code Path | Lines | Description |
|---|---|---|---|
| G-1 | Real GitHub API call | 34-71 | Hit the live GitHub Search API (rate limits permitting) and verify `TrendingRepo` objects are returned with valid fields. |
| G-2 | `RequestException` handling | 53-57 | Trigger a real network error (e.g. unreachable host) and verify an empty list is returned without raising. |

### 3.9 `pipeline.py` — mock-only coverage

| # | Code Path | Lines | Description |
|---|---|---|---|
| PL-1 | Full E2E dry-run | 34-103 | Run the entire pipeline end-to-end in dry-run mode with a real (or fixture) PDF, real ChromaDB, and a stubbed LLM server. Verify console output contains the summary. |
| PL-2 | Mixed parseable/unparseable | 48-73 | Feed 2 papers (1 valid PDF, 1 corrupt) and verify only the valid one reaches the summary. |
| PL-3 | `VectorStoreError` → `sys.exit(1)` | 54-67 | Make ChromaDB unreachable mid-pipeline and verify the process exits with code 1. |
| PL-4 | `main()` CLI args | 106-142 | Invoke `main()` with `--dry-run`, `--topics cs.CL`, and `-v` flags. Verify settings are overridden correctly. |
| PL-5 | GitHub-enabled E2E | 76-79 | Run with `github_enabled=True` and verify the GitHub section appears in the final email output. |
| PL-6 | All-unparseable early exit | 71-73 | Feed only unparseable papers and verify the pipeline logs and returns without calling `summarize`. |

### 3.10 `config.py` — minimal coverage

| # | Code Path | Lines | Description |
|---|---|---|---|
| CFG-1 | `.env` file loading | 18-22 | Create a temp `.env` file with overrides, point `Settings` at it, verify values are read. |
| CFG-2 | Environment variable override | 11-60 | Set `os.environ["ARXIV_MAX_RESULTS"] = "10"`, create `Settings`, verify `arxiv_max_results == 10`. |
| CFG-3 | List parsing | 25 | Set `ARXIV_TOPICS=cs.AI,cs.CL,stat.ML` via env, verify the list is `["cs.AI", "cs.CL", "stat.ML"]`. |
| CFG-4 | litellm model string | 30 | Set `LLM_MODEL=anthropic/claude-sonnet-4-20250514` via env, verify `settings.llm_model` matches. |
| CFG-5 | `llm_api_base` nullable | 32 | Verify `llm_api_base` defaults to `None` when unset, and accepts a URL string when set. |

---

## 4. Test Infrastructure Requirements

### 4.1 New Dependencies (dev only)

```toml
[project.optional-dependencies]
dev = [
    # ... existing ...
    "pytest-timeout>=2.2",       # guard against hanging integration tests
    "aiosmtpd>=1.4",             # local SMTP server for email integration tests
    "reportlab>=4.0",            # generate fixture PDFs programmatically
    "respx>=0.21",               # or responses>=0.25 for HTTP stubbing
]
```

### 4.2 Fixture Files

| Fixture | Location | Purpose |
|---|---|---|
| `tests/fixtures/sample.pdf` | Static | A 2-page text-based PDF for extractor integration tests. |
| `tests/fixtures/empty.pdf` | Static | A valid PDF with no extractable text (image-only). |
| `tests/fixtures/corrupt.bin` | Static | A non-PDF file for error handling tests. |

### 4.3 Stub LLM Server

For summarizer/postprocessor integration tests without hitting a real LLM:

- Use a lightweight ASGI app (e.g. `fastapi` or plain `http.server`) that serves
  OpenAI-compatible `/v1/chat/completions` responses.
- Configurable to return 429 (rate limit) for backoff tests.
- Point `settings.llm_api_base` at `http://localhost:<port>/v1` and use
  `settings.llm_model = "openai/test-model"` (litellm routes via the provider prefix).
- litellm's `completion()` function accepts `api_base` to override the endpoint,
  making it straightforward to redirect all LLM calls to the stub.

### 4.4 Test Directory Structure

```
tests/
├── unit/                          # Existing mock-based tests (move here)
│   ├── test_arxiv_topics.py
│   ├── test_config.py
│   ├── test_emailer.py
│   ├── test_extractor.py
│   ├── test_fetcher.py
│   ├── test_github_trending.py
│   ├── test_pipeline.py
│   ├── test_postprocessor.py
│   └── test_summarizer.py
├── integration/                   # New: real dependencies, stubbed externals
│   ├── conftest.py               # Shared fixtures (settings, tmp dirs, stub server)
│   ├── test_chunker_integration.py
│   ├── test_vectorstore_integration.py
│   ├── test_extractor_integration.py
│   ├── test_emailer_integration.py
│   ├── test_summarizer_integration.py
│   └── test_config_integration.py
├── e2e/                           # New: full pipeline flows
│   ├── conftest.py               # Pipeline-level fixtures
│   ├── test_pipeline_e2e.py
│   └── test_cli_e2e.py
├── fixtures/
│   ├── sample.pdf
│   ├── empty.pdf
│   └── corrupt.bin
└── conftest.py                    # Shared pytest configuration
```

### 4.5 Pytest Markers

```python
# conftest.py
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, no external dependencies")
    config.addinivalue_line("markers", "integration: real local dependencies (ChromaDB, Chonkie, PyMuPDF)")
    config.addinivalue_line("markers", "e2e: full pipeline runs")
    config.addinivalue_line("markers", "network: requires internet access (arXiv, GitHub)")
```

```ini
# pyproject.toml addition
[tool.pytest.ini_options]
markers = [
    "unit: fast, no external dependencies",
    "integration: real local dependencies",
    "e2e: full pipeline runs",
    "network: requires internet access",
]
```

---

## 5. Implementation Plan — Phased Rollout

### Phase 1: Fill Zero-Coverage Gaps (Priority: Critical)

**Goal:** Get `chunker` and `vectorstore` under test with real dependencies.

| Test ID | Module | Test | Marker |
|---|---|---|---|
| C-1 | chunker | `chunk_text()` happy path with real Chonkie + embedding model | `integration` |
| C-2 | chunker | Empty string input | `integration` |
| C-3 | chunker | Short single-sentence input | `integration` |
| C-4 | chunker | Long multi-page input | `integration` |
| V-1 | vectorstore | `_get_collection()` with real ChromaDB temp dir | `integration` |
| V-2 | vectorstore | `_get_collection()` failure path | `integration` |
| V-3 | vectorstore | `store_chunks()` — write + read-back verification | `integration` |
| V-4 | vectorstore | `store_chunks()` upsert idempotency | `integration` |
| V-5 | vectorstore | `store_unparseable()` metadata verification | `integration` |
| V-6 | vectorstore | `store_chunks()` with empty chunk list | `integration` |

### Phase 2: Replace Mocks With Real Local Dependencies (Priority: High)

**Goal:** Verify extractor, emailer, and config against real local resources.

| Test ID | Module | Test | Marker |
|---|---|---|---|
| E-1 | extractor | Real PDF → text extraction | `integration` |
| E-2 | extractor | Multi-page PDF | `integration` |
| E-3 | extractor | Corrupt file → `parseable=False` | `integration` |
| EM-1 | emailer | Real SMTP send via `aiosmtpd` | `integration` |
| EM-2 | emailer | Jinja2 template edge cases | `integration` |
| CFG-1 | config | `.env` file loading | `integration` |
| CFG-2 | config | `os.environ` override | `integration` |
| CFG-3 | config | List parsing from env | `integration` |

### Phase 3: Stub-Server Tests for LLM Modules (Priority: High)

**Goal:** Verify summarizer and postprocessor via litellm against a local OpenAI-compatible stub.

| Test ID | Module | Test | Marker |
|---|---|---|---|
| S-1 | summarizer | Successful completion via stub server | `integration` |
| S-2 | summarizer | Rate-limit retry (429 → 200) | `integration` |
| S-3 | summarizer | Rate-limit exhaustion (5× 429 → re-raise) | `integration` |
| S-5 | summarizer | `null` content → empty string | `integration` |
| S-6 | summarizer | Multi-provider model routing via litellm | `integration` |
| P-1 | postprocessor | Both prompts via litellm stub server | `integration` |
| P-2 | postprocessor | Rate-limit retry via litellm stub server | `integration` |

### Phase 4: End-to-End Pipeline Tests (Priority: High)

**Goal:** Run the full pipeline with real local dependencies and stubbed external APIs.

| Test ID | Test | Marker |
|---|---|---|
| PL-1 | Full dry-run E2E: fixture PDF → Chonkie → ChromaDB → litellm (stub server) → console output | `e2e` |
| PL-2 | Mixed parseable + unparseable papers | `e2e` |
| PL-3 | `VectorStoreError` causes `sys.exit(1)` | `e2e` |
| PL-4 | CLI `main()` with `--dry-run --topics cs.CL -v` | `e2e` |
| PL-5 | GitHub-enabled pipeline (stubbed GitHub API) | `e2e` |
| PL-6 | All-unparseable papers → early exit without summarization | `e2e` |

### Phase 5: Optional Network Tests (Priority: Low)

**Goal:** Smoke tests that hit real external APIs. Run only in CI with network access.

| Test ID | Module | Test | Marker |
|---|---|---|---|
| F-1 | fetcher | Live arXiv API query | `network` |
| F-2 | fetcher | Download a real PDF from arXiv | `network` |
| G-1 | github_trending | Live GitHub Search API | `network` |

---

## 6. E2E Test Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    E2E Test Harness                          │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌────────────────────┐    │
│  │ Fixture  │───▶│ extract  │───▶│  chunk_text()      │    │
│  │ PDFs     │    │ _text()  │    │  (real Chonkie)    │    │
│  └──────────┘    └──────────┘    └────────┬───────────┘    │
│                  (real PyMuPDF)            │                 │
│                                           ▼                 │
│                                  ┌────────────────────┐    │
│                                  │  store_chunks()    │    │
│                                  │  (real ChromaDB    │    │
│                                  │   in tmp dir)      │    │
│                                  └────────┬───────────┘    │
│                                           │                 │
│                                           ▼                 │
│                                  ┌────────────────────┐    │
│                                  │  summarize()       │    │
│                                  │  (litellm → stub   │    │
│                                  │   LLM server)      │    │
│                                  └────────┬───────────┘    │
│                                           │                 │
│                                           ▼                 │
│                                  ┌────────────────────┐    │
│                                  │  send_digest()     │    │
│                                  │  (dry-run / local  │    │
│                                  │   SMTP stub)       │    │
│                                  └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Key principle:** Real local dependencies (PyMuPDF, Chonkie, ChromaDB), stubbed
remote services (LLM via litellm pointed at a stub server, optionally SMTP).
Network tests are isolated behind the `@pytest.mark.network` marker.

---

## 7. Suggested `conftest.py` Fixtures

```python
# tests/integration/conftest.py

import pytest
from pathlib import Path
from digest_pipeline.config import Settings

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

@pytest.fixture
def sample_pdf() -> Path:
    return FIXTURES_DIR / "sample.pdf"

@pytest.fixture
def corrupt_file() -> Path:
    return FIXTURES_DIR / "corrupt.bin"

@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        llm_api_key="test-key",
        llm_model="openai/test-model",
        llm_api_base="http://localhost:8080/v1",  # stub server
        chroma_persist_dir=tmp_path / "chromadb",
        chroma_collection="test_collection",
        smtp_user="test",
        smtp_password="test",
        email_from="test@test.com",
        email_to="recipient@test.com",
        dry_run=True,
    )
```

---

## 8. CI Configuration Recommendations

```yaml
# Example GitHub Actions job structure
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - run: pytest -m unit --timeout=30

  integration:
    runs-on: ubuntu-latest
    steps:
      - run: pytest -m integration --timeout=120

  e2e:
    runs-on: ubuntu-latest
    steps:
      - run: pytest -m e2e --timeout=300

  network:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'  # nightly only
    steps:
      - run: pytest -m network --timeout=300
```

---

## 9. Success Criteria

| Metric | Current | Target |
|---|---|---|
| Modules with zero test coverage | 2 (`chunker`, `vectorstore`) | 0 |
| Tests using real local dependencies | 0 | 20+ |
| Full E2E pipeline tests | 0 | 6 |
| EARS requirements verified end-to-end | 0 / 15 | 15 / 15 |
| Network smoke tests | 0 | 3 |

---

## 10. EARS Traceability Matrix

Every EARS requirement from the design document mapped to its proposed test:

| EARS ID | Requirement | Existing Test | Proposed Integration/E2E Test |
|---|---|---|---|
| 2.1-1 | PyMuPDF text extraction | Mocked (`test_extractor.py`) | E-1, E-2 |
| 2.1-2 | Chonkie SemanticChunker | **None** | C-1, C-2, C-3, C-4 |
| 2.1-3 | ChromaDB storage with metadata | **None** | V-3, V-4, V-5 |
| 2.1-4 | HTML/Plaintext email format | `test_emailer.py` (partial) | EM-1, EM-2 |
| 2.2-1 | Query arXiv API on trigger | Mocked (`test_fetcher.py`) | F-1, PL-1 |
| 2.2-2 | Filter to 24-hour window | `test_fetcher.py` (pure logic) | F-1 |
| 2.2-3 | Pass abstracts to LLM after storage | Mocked (`test_pipeline.py`) | PL-1 |
| 2.2-4 | Dispatch email after summary | Mocked (`test_emailer.py`) | PL-1, EM-1 |
| 2.3-1 | Max token limit on LLM | Mocked (`test_postprocessor.py`) | S-1, S-4 |
| 2.3-2 | SSL/TLS for SMTP | Mocked (`test_emailer.py`) | EM-3 |
| 2.3-3 | Dry-run prints to console | `test_emailer.py` (real) | PL-1, PL-4 |
| 2.4-1 | Skip paper after 3 PDF failures | Mocked (`test_fetcher.py`) | F-3 |
| 2.4-2 | Flag unparseable documents | Mocked (`test_extractor.py`, `test_pipeline.py`) | E-3, V-5, PL-2 |
| 2.4-3 | Halt on ChromaDB failure | Mocked (`test_pipeline.py`) | V-2, PL-3 |
| 2.4-4 | LLM rate-limit backoff (litellm) | **None** | S-2, S-3, P-2 |
| 2.5-1 | GitHub trending query | Mocked (`test_github_trending.py`) | G-1 |
| 2.5-2 | Append repos to LLM prompt | `test_summarizer.py` (pure logic) | PL-5 |
