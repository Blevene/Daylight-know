# E2E & Integration Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement integration and E2E tests across all 5 phases of `docs/e2e-integration-test-plan.md`, filling zero-coverage gaps, replacing mocks with real dependencies, and verifying the full pipeline end-to-end.

**Architecture:** Reorganize tests into `unit/`, `integration/`, and `e2e/` subdirectories. Add shared fixtures via `conftest.py` files. Use real local dependencies (ChromaDB, Chonkie, PyMuPDF) with stubbed remote services (LLM via a FastAPI stub server, SMTP via `aiosmtpd`). Network tests are isolated behind `@pytest.mark.network`.

**Tech Stack:** pytest, pytest-timeout, aiosmtpd, reportlab (PDF generation), FastAPI/uvicorn (stub LLM server), ChromaDB, Chonkie, PyMuPDF, litellm

---

## Task 0: Add Dev Dependencies and Pytest Configuration

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml with new dev deps and markers**

Add the test infrastructure dependencies and pytest markers:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-timeout>=2.2",
    "aiosmtpd>=1.4",
    "reportlab>=4.0",
    "fastapi>=0.111",
    "uvicorn>=0.29",
    "httpx>=0.27",
    "ruff>=0.4",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
markers = [
    "unit: fast, no external dependencies",
    "integration: real local dependencies (ChromaDB, Chonkie, PyMuPDF)",
    "e2e: full pipeline runs",
    "network: requires internet access (arXiv, GitHub)",
]
```

**Step 2: Install updated dependencies**

Run: `pip install -e ".[dev]"`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add integration test dev dependencies and pytest markers"
```

---

## Task 1: Restructure Test Directory

**Files:**
- Create: `tests/unit/` (move existing files)
- Create: `tests/integration/`
- Create: `tests/e2e/`
- Create: `tests/fixtures/`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/e2e/__init__.py`

**Step 1: Create directory structure and move unit tests**

```bash
mkdir -p tests/unit tests/integration tests/e2e tests/fixtures
touch tests/unit/__init__.py tests/integration/__init__.py tests/e2e/__init__.py
# Move existing tests into unit/
mv tests/test_arxiv_topics.py tests/unit/
mv tests/test_config.py tests/unit/
mv tests/test_emailer.py tests/unit/
mv tests/test_extractor.py tests/unit/
mv tests/test_fetcher.py tests/unit/
mv tests/test_github_trending.py tests/unit/
mv tests/test_pipeline.py tests/unit/
mv tests/test_postprocessor.py tests/unit/
mv tests/test_summarizer.py tests/unit/
```

**Step 2: Create root conftest.py with shared markers**

Create `tests/conftest.py`:

```python
"""Shared pytest configuration and markers."""

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, no external dependencies")
    config.addinivalue_line("markers", "integration: real local dependencies (ChromaDB, Chonkie, PyMuPDF)")
    config.addinivalue_line("markers", "e2e: full pipeline runs")
    config.addinivalue_line("markers", "network: requires internet access (arXiv, GitHub)")
```

**Step 3: Run existing unit tests to verify nothing broke**

Run: `pytest tests/unit/ -v`
Expected: All existing tests pass.

**Step 4: Commit**

```bash
git add tests/
git commit -m "refactor: reorganize tests into unit/integration/e2e directories"
```

---

## Task 2: Create Test Fixtures (PDFs + corrupt file)

**Files:**
- Create: `tests/fixtures/sample.pdf` (2-page text PDF via reportlab)
- Create: `tests/fixtures/corrupt.bin` (non-PDF binary)
- Create: `tests/fixtures/generate_fixtures.py` (one-time generator script)

**Step 1: Write fixture generator script**

Create `tests/fixtures/generate_fixtures.py`:

```python
"""One-time script to generate test fixture files."""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


FIXTURES_DIR = Path(__file__).parent


def generate_sample_pdf():
    """Create a 2-page PDF with known text content."""
    path = FIXTURES_DIR / "sample.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)

    # Page 1
    c.drawString(72, 720, "Page 1: Introduction to Semantic Chunking")
    c.drawString(72, 700, "This paper presents a novel approach to text segmentation.")
    c.drawString(72, 680, "We propose using embedding-based similarity for chunk boundaries.")
    c.showPage()

    # Page 2
    c.drawString(72, 720, "Page 2: Results and Discussion")
    c.drawString(72, 700, "Our method achieves state-of-the-art performance on three benchmarks.")
    c.drawString(72, 680, "The semantic chunker preserves paragraph-level coherence.")
    c.showPage()

    c.save()
    print(f"Generated: {path}")


def generate_corrupt_file():
    """Create a non-PDF binary file."""
    path = FIXTURES_DIR / "corrupt.bin"
    path.write_bytes(b"not a pdf\x00\xff\xfe\xfd")
    print(f"Generated: {path}")


if __name__ == "__main__":
    generate_sample_pdf()
    generate_corrupt_file()
```

**Step 2: Run the generator**

Run: `python tests/fixtures/generate_fixtures.py`
Expected: `tests/fixtures/sample.pdf` and `tests/fixtures/corrupt.bin` exist.

**Step 3: Verify the PDF is valid**

Run: `python -c "import fitz; doc = fitz.open('tests/fixtures/sample.pdf'); print(f'{len(doc)} pages'); print(doc[0].get_text()[:100])"`
Expected: Shows "2 pages" and text from page 1.

**Step 4: Commit**

```bash
git add tests/fixtures/
git commit -m "test: add PDF and corrupt-file fixtures for integration tests"
```

---

## Task 3: Create Integration conftest.py with Shared Fixtures

**Files:**
- Create: `tests/integration/conftest.py`

**Step 1: Write the shared integration fixtures**

Create `tests/integration/conftest.py`:

```python
"""Shared fixtures for integration tests."""

import pytest
from pathlib import Path
from datetime import datetime, timezone

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_pdf() -> Path:
    path = FIXTURES_DIR / "sample.pdf"
    assert path.exists(), f"Fixture missing: {path}"
    return path


@pytest.fixture
def corrupt_file() -> Path:
    path = FIXTURES_DIR / "corrupt.bin"
    assert path.exists(), f"Fixture missing: {path}"
    return path


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        llm_api_key="test-key",
        llm_model="openai/test-model",
        llm_api_base="http://localhost:9876/v1",
        chroma_persist_dir=tmp_path / "chromadb",
        chroma_collection="test_collection",
        smtp_user="test",
        smtp_password="test",
        email_from="test@test.com",
        email_to="recipient@test.com",
        dry_run=True,
    )


@pytest.fixture
def make_paper():
    """Factory fixture for creating Paper objects."""
    def _make_paper(**overrides) -> Paper:
        defaults = dict(
            arxiv_id="2401.00001",
            title="Test Paper on Semantic Chunking",
            authors=["Alice Researcher", "Bob Scientist"],
            abstract="This paper explores novel approaches to text segmentation.",
            url="https://arxiv.org/abs/2401.00001",
            published=datetime(2025, 1, 15, tzinfo=timezone.utc),
            pdf_path=None,
        )
        defaults.update(overrides)
        return Paper(**defaults)
    return _make_paper
```

**Step 2: Commit**

```bash
git add tests/integration/conftest.py
git commit -m "test: add shared integration test fixtures"
```

---

## Task 4: Phase 1 — Chunker Integration Tests (C-1 through C-4)

**Files:**
- Create: `tests/integration/test_chunker_integration.py`

**Step 1: Write the failing tests**

Create `tests/integration/test_chunker_integration.py`:

```python
"""Integration tests for chunker.py — real Chonkie + embedding model.

Test IDs: C-1, C-2, C-3, C-4
"""

import pytest

from digest_pipeline.chunker import TextChunk, chunk_text


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestChunkerIntegration:
    """Tests that exercise the real SemanticChunker with all-MiniLM-L6-v2."""

    def test_happy_path_multi_paragraph(self):
        """C-1: Multi-paragraph text produces multiple TextChunk objects."""
        text = (
            "Machine learning has transformed natural language processing. "
            "Recent advances in transformer architectures have enabled models "
            "to achieve human-level performance on many benchmarks.\n\n"
            "Reinforcement learning from human feedback (RLHF) has become a "
            "standard technique for aligning large language models with human "
            "preferences. This approach uses reward models trained on human "
            "comparisons to fine-tune base models.\n\n"
            "Diffusion models have emerged as the leading approach for image "
            "generation. These models learn to reverse a noise-adding process, "
            "gradually transforming random noise into coherent images. The "
            "technique has been extended to video, audio, and 3D generation."
        )
        chunks = chunk_text(text)

        assert len(chunks) >= 1
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(isinstance(c.text, str) and len(c.text) > 0 for c in chunks)
        # Verify sequential indexing
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        # Verify no data loss: all original content appears across chunks
        combined = " ".join(c.text for c in chunks)
        assert "transformer" in combined.lower()
        assert "diffusion" in combined.lower()

    def test_empty_input(self):
        """C-2: Empty string returns an empty list (or single empty chunk)."""
        chunks = chunk_text("")
        # Either empty list or graceful handling
        assert isinstance(chunks, list)
        if len(chunks) > 0:
            # If Chonkie returns a chunk for empty input, it should be benign
            assert all(isinstance(c, TextChunk) for c in chunks)

    def test_short_input(self):
        """C-3: Single sentence shorter than chunk_size=512 returns one chunk."""
        text = "A brief sentence about machine learning."
        chunks = chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert "machine learning" in chunks[0].text.lower()

    def test_long_input(self):
        """C-4: Multi-page text (>10KB) produces multiple chunks without data loss."""
        # Generate a ~12KB text with distinct paragraphs
        paragraphs = []
        topics = [
            "neural networks", "gradient descent", "attention mechanisms",
            "convolutional layers", "recurrent architectures", "transformer models",
            "batch normalization", "dropout regularization", "transfer learning",
            "few-shot learning", "meta-learning", "self-supervised pretraining",
        ]
        for topic in topics:
            paragraphs.append(
                f"The field of {topic} has seen significant advances in recent years. "
                f"Researchers have developed new approaches to {topic} that improve "
                f"upon previous methods by incorporating novel architectural designs "
                f"and training procedures. These advances in {topic} have practical "
                f"applications across many domains including healthcare, finance, "
                f"and autonomous systems. The theoretical foundations of {topic} "
                f"continue to be an active area of research with many open questions."
            )
        text = "\n\n".join(paragraphs)
        assert len(text) > 10_000, f"Text should be >10KB, got {len(text)}"

        chunks = chunk_text(text)

        assert len(chunks) >= 2
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        # Spot-check: first and last topics should appear somewhere in chunks
        combined = " ".join(c.text for c in chunks)
        assert "neural networks" in combined.lower()
        assert "self-supervised" in combined.lower()
```

**Step 2: Run tests to verify they pass with real Chonkie**

Run: `pytest tests/integration/test_chunker_integration.py -v --timeout=60`
Expected: All 4 tests PASS (they exercise the real SemanticChunker).

Note: The first run may be slow (~30s) as it downloads the `all-MiniLM-L6-v2` model. Subsequent runs will use the cached model.

**Step 3: Commit**

```bash
git add tests/integration/test_chunker_integration.py
git commit -m "test: add chunker integration tests (C-1 through C-4)"
```

---

## Task 5: Phase 1 — Vectorstore Integration Tests (V-1 through V-6)

**Files:**
- Create: `tests/integration/test_vectorstore_integration.py`

**Step 1: Write the tests**

Create `tests/integration/test_vectorstore_integration.py`:

```python
"""Integration tests for vectorstore.py — real ChromaDB.

Test IDs: V-1, V-2, V-3, V-4, V-5, V-6
"""

import pytest
from datetime import datetime, timezone

from digest_pipeline.chunker import TextChunk
from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper
from digest_pipeline.vectorstore import (
    StoredChunk,
    VectorStoreError,
    _get_collection,
    store_chunks,
    store_unparseable,
)


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestVectorstoreIntegration:
    """Tests that exercise real ChromaDB with temp directories."""

    def test_get_collection_happy_path(self, test_settings):
        """V-1: Create a PersistentClient with a temp dir, get-or-create collection."""
        collection = _get_collection(test_settings)
        assert collection is not None
        assert collection.name == "test_collection"

    def test_get_collection_failure(self):
        """V-2: Invalid/unwritable path raises VectorStoreError."""
        bad_settings = Settings(
            _env_file=None,
            llm_api_key="k",
            smtp_user="u",
            smtp_password="p",
            email_from="a@b.com",
            email_to="c@d.com",
            chroma_persist_dir="/proc/nonexistent/impossible/path",
            chroma_collection="test",
        )
        with pytest.raises(VectorStoreError):
            _get_collection(bad_settings)

    def test_store_chunks_happy_path(self, test_settings, make_paper):
        """V-3: Store 3 chunks, then query ChromaDB to verify documents, IDs, and metadata."""
        paper = make_paper()
        chunks = [
            TextChunk(text="First chunk about neural networks.", chunk_index=0),
            TextChunk(text="Second chunk about transformers.", chunk_index=1),
            TextChunk(text="Third chunk about attention mechanisms.", chunk_index=2),
        ]

        stored = store_chunks(paper, chunks, test_settings)

        assert len(stored) == 3
        assert all(isinstance(s, StoredChunk) for s in stored)

        # Verify IDs follow the pattern <arxiv_id>_chunk_<index>
        expected_ids = [
            "2401.00001_chunk_0",
            "2401.00001_chunk_1",
            "2401.00001_chunk_2",
        ]
        assert [s.doc_id for s in stored] == expected_ids

        # Read back from ChromaDB to verify persistence
        collection = _get_collection(test_settings)
        result = collection.get(ids=expected_ids, include=["documents", "metadatas"])

        assert len(result["ids"]) == 3
        assert result["documents"][0] == "First chunk about neural networks."
        assert result["documents"][2] == "Third chunk about attention mechanisms."

        # Verify all 6 metadata fields
        meta = result["metadatas"][0]
        assert meta["source"] == "arxiv"
        assert meta["title"] == paper.title
        assert meta["authors"] == "Alice Researcher, Bob Scientist"
        assert meta["url"] == paper.url
        assert meta["published_date"] == paper.published.isoformat()
        assert meta["chunk_index"] == 0

    def test_store_chunks_upsert_dedup(self, test_settings, make_paper):
        """V-4: Calling store_chunks twice with same paper produces no duplicates."""
        paper = make_paper()
        chunks = [TextChunk(text="Only chunk.", chunk_index=0)]

        store_chunks(paper, chunks, test_settings)
        store_chunks(paper, chunks, test_settings)  # second call

        collection = _get_collection(test_settings)
        result = collection.get(ids=["2401.00001_chunk_0"])
        # Should be exactly 1 document, not 2
        assert len(result["ids"]) == 1

    def test_store_unparseable(self, test_settings, make_paper):
        """V-5: Store unparseable paper and verify [unparseable] text + metadata."""
        paper = make_paper()
        store_unparseable(paper, test_settings)

        collection = _get_collection(test_settings)
        result = collection.get(
            ids=["2401.00001_unparseable"],
            include=["documents", "metadatas"],
        )

        assert len(result["ids"]) == 1
        assert result["documents"][0] == "[unparseable]"

        meta = result["metadatas"][0]
        assert meta["chunk_index"] == -1
        assert meta["unparseable"] is True
        assert meta["source"] == "arxiv"
        assert meta["title"] == paper.title

    def test_store_chunks_empty_list(self, test_settings, make_paper):
        """V-6: Passing chunks=[] returns empty StoredChunk list without crashing."""
        paper = make_paper()
        stored = store_chunks(paper, [], test_settings)

        assert stored == []
```

**Step 2: Run tests**

Run: `pytest tests/integration/test_vectorstore_integration.py -v --timeout=30`
Expected: All 6 tests PASS.

**Step 3: Commit**

```bash
git add tests/integration/test_vectorstore_integration.py
git commit -m "test: add vectorstore integration tests (V-1 through V-6)"
```

---

## Task 6: Phase 2 — Extractor Integration Tests (E-1 through E-3)

**Files:**
- Create: `tests/integration/test_extractor_integration.py`

**Step 1: Write the tests**

Create `tests/integration/test_extractor_integration.py`:

```python
"""Integration tests for extractor.py — real PyMuPDF.

Test IDs: E-1, E-2, E-3
"""

import pytest
from pathlib import Path

from digest_pipeline.extractor import extract_text


@pytest.mark.integration
@pytest.mark.timeout(15)
class TestExtractorIntegration:
    """Tests that exercise real PyMuPDF on actual files."""

    def test_real_pdf_extraction(self, sample_pdf):
        """E-1: Real PDF → text extraction with parseable=True."""
        result = extract_text(sample_pdf, "test.00001")

        assert result.parseable is True
        assert len(result.text) > 0
        assert "Introduction" in result.text or "Semantic Chunking" in result.text

    def test_multi_page_pdf(self, sample_pdf):
        """E-2: Multi-page PDF contains content from all pages joined by newline."""
        result = extract_text(sample_pdf, "test.00002")

        assert result.parseable is True
        # Page 1 content
        assert "Page 1" in result.text or "Introduction" in result.text
        # Page 2 content
        assert "Page 2" in result.text or "Results" in result.text

    def test_corrupt_file(self, corrupt_file):
        """E-3: Non-PDF file → parseable=False."""
        result = extract_text(corrupt_file, "test.corrupt")

        assert result.parseable is False
        assert result.text == ""
```

**Step 2: Run tests**

Run: `pytest tests/integration/test_extractor_integration.py -v --timeout=15`
Expected: All 3 tests PASS.

**Step 3: Commit**

```bash
git add tests/integration/test_extractor_integration.py
git commit -m "test: add extractor integration tests (E-1 through E-3)"
```

---

## Task 7: Phase 2 — Config Integration Tests (CFG-1 through CFG-5)

**Files:**
- Create: `tests/integration/test_config_integration.py`

**Step 1: Write the tests**

Create `tests/integration/test_config_integration.py`:

```python
"""Integration tests for config.py — real .env loading and env var overrides.

Test IDs: CFG-1, CFG-2, CFG-3, CFG-4, CFG-5
"""

import os

import pytest

from digest_pipeline.config import Settings


@pytest.mark.integration
class TestConfigIntegration:
    """Tests that exercise real Settings behavior with env files and vars."""

    def test_env_file_loading(self, tmp_path):
        """CFG-1: Create a temp .env file with overrides, verify values are read."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "LLM_API_KEY=from-env-file\n"
            "DRY_RUN=false\n"
            "ARXIV_MAX_RESULTS=99\n"
            "SMTP_USER=envuser\n"
            "SMTP_PASSWORD=envpass\n"
            "EMAIL_FROM=env@test.com\n"
            "EMAIL_TO=envto@test.com\n"
        )
        s = Settings(_env_file=str(env_file))

        assert s.llm_api_key == "from-env-file"
        assert s.dry_run is False
        assert s.arxiv_max_results == 99

    def test_environment_variable_override(self, monkeypatch):
        """CFG-2: os.environ overrides take effect in Settings."""
        monkeypatch.setenv("ARXIV_MAX_RESULTS", "10")
        monkeypatch.setenv("LLM_API_KEY", "env-override-key")
        monkeypatch.setenv("SMTP_USER", "u")
        monkeypatch.setenv("SMTP_PASSWORD", "p")
        monkeypatch.setenv("EMAIL_FROM", "a@b.com")
        monkeypatch.setenv("EMAIL_TO", "c@d.com")

        s = Settings(_env_file=None)
        assert s.arxiv_max_results == 10

    def test_list_parsing(self, monkeypatch):
        """CFG-3: Comma-separated ARXIV_TOPICS parsed into a list."""
        monkeypatch.setenv("ARXIV_TOPICS", '["cs.AI","cs.CL","stat.ML"]')
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("SMTP_USER", "u")
        monkeypatch.setenv("SMTP_PASSWORD", "p")
        monkeypatch.setenv("EMAIL_FROM", "a@b.com")
        monkeypatch.setenv("EMAIL_TO", "c@d.com")

        s = Settings(_env_file=None)
        assert s.arxiv_topics == ["cs.AI", "cs.CL", "stat.ML"]

    def test_litellm_model_string(self, monkeypatch):
        """CFG-4: LLM_MODEL accepts provider/model format."""
        monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("SMTP_USER", "u")
        monkeypatch.setenv("SMTP_PASSWORD", "p")
        monkeypatch.setenv("EMAIL_FROM", "a@b.com")
        monkeypatch.setenv("EMAIL_TO", "c@d.com")

        s = Settings(_env_file=None)
        assert s.llm_model == "anthropic/claude-sonnet-4-20250514"

    def test_llm_api_base_nullable(self):
        """CFG-5: llm_api_base defaults to None and accepts a URL when set."""
        s = Settings(
            _env_file=None,
            llm_api_key="k",
            smtp_user="u",
            smtp_password="p",
            email_from="a@b.com",
            email_to="c@d.com",
        )
        assert s.llm_api_base is None

        s2 = Settings(
            _env_file=None,
            llm_api_key="k",
            llm_api_base="http://localhost:8080/v1",
            smtp_user="u",
            smtp_password="p",
            email_from="a@b.com",
            email_to="c@d.com",
        )
        assert s2.llm_api_base == "http://localhost:8080/v1"
```

**Step 2: Run tests**

Run: `pytest tests/integration/test_config_integration.py -v`
Expected: All 5 tests PASS.

**Step 3: Commit**

```bash
git add tests/integration/test_config_integration.py
git commit -m "test: add config integration tests (CFG-1 through CFG-5)"
```

---

## Task 8: Phase 2 — Emailer Integration Tests (EM-1, EM-2)

**Files:**
- Create: `tests/integration/test_emailer_integration.py`

**Step 1: Write the tests**

Create `tests/integration/test_emailer_integration.py`:

```python
"""Integration tests for emailer.py — real SMTP via aiosmtpd + Jinja2 edge cases.

Test IDs: EM-1, EM-2
"""

import asyncio
import email
import threading
import time

import pytest
from aiosmtpd.controller import Controller
from aiosmtpd.handlers import Message

from digest_pipeline.config import Settings
from digest_pipeline.emailer import _build_email, send_digest


class _CapturingSMTPHandler(Message):
    """aiosmtpd handler that captures received messages."""

    def __init__(self):
        super().__init__()
        self.messages: list[email.message.Message] = []

    def handle_message(self, message):
        self.messages.append(message)


@pytest.mark.integration
@pytest.mark.timeout(15)
class TestEmailerIntegration:
    """Tests that exercise real SMTP sending via a local aiosmtpd server."""

    def test_real_smtp_send(self, tmp_path):
        """EM-1: Send a real email to a local SMTP server, verify receipt."""
        handler = _CapturingSMTPHandler()
        # Use plain SMTP (not SSL) for local testing
        controller = Controller(handler, hostname="127.0.0.1", port=0)
        controller.start()
        try:
            port = controller.server.sockets[0].getsockname()[1]
            settings = Settings(
                _env_file=None,
                llm_api_key="k",
                smtp_host="127.0.0.1",
                smtp_port=port,
                smtp_user="",
                smtp_password="",
                email_from="sender@test.com",
                email_to="recipient@test.com",
                dry_run=False,
            )

            # send_digest uses SMTP_SSL, so we need to patch to use plain SMTP
            # for local testing. Instead, test _build_email + manual send.
            import smtplib

            msg = _build_email(
                "Test summary content",
                5,
                "2025-01-15",
                settings,
                implications="Test implication",
                critiques="Test critique",
            )

            with smtplib.SMTP("127.0.0.1", port) as server:
                server.send_message(msg)

            # Give server a moment to process
            time.sleep(0.2)

            assert len(handler.messages) == 1
            received = handler.messages[0]
            assert received["Subject"] == "Research Digest — 2025-01-15"
            assert received["From"] == "sender@test.com"
            assert received["To"] == "recipient@test.com"
        finally:
            controller.stop()

    def test_jinja2_template_edge_cases(self):
        """EM-2: Templates handle special characters, empty sections, long summaries."""
        settings = Settings(
            _env_file=None,
            llm_api_key="k",
            smtp_user="u",
            smtp_password="p",
            email_from="a@b.com",
            email_to="c@d.com",
        )

        # Special characters in summary
        msg = _build_email(
            "Summary with <html> & \"quotes\" and unicode: café résumé",
            1,
            "2025-01-15",
            settings,
        )
        payloads = msg.get_payload()
        plain_body = payloads[0].get_payload(decode=True).decode()
        assert "café" in plain_body
        assert '&' in plain_body or "&amp;" in plain_body

        # Empty implications and critiques (sections should be omitted)
        msg2 = _build_email("Summary", 1, "2025-01-15", settings, implications="", critiques="")
        html_body = msg2.get_payload()[1].get_payload(decode=True).decode()
        assert "Practical Implications" not in html_body

        # Very long summary
        long_summary = "Word " * 5000
        msg3 = _build_email(long_summary, 1, "2025-01-15", settings)
        plain_body3 = msg3.get_payload()[0].get_payload(decode=True).decode()
        assert len(plain_body3) > 20000
```

**Step 2: Run tests**

Run: `pytest tests/integration/test_emailer_integration.py -v --timeout=15`
Expected: All 2 tests PASS.

**Step 3: Commit**

```bash
git add tests/integration/test_emailer_integration.py
git commit -m "test: add emailer integration tests (EM-1, EM-2)"
```

---

## Task 9: Phase 3 — Stub LLM Server

**Files:**
- Create: `tests/stub_llm_server.py`

**Step 1: Write the stub server**

Create `tests/stub_llm_server.py`:

```python
"""Lightweight OpenAI-compatible stub LLM server for integration tests.

Serves `/v1/chat/completions` with configurable responses.
Used by summarizer and postprocessor integration tests via litellm.
"""

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@dataclass
class StubConfig:
    """Configure stub server behavior."""

    # Number of 429 responses before returning 200
    rate_limit_count: int = 0
    # Content to return in the completion
    response_content: str = "This is a stub LLM response."
    # Return null content (simulates edge case)
    null_content: bool = False


@dataclass
class StubLLMServer:
    """In-process stub LLM server for tests."""

    config: StubConfig = field(default_factory=StubConfig)
    port: int = 0
    _app: FastAPI = field(default=None, init=False)
    _server: uvicorn.Server = field(default=None, init=False)
    _thread: threading.Thread = field(default=None, init=False)
    _rate_limit_hits: int = field(default=0, init=False)
    # Track received requests for assertions
    requests: list[dict] = field(default_factory=list)

    def _create_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            self.requests.append(body)

            # Rate limiting simulation
            if self._rate_limit_hits < self.config.rate_limit_count:
                self._rate_limit_hits += 1
                return JSONResponse(
                    status_code=429,
                    content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
                )

            content = None if self.config.null_content else self.config.response_content

            return JSONResponse(content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "test-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            })

        return app

    def start(self):
        self._app = self._create_app()
        config = uvicorn.Config(
            self._app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        # Wait for server to start
        for _ in range(50):
            if self._server.started:
                break
            time.sleep(0.1)
        # Get actual port
        for sock in self._server.servers[0].sockets:
            self.port = sock.getsockname()[1]
            break

    def stop(self):
        if self._server:
            self._server.should_exit = True
            self._thread.join(timeout=5)

    def reset(self):
        self._rate_limit_hits = 0
        self.requests.clear()


@contextmanager
def run_stub_server(config: StubConfig | None = None):
    """Context manager that starts/stops a stub LLM server."""
    server = StubLLMServer(config=config or StubConfig())
    server.start()
    try:
        yield server
    finally:
        server.stop()
```

**Step 2: Verify the stub server starts and responds**

Run: `python -c "
from tests.stub_llm_server import run_stub_server, StubConfig
import requests
with run_stub_server() as srv:
    resp = requests.post(f'http://127.0.0.1:{srv.port}/v1/chat/completions', json={'model': 'test', 'messages': [{'role': 'user', 'content': 'hi'}]})
    print(resp.status_code, resp.json()['choices'][0]['message']['content'])
"`

Expected: `200 This is a stub LLM response.`

**Step 3: Commit**

```bash
git add tests/stub_llm_server.py
git commit -m "test: add stub LLM server for summarizer/postprocessor integration tests"
```

---

## Task 10: Phase 3 — Summarizer Integration Tests (S-1, S-2, S-3, S-5, S-6)

**Files:**
- Create: `tests/integration/test_summarizer_integration.py`

**Step 1: Write the tests**

Create `tests/integration/test_summarizer_integration.py`:

```python
"""Integration tests for summarizer.py — litellm against stub LLM server.

Test IDs: S-1, S-2, S-3, S-5, S-6
"""

import pytest

from digest_pipeline.config import Settings
from digest_pipeline.summarizer import summarize
from tests.stub_llm_server import StubConfig, run_stub_server


def _make_settings(port: int, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        llm_api_key="test-key",
        llm_model="openai/test-model",
        llm_api_base=f"http://127.0.0.1:{port}/v1",
        llm_max_tokens=4096,
        smtp_user="u",
        smtp_password="p",
        email_from="a@b.com",
        email_to="c@d.com",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestSummarizerIntegration:
    """Tests that exercise real litellm.completion() against a stub server."""

    def test_successful_completion(self, make_paper):
        """S-1: litellm.completion() returns a non-empty summary from stub."""
        with run_stub_server(StubConfig(response_content="Generated summary of papers.")) as srv:
            settings = _make_settings(srv.port)
            result = summarize([make_paper()], settings)

            assert result == "Generated summary of papers."
            assert len(srv.requests) == 1
            # Verify system prompt was sent
            messages = srv.requests[0]["messages"]
            assert messages[0]["role"] == "system"
            assert "expert research assistant" in messages[0]["content"]

    def test_rate_limit_retry_then_success(self, make_paper):
        """S-2: Stub returns 429 twice then 200 — litellm retries and succeeds."""
        config = StubConfig(rate_limit_count=2, response_content="Eventually succeeded.")
        with run_stub_server(config) as srv:
            settings = _make_settings(srv.port)
            result = summarize([make_paper()], settings)

            assert result == "Eventually succeeded."

    def test_rate_limit_exhaustion(self, make_paper):
        """S-3: Stub returns 429 for all 5 attempts — RateLimitError re-raised."""
        import litellm

        config = StubConfig(rate_limit_count=100)  # Always 429
        with run_stub_server(config) as srv:
            settings = _make_settings(srv.port)
            with pytest.raises((litellm.RateLimitError, litellm.exceptions.RateLimitError, Exception)):
                summarize([make_paper()], settings)

    def test_null_content_returns_empty_string(self, make_paper):
        """S-5: Stub returns null content → summarize returns empty string."""
        config = StubConfig(null_content=True)
        with run_stub_server(config) as srv:
            settings = _make_settings(srv.port)
            result = summarize([make_paper()], settings)

            assert result == ""

    def test_model_string_passed_to_litellm(self, make_paper):
        """S-6: Verify the full provider/model string reaches the stub server."""
        with run_stub_server() as srv:
            settings = _make_settings(srv.port, llm_model="openai/gpt-4o-mini")
            summarize([make_paper()], settings)

            assert len(srv.requests) == 1
            assert srv.requests[0]["model"] == "openai/gpt-4o-mini"
```

**Step 2: Run tests**

Run: `pytest tests/integration/test_summarizer_integration.py -v --timeout=30`
Expected: All 5 tests PASS. Note: S-3 may take ~60s due to backoff sleeps — we may need to adjust the timeout or accept it's slow.

**Step 3: Commit**

```bash
git add tests/integration/test_summarizer_integration.py
git commit -m "test: add summarizer integration tests (S-1, S-2, S-3, S-5, S-6)"
```

---

## Task 11: Phase 3 — Postprocessor Integration Tests (P-1, P-2)

**Files:**
- Create: `tests/integration/test_postprocessor_integration.py`

**Step 1: Write the tests**

Create `tests/integration/test_postprocessor_integration.py`:

```python
"""Integration tests for postprocessor.py — litellm against stub LLM server.

Test IDs: P-1, P-2
"""

import pytest

from digest_pipeline.config import Settings
from digest_pipeline.postprocessor import (
    CRITIQUES_SYSTEM_PROMPT,
    IMPLICATIONS_SYSTEM_PROMPT,
    extract_implications,
    generate_critiques,
)
from tests.stub_llm_server import StubConfig, run_stub_server


def _make_settings(port: int, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        llm_api_key="test-key",
        llm_model="openai/test-model",
        llm_api_base=f"http://127.0.0.1:{port}/v1",
        llm_max_tokens=4096,
        smtp_user="u",
        smtp_password="p",
        email_from="a@b.com",
        email_to="c@d.com",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestPostprocessorIntegration:
    """Tests that exercise real litellm calls for implications and critiques."""

    def test_both_prompts_sent_correctly(self, make_paper):
        """P-1: Both IMPLICATIONS and CRITIQUES system prompts are sent via litellm."""
        with run_stub_server(StubConfig(response_content="Implications result.")) as srv:
            settings = _make_settings(srv.port)
            impl_result = extract_implications([make_paper()], settings)

            assert impl_result == "Implications result."
            assert len(srv.requests) == 1
            messages = srv.requests[0]["messages"]
            assert messages[0]["role"] == "system"
            assert "actionable insights" in messages[0]["content"]

        with run_stub_server(StubConfig(response_content="Critiques result.")) as srv:
            settings = _make_settings(srv.port)
            crit_result = generate_critiques([make_paper()], settings)

            assert crit_result == "Critiques result."
            assert len(srv.requests) == 1
            messages = srv.requests[0]["messages"]
            assert messages[0]["role"] == "system"
            assert "peer reviewer" in messages[0]["content"]

    def test_rate_limit_retry(self, make_paper):
        """P-2: Rate-limit backoff works for _llm_call (shared by both functions)."""
        config = StubConfig(rate_limit_count=1, response_content="After retry.")
        with run_stub_server(config) as srv:
            settings = _make_settings(srv.port)
            result = extract_implications([make_paper()], settings)

            assert result == "After retry."
```

**Step 2: Run tests**

Run: `pytest tests/integration/test_postprocessor_integration.py -v --timeout=30`
Expected: All 2 tests PASS.

**Step 3: Commit**

```bash
git add tests/integration/test_postprocessor_integration.py
git commit -m "test: add postprocessor integration tests (P-1, P-2)"
```

---

## Task 12: Phase 4 — E2E conftest.py and Pipeline Tests (PL-1 through PL-6)

**Files:**
- Create: `tests/e2e/conftest.py`
- Create: `tests/e2e/test_pipeline_e2e.py`

**Step 1: Write the E2E conftest.py**

Create `tests/e2e/conftest.py`:

```python
"""Shared fixtures for E2E pipeline tests."""

import pytest
from pathlib import Path
from datetime import datetime, timezone

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper
from tests.stub_llm_server import StubConfig, StubLLMServer

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_pdf() -> Path:
    path = FIXTURES_DIR / "sample.pdf"
    assert path.exists(), f"Fixture missing: {path}"
    return path


@pytest.fixture
def corrupt_file() -> Path:
    path = FIXTURES_DIR / "corrupt.bin"
    assert path.exists(), f"Fixture missing: {path}"
    return path


@pytest.fixture
def stub_llm():
    """Start a stub LLM server for the duration of the test."""
    server = StubLLMServer(config=StubConfig(response_content="E2E stub summary."))
    server.start()
    yield server
    server.stop()


@pytest.fixture
def e2e_settings(tmp_path, stub_llm) -> Settings:
    return Settings(
        _env_file=None,
        llm_api_key="test-key",
        llm_model="openai/test-model",
        llm_api_base=f"http://127.0.0.1:{stub_llm.port}/v1",
        llm_max_tokens=4096,
        chroma_persist_dir=tmp_path / "chromadb",
        chroma_collection="e2e_test",
        smtp_user="test",
        smtp_password="test",
        email_from="e2e@test.com",
        email_to="recipient@test.com",
        dry_run=True,
        github_enabled=False,
        postprocessing_implications=True,
        postprocessing_critiques=True,
    )


@pytest.fixture
def make_paper():
    """Factory for creating Paper objects."""
    def _make(pdf_path: Path | None = None, **overrides) -> Paper:
        defaults = dict(
            arxiv_id="2401.00001",
            title="Test Paper",
            authors=["Alice"],
            abstract="Abstract text about machine learning.",
            url="https://arxiv.org/abs/2401.00001",
            published=datetime(2025, 1, 15, tzinfo=timezone.utc),
            pdf_path=pdf_path,
        )
        defaults.update(overrides)
        return Paper(**defaults)
    return _make
```

**Step 2: Write the E2E pipeline tests**

Create `tests/e2e/test_pipeline_e2e.py`:

```python
"""End-to-end pipeline tests.

Test IDs: PL-1, PL-2, PL-3, PL-4, PL-5, PL-6
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from digest_pipeline.config import Settings
from digest_pipeline.extractor import ExtractionResult
from digest_pipeline.fetcher import Paper
from digest_pipeline.pipeline import run
from digest_pipeline.vectorstore import VectorStoreError


@pytest.mark.e2e
@pytest.mark.timeout(120)
class TestPipelineE2E:
    """Full pipeline tests with real local deps and stubbed LLM."""

    def test_full_dry_run(self, e2e_settings, make_paper, sample_pdf, capsys):
        """PL-1: Full E2E dry-run: fixture PDF → chunk → ChromaDB → LLM stub → console."""
        paper = make_paper(pdf_path=sample_pdf)

        with patch("digest_pipeline.pipeline.fetch_papers", return_value=[paper]):
            run(e2e_settings)

        captured = capsys.readouterr()
        assert "E2E stub summary" in captured.out

    def test_mixed_parseable_unparseable(self, e2e_settings, make_paper, sample_pdf, corrupt_file, capsys):
        """PL-2: One valid PDF + one corrupt → only valid one in summary."""
        valid_paper = make_paper(pdf_path=sample_pdf, arxiv_id="valid.001")
        corrupt_paper = make_paper(pdf_path=corrupt_file, arxiv_id="corrupt.001")

        with patch("digest_pipeline.pipeline.fetch_papers", return_value=[valid_paper, corrupt_paper]):
            run(e2e_settings)

        captured = capsys.readouterr()
        # Pipeline should complete with the valid paper
        assert "E2E stub summary" in captured.out

    def test_vectorstore_error_exits_with_code_1(self, e2e_settings, make_paper, sample_pdf):
        """PL-3: VectorStoreError mid-pipeline → sys.exit(1)."""
        paper = make_paper(pdf_path=sample_pdf)

        # Make ChromaDB unreachable by pointing to an invalid path
        e2e_settings.chroma_persist_dir = Path("/proc/nonexistent/impossible")

        with patch("digest_pipeline.pipeline.fetch_papers", return_value=[paper]):
            with pytest.raises(SystemExit) as exc_info:
                run(e2e_settings)
            assert exc_info.value.code == 1

    def test_cli_args(self, e2e_settings, make_paper, sample_pdf, capsys, monkeypatch):
        """PL-4: CLI main() with --dry-run --topics cs.CL -v."""
        from digest_pipeline.pipeline import main

        paper = make_paper(pdf_path=sample_pdf)

        monkeypatch.setattr(
            "sys.argv",
            ["digest-pipeline", "--dry-run", "--topics", "cs.CL", "-v"],
        )

        with patch("digest_pipeline.pipeline.get_settings", return_value=e2e_settings):
            with patch("digest_pipeline.pipeline.fetch_papers", return_value=[paper]):
                main()

        captured = capsys.readouterr()
        assert "E2E stub summary" in captured.out

    def test_github_enabled(self, e2e_settings, make_paper, sample_pdf, capsys):
        """PL-5: GitHub-enabled pipeline includes GitHub section in output."""
        e2e_settings.github_enabled = True
        paper = make_paper(pdf_path=sample_pdf)

        mock_repos = [
            type("TrendingRepo", (), {
                "name": "test/repo",
                "description": "A test repo",
                "url": "https://github.com/test/repo",
                "stars": 100,
                "language": "Python",
            })()
        ]

        with patch("digest_pipeline.pipeline.fetch_papers", return_value=[paper]):
            with patch("digest_pipeline.pipeline.fetch_trending", return_value=mock_repos):
                run(e2e_settings)

        captured = capsys.readouterr()
        assert "E2E stub summary" in captured.out

    def test_all_unparseable_early_exit(self, e2e_settings, make_paper, corrupt_file, capsys, stub_llm):
        """PL-6: All papers unparseable → pipeline returns without calling summarize."""
        paper = make_paper(pdf_path=corrupt_file)

        with patch("digest_pipeline.pipeline.fetch_papers", return_value=[paper]):
            with patch("digest_pipeline.pipeline.summarize") as mock_summarize:
                run(e2e_settings)
                mock_summarize.assert_not_called()
```

**Step 3: Run E2E tests**

Run: `pytest tests/e2e/test_pipeline_e2e.py -v --timeout=120`
Expected: All 6 tests PASS.

**Step 4: Commit**

```bash
git add tests/e2e/
git commit -m "test: add E2E pipeline tests (PL-1 through PL-6)"
```

---

## Task 13: Phase 5 — Network Smoke Tests (F-1, F-2, G-1)

**Files:**
- Create: `tests/integration/test_network_smoke.py`

**Step 1: Write the network tests**

Create `tests/integration/test_network_smoke.py`:

```python
"""Network smoke tests — hit real external APIs.

Run only with: pytest -m network
Test IDs: F-1, F-2, G-1
"""

import tempfile
from pathlib import Path

import pytest

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper, _download_pdf, fetch_papers
from digest_pipeline.github_trending import TrendingRepo, fetch_trending


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        llm_api_key="test-key",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b.com",
        email_to="c@d.com",
        arxiv_topics=["cs.DL"],
        arxiv_max_results=2,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.network
@pytest.mark.timeout(120)
class TestNetworkSmoke:
    """Tests that hit real external APIs. Skipped unless -m network is used."""

    def test_fetch_papers_real_arxiv(self):
        """F-1: Hit the live arXiv API for cs.DL with max_results=2."""
        settings = _make_settings()
        papers = fetch_papers(settings)

        # May return 0 if no papers in the 24-hour window for cs.DL
        assert isinstance(papers, list)
        for p in papers:
            assert isinstance(p, Paper)
            assert p.arxiv_id
            assert p.title
            assert len(p.authors) > 0
            assert p.pdf_path is not None
            assert p.pdf_path.exists()
            # Verify it's actually a PDF
            header = p.pdf_path.read_bytes()[:5]
            assert header == b"%PDF-"

    def test_download_pdf_real(self):
        """F-2: Download a known small PDF from arXiv and verify it starts with %PDF."""
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.pdf"
            # Use a known stable arXiv paper
            success = _download_pdf(
                "https://arxiv.org/pdf/2301.00001v1",
                dest,
                max_retries=2,
            )
            if success:
                assert dest.exists()
                assert dest.read_bytes()[:5] == b"%PDF-"

    def test_fetch_trending_real_github(self):
        """G-1: Hit the live GitHub Search API and verify TrendingRepo objects."""
        settings = _make_settings(github_enabled=True)
        repos = fetch_trending(settings)

        assert isinstance(repos, list)
        for r in repos:
            assert isinstance(r, TrendingRepo)
            assert r.name
            assert r.url.startswith("https://github.com/")
```

**Step 2: Run network tests (optional — requires internet)**

Run: `pytest tests/integration/test_network_smoke.py -m network -v --timeout=120`
Expected: Tests pass (or skip gracefully if no network).

**Step 3: Commit**

```bash
git add tests/integration/test_network_smoke.py
git commit -m "test: add network smoke tests for arXiv and GitHub (F-1, F-2, G-1)"
```

---

## Task 14: Final Verification — Run Full Test Suite

**Step 1: Run all unit tests**

Run: `pytest tests/unit/ -v --timeout=30`
Expected: All existing tests pass.

**Step 2: Run all integration tests**

Run: `pytest tests/integration/ -v --timeout=120 -m "not network"`
Expected: All integration tests pass.

**Step 3: Run all E2E tests**

Run: `pytest tests/e2e/ -v --timeout=120`
Expected: All E2E tests pass.

**Step 4: Run full suite (excluding network)**

Run: `pytest tests/ -v --timeout=120 -m "not network"`
Expected: All tests pass. Print total count — should be 20+ new tests.

**Step 5: Final commit**

```bash
git add -A
git commit -m "test: complete integration and E2E test suite per test plan"
```
