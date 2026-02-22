"""Shared fixtures for E2E pipeline tests."""

import json

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
    """Start a stub LLM server that returns JSON-formatted responses."""
    json_response = json.dumps({"paper_1": "E2E stub summary."})
    server = StubLLMServer(config=StubConfig(response_content=json_response))
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
            paper_id="2401.00001",
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
