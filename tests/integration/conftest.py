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
