"""Tests for the pipeline orchestrator."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from digest_pipeline.config import Settings
from digest_pipeline.extractor import ExtractionResult
from digest_pipeline.fetcher import Paper
from digest_pipeline.pipeline import run


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        llm_api_key="test-key",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b.com",
        email_to="c@d.com",
        dry_run=True,
        github_enabled=False,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        arxiv_id="2401.00001",
        title="Test Paper",
        authors=["Alice"],
        abstract="Abstract text.",
        url="https://arxiv.org/abs/2401.00001",
        published=datetime(2025, 1, 15, tzinfo=timezone.utc),
        pdf_path=Path("/tmp/fake.pdf"),
    )
    defaults.update(overrides)
    return Paper(**defaults)


@patch("digest_pipeline.pipeline.fetch_papers", return_value=[])
def test_run_no_papers(mock_fetch):
    """Pipeline exits gracefully when no papers are found."""
    run(_make_settings())
    mock_fetch.assert_called_once()


@patch("digest_pipeline.pipeline.send_digest")
@patch("digest_pipeline.pipeline.summarize", return_value="Summary")
@patch("digest_pipeline.pipeline.store_chunks", return_value=[])
@patch("digest_pipeline.pipeline.chunk_text", return_value=[])
@patch(
    "digest_pipeline.pipeline.extract_text",
    return_value=ExtractionResult(arxiv_id="2401.00001", text="content", parseable=True),
)
@patch("digest_pipeline.pipeline.fetch_papers")
def test_run_full_pipeline(mock_fetch, mock_extract, mock_chunk, mock_store, mock_summarize, mock_email):
    paper = _make_paper()
    mock_fetch.return_value = [paper]
    settings = _make_settings()

    run(settings)

    mock_fetch.assert_called_once_with(settings)
    mock_extract.assert_called_once()
    mock_chunk.assert_called_once_with("content")
    mock_store.assert_called_once()
    mock_summarize.assert_called_once()
    mock_email.assert_called_once()


@patch("digest_pipeline.pipeline.store_unparseable")
@patch(
    "digest_pipeline.pipeline.extract_text",
    return_value=ExtractionResult(arxiv_id="2401.00001", text="", parseable=False),
)
@patch("digest_pipeline.pipeline.fetch_papers")
def test_run_unparseable_paper(mock_fetch, mock_extract, mock_store_unparse):
    paper = _make_paper()
    mock_fetch.return_value = [paper]

    run(_make_settings())

    mock_store_unparse.assert_called_once()
