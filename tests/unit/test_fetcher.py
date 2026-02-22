"""Tests for the arXiv fetcher module."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from digest_pipeline.fetcher import Paper, download_pdf, _within_last_24h, fetch_papers


def test_within_last_24h_recent():
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    assert _within_last_24h(recent) is True


def test_within_last_24h_old():
    old = datetime.now(timezone.utc) - timedelta(hours=25)
    assert _within_last_24h(old) is False


def test_within_last_24h_naive_datetime():
    recent = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    assert _within_last_24h(recent) is True


@patch("digest_pipeline.fetcher.requests.get")
def testdownload_pdf_success(mock_get, tmp_path):
    mock_resp = MagicMock()
    mock_resp.content = b"%PDF-fake-content"
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    dest = tmp_path / "test.pdf"
    assert download_pdf("http://example.com/paper.pdf", dest, max_retries=3) is True
    assert dest.read_bytes() == b"%PDF-fake-content"


@patch("digest_pipeline.fetcher.requests.get", side_effect=Exception("network error"))
def testdownload_pdf_all_retries_fail(mock_get, tmp_path):
    dest = tmp_path / "test.pdf"
    assert download_pdf("http://example.com/paper.pdf", dest, max_retries=2) is False
    assert mock_get.call_count == 2
