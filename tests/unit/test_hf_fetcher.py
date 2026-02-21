"""Tests for the HuggingFace Daily Papers fetcher module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from digest_pipeline.hf_fetcher import fetch_hf_papers


def _make_hf_entry(paper_id="2401.00001", title="HF Paper", hours_ago=1):
    """Build a fake HuggingFace daily papers API response entry."""
    published = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {
        "publishedAt": published,
        "paper": {
            "id": paper_id,
            "title": title,
            "summary": "This paper does something novel.",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
        },
    }


@patch("digest_pipeline.hf_fetcher.requests.get")
def test_fetch_hf_papers_success(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [_make_hf_entry()]
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(huggingface_enabled=True, huggingface_max_results=10)
    papers = fetch_hf_papers(settings)

    assert len(papers) == 1
    assert papers[0].source == "huggingface"
    assert papers[0].paper_id == "hf_2401.00001"
    assert papers[0].title == "HF Paper"
    assert papers[0].authors == ["Alice", "Bob"]
    assert papers[0].pdf_path is None
    assert "huggingface.co/papers" in papers[0].url


@patch("digest_pipeline.hf_fetcher.requests.get")
def test_fetch_hf_papers_filters_old(mock_get, make_settings):
    """Papers older than 24h are filtered out."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = [
        _make_hf_entry(paper_id="new", hours_ago=1),
        _make_hf_entry(paper_id="old", hours_ago=25),
    ]
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(huggingface_enabled=True, huggingface_max_results=10)
    papers = fetch_hf_papers(settings)

    assert len(papers) == 1
    assert papers[0].paper_id == "hf_new"


@patch("digest_pipeline.hf_fetcher.requests.get")
def test_fetch_hf_papers_respects_max_results(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [
        _make_hf_entry(paper_id=f"p{i}", hours_ago=1) for i in range(10)
    ]
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(huggingface_enabled=True, huggingface_max_results=3)
    papers = fetch_hf_papers(settings)

    assert len(papers) <= 3


@patch("digest_pipeline.hf_fetcher.requests.get", side_effect=Exception("network error"))
def test_fetch_hf_papers_handles_error(mock_get, make_settings):
    settings = make_settings(huggingface_enabled=True)
    papers = fetch_hf_papers(settings)

    assert papers == []


@patch("digest_pipeline.hf_fetcher.requests.get")
def test_fetch_hf_papers_sends_auth_header(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.json.return_value = []
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(huggingface_enabled=True, huggingface_token="hf_test123")
    fetch_hf_papers(settings)

    call_kwargs = mock_get.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer hf_test123"
