"""Tests for the HuggingFace Daily Papers fetcher module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from digest_pipeline.hf_fetcher import (
    HFDailyPaper,
    fetch_hf_daily,
    normalize_arxiv_id,
    reconcile_hf_papers,
)


def _make_hf_entry(paper_id="2401.00001", title="HF Paper", hours_ago=1, upvotes=5):
    """Build a fake HuggingFace daily papers API response entry."""
    published = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return {
        "publishedAt": published,
        "paper": {
            "id": paper_id,
            "title": title,
            "summary": "This paper does something novel.",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "upvotes": upvotes,
        },
    }


# ── normalize_arxiv_id ───────────────────────────────────────────


def test_normalize_arxiv_id_strips_version():
    assert normalize_arxiv_id("2401.00001v2") == "2401.00001"


def test_normalize_arxiv_id_no_version():
    assert normalize_arxiv_id("2401.00001") == "2401.00001"


def test_normalize_arxiv_id_high_version():
    assert normalize_arxiv_id("2401.00001v15") == "2401.00001"


# ── fetch_hf_daily ───────────────────────────────────────────────


@patch("digest_pipeline.hf_fetcher.requests.get")
def test_fetch_hf_daily_success(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [_make_hf_entry()]
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(huggingface_enabled=True, huggingface_max_results=10)
    papers = fetch_hf_daily(settings)

    assert len(papers) == 1
    assert isinstance(papers[0], HFDailyPaper)
    assert papers[0].arxiv_id == "2401.00001"
    assert papers[0].title == "HF Paper"
    assert papers[0].authors == ["Alice", "Bob"]
    assert papers[0].upvotes == 5
    assert "huggingface.co/papers" in papers[0].url


@patch("digest_pipeline.hf_fetcher.requests.get")
def test_fetch_hf_daily_filters_old(mock_get, make_settings):
    """Papers older than 24h are filtered out."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = [
        _make_hf_entry(paper_id="new", hours_ago=1),
        _make_hf_entry(paper_id="old", hours_ago=25),
    ]
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(huggingface_enabled=True, huggingface_max_results=10)
    papers = fetch_hf_daily(settings)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "new"


@patch("digest_pipeline.hf_fetcher.requests.get")
def test_fetch_hf_daily_respects_max_results(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [
        _make_hf_entry(paper_id=f"p{i}", hours_ago=1) for i in range(10)
    ]
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(huggingface_enabled=True, huggingface_max_results=3)
    papers = fetch_hf_daily(settings)

    assert len(papers) <= 3


@patch("digest_pipeline.hf_fetcher.requests.get", side_effect=Exception("network error"))
def test_fetch_hf_daily_handles_error(mock_get, make_settings):
    settings = make_settings(huggingface_enabled=True)
    papers = fetch_hf_daily(settings)

    assert papers == []


@patch("digest_pipeline.hf_fetcher.requests.get")
def test_fetch_hf_daily_sends_auth_header(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.json.return_value = []
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(huggingface_enabled=True, huggingface_token="hf_test123")
    fetch_hf_daily(settings)

    call_kwargs = mock_get.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer hf_test123"


# ── reconcile_hf_papers ─────────────────────────────────────────


def test_reconcile_new_paper():
    """An HF paper not in known_ids is returned as a new Paper."""
    hf = HFDailyPaper(
        arxiv_id="2401.99999",
        title="New Paper",
        authors=["Alice"],
        abstract="Abstract",
        url="https://huggingface.co/papers/2401.99999",
        published=datetime.now(timezone.utc),
        upvotes=10,
    )
    new, trending = reconcile_hf_papers([hf], known_ids=set())

    assert len(new) == 1
    assert len(trending) == 0
    assert new[0].paper_id == "hf_2401.99999"
    assert new[0].source == "huggingface"
    assert new[0].title == "New Paper"


def test_reconcile_trending_paper():
    """An HF paper already in known_ids is returned as trending."""
    hf = HFDailyPaper(
        arxiv_id="2401.00001",
        title="Known Paper",
        authors=["Bob"],
        abstract="Abstract",
        url="https://huggingface.co/papers/2401.00001",
        published=datetime.now(timezone.utc),
        upvotes=42,
    )
    new, trending = reconcile_hf_papers([hf], known_ids={"2401.00001"})

    assert len(new) == 0
    assert len(trending) == 1
    assert trending[0].arxiv_id == "2401.00001"
    assert trending[0].upvotes == 42


def test_reconcile_version_suffix_match():
    """An HF paper with version suffix still matches a known ID."""
    hf = HFDailyPaper(
        arxiv_id="2401.00001v3",
        title="Versioned",
        authors=[],
        abstract="",
        url="https://huggingface.co/papers/2401.00001v3",
        published=datetime.now(timezone.utc),
        upvotes=7,
    )
    new, trending = reconcile_hf_papers([hf], known_ids={"2401.00001"})

    assert len(new) == 0
    assert len(trending) == 1


def test_reconcile_mixed():
    """Mix of new and trending papers are correctly split."""
    papers = [
        HFDailyPaper(
            arxiv_id="2401.00001",
            title="Known",
            authors=[],
            abstract="",
            url="url1",
            published=datetime.now(timezone.utc),
            upvotes=10,
        ),
        HFDailyPaper(
            arxiv_id="2401.99999",
            title="New",
            authors=[],
            abstract="",
            url="url2",
            published=datetime.now(timezone.utc),
            upvotes=20,
        ),
    ]
    new, trending = reconcile_hf_papers(papers, known_ids={"2401.00001"})

    assert len(new) == 1
    assert len(trending) == 1
    assert new[0].title == "New"
    assert trending[0].title == "Known"


def test_reconcile_empty():
    """Empty input returns empty results."""
    new, trending = reconcile_hf_papers([], known_ids={"2401.00001"})

    assert new == []
    assert trending == []
