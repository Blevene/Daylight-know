"""Tests for the Semantic Scholar fetcher module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from digest_pipeline.s2_fetcher import fetch_s2_papers


def _make_s2_result(paper_id="abc123", title="S2 Paper", pub_date=None):
    """Build a fake Semantic Scholar search result."""
    if pub_date is None:
        pub_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": "A study on something interesting.",
        "authors": [{"name": "Carol"}, {"name": "Dave"}],
        "publicationDate": pub_date,
        "url": f"https://www.semanticscholar.org/paper/{paper_id}",
        "externalIds": {"ArXiv": "2401.99999"},
        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
    }


@patch("digest_pipeline.s2_fetcher.requests.get")
def test_fetch_s2_papers_success(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [_make_s2_result()]}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(
        semanticscholar_enabled=True,
        semanticscholar_query="deep learning",
        semanticscholar_max_results=10,
    )
    papers = fetch_s2_papers(settings)

    assert len(papers) == 1
    assert papers[0].source == "semanticscholar"
    assert papers[0].paper_id == "s2_abc123"
    assert papers[0].title == "S2 Paper"
    assert papers[0].authors == ["Carol", "Dave"]
    assert papers[0].pdf_path is None


@patch("digest_pipeline.s2_fetcher.requests.get")
def test_fetch_s2_papers_uses_query(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(
        semanticscholar_enabled=True,
        semanticscholar_query="reinforcement learning",
        semanticscholar_max_results=5,
    )
    fetch_s2_papers(settings)

    call_kwargs = mock_get.call_args
    assert call_kwargs.kwargs["params"]["query"] == "reinforcement learning"
    assert call_kwargs.kwargs["params"]["limit"] == 5


@patch("digest_pipeline.s2_fetcher.requests.get")
def test_fetch_s2_papers_sends_api_key(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(
        semanticscholar_enabled=True,
        semanticscholar_api_key="test-s2-key",
    )
    fetch_s2_papers(settings)

    call_kwargs = mock_get.call_args
    assert call_kwargs.kwargs["headers"]["x-api-key"] == "test-s2-key"


@patch("digest_pipeline.s2_fetcher.requests.get", side_effect=Exception("network error"))
def test_fetch_s2_papers_handles_error(mock_get, make_settings):
    settings = make_settings(semanticscholar_enabled=True)
    papers = fetch_s2_papers(settings)

    assert papers == []


@patch("digest_pipeline.s2_fetcher.requests.get")
def test_fetch_s2_papers_skips_missing_fields(mock_get, make_settings):
    """Papers without an ID or title are skipped."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {"paperId": "", "title": "No ID", "abstract": "x", "authors": [], "publicationDate": None, "url": "", "externalIds": {}, "openAccessPdf": None},
            {"paperId": "valid", "title": "", "abstract": "x", "authors": [], "publicationDate": None, "url": "", "externalIds": {}, "openAccessPdf": None},
            _make_s2_result(paper_id="good", title="Good Paper"),
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(semanticscholar_enabled=True)
    papers = fetch_s2_papers(settings)

    assert len(papers) == 1
    assert papers[0].paper_id == "s2_good"


@patch("digest_pipeline.s2_fetcher.requests.get")
def test_fetch_s2_papers_passes_fields_of_study(mock_get, make_settings):
    """fieldsOfStudy is included in params when configured."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(
        semanticscholar_enabled=True,
        semanticscholar_fields_of_study=["Computer Science", "Mathematics"],
    )
    fetch_s2_papers(settings)

    call_kwargs = mock_get.call_args
    assert call_kwargs.kwargs["params"]["fieldsOfStudy"] == "Computer Science,Mathematics"


@patch("digest_pipeline.s2_fetcher.requests.get")
def test_fetch_s2_papers_omits_fields_of_study_when_empty(mock_get, make_settings):
    """fieldsOfStudy is omitted from params when not configured."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = make_settings(semanticscholar_enabled=True)
    fetch_s2_papers(settings)

    call_kwargs = mock_get.call_args
    assert "fieldsOfStudy" not in call_kwargs.kwargs["params"]
