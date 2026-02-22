"""Tests for the OpenAlex fetcher module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import requests

from digest_pipeline.openalex_fetcher import (
    OPENALEX_FIELDS,
    fetch_openalex_papers,
    reconstruct_abstract,
)


class TestReconstructAbstract:
    def test_basic_reconstruction(self):
        inv_index = {"Hello": [0], "world": [1]}
        assert reconstruct_abstract(inv_index) == "Hello world"

    def test_word_at_multiple_positions(self):
        inv_index = {"the": [0, 2], "cat": [1], "sat": [3]}
        assert reconstruct_abstract(inv_index) == "the cat the sat"

    def test_empty_index(self):
        assert reconstruct_abstract({}) == ""

    def test_none_input(self):
        assert reconstruct_abstract(None) == ""


_SENTINEL = object()


def _make_openalex_work(
    openalex_id="W123",
    title="OA Paper",
    abstract_words=_SENTINEL,
    pub_date=None,
    doi=None,
    topics=None,
    oa_url=None,
):
    """Build a fake OpenAlex work object."""
    if pub_date is None:
        pub_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if abstract_words is _SENTINEL:
        abstract_words = {"A": [0], "study": [1], "on": [2], "testing.": [3]}

    work = {
        "id": f"https://openalex.org/{openalex_id}",
        "title": title,
        "authorships": [
            {"author": {"display_name": "Alice"}},
            {"author": {"display_name": "Bob"}},
        ],
        "abstract_inverted_index": abstract_words,
        "publication_date": pub_date,
        "doi": f"https://doi.org/{doi}" if doi else None,
        "ids": {"openalex": f"https://openalex.org/{openalex_id}"},
        "open_access": {"oa_url": oa_url},
        "primary_location": {
            "source": {"display_name": "Test Journal"},
        },
        "topics": topics or [],
    }
    return work


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_success(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meta": {"count": 1},
        "results": [_make_openalex_work()],
    }
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_query="deep learning",
        openalex_max_results=10,
    )
    papers = fetch_openalex_papers(settings)

    assert len(papers) == 1
    assert papers[0].source == "openalex"
    assert papers[0].paper_id.startswith("oa_")
    assert papers[0].title == "OA Paper"
    assert papers[0].authors == ["Alice", "Bob"]
    assert papers[0].abstract == "A study on testing."
    assert papers[0].pdf_path is None


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_uses_query_and_filters(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_query="reinforcement learning",
        openalex_max_results=5,
        openalex_fields=["Computer Science"],
    )
    fetch_openalex_papers(settings)

    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params", {})
    assert "reinforcement learning" in params.get("search", "")
    assert "from_publication_date" in params.get("filter", "")


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_sends_api_key(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_api_key="test-oa-key",
    )
    fetch_openalex_papers(settings)

    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params", {})
    assert params.get("api_key") == "test-oa-key"


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_sends_email_in_user_agent(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_email="user@example.com",
    )
    fetch_openalex_papers(settings)

    call_args = mock_get.call_args
    headers = call_args.kwargs.get("headers") or call_args[1].get("headers", {})
    assert "user@example.com" in headers.get("User-Agent", "")


@patch("digest_pipeline.openalex_fetcher.time.sleep")
@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_handles_error(mock_get, mock_sleep, make_settings):
    mock_get.side_effect = Exception("network error")

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(settings)

    assert papers == []
    assert mock_get.call_count == 3  # retried 3 times


@patch("digest_pipeline.openalex_fetcher.time.sleep")
@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_retries_on_429(mock_get, mock_sleep, make_settings):
    """Retries on HTTP 429 and succeeds on second attempt."""
    error_resp = MagicMock()
    error_resp.status_code = 429
    error_resp.headers = {"Retry-After": "1"}
    error_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=error_resp)

    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    success_resp.raise_for_status = MagicMock()

    mock_get.side_effect = [error_resp, success_resp]

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(settings)

    assert papers == []  # no results, but didn't crash
    assert mock_get.call_count == 2


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_skips_no_abstract(mock_get, make_settings):
    """Papers without abstracts are skipped."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meta": {"count": 3},
        "results": [
            _make_openalex_work(openalex_id="W1", abstract_words=None),
            _make_openalex_work(openalex_id="W2", abstract_words={}),
            _make_openalex_work(openalex_id="W3", title="Good Paper"),
        ],
    }
    mock_get.return_value = mock_resp

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(settings)

    assert len(papers) == 1
    assert papers[0].title == "Good Paper"


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_deduplicates_known_ids(mock_get, make_settings):
    """Papers with DOIs already in known set are skipped."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meta": {"count": 2},
        "results": [
            _make_openalex_work(openalex_id="W1", doi="10.1234/dup"),
            _make_openalex_work(openalex_id="W2", title="New Paper", doi="10.1234/new"),
        ],
    }
    mock_get.return_value = mock_resp

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(
        settings,
        known_paper_ids={"10.1234/dup"},
    )

    assert len(papers) == 1
    assert papers[0].title == "New Paper"


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_extracts_topics(mock_get, make_settings):
    """Topic hierarchy from OpenAlex is stored in fields_of_study."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meta": {"count": 1},
        "results": [
            _make_openalex_work(
                topics=[
                    {
                        "display_name": "Neural Networks",
                        "subfield": {"display_name": "Artificial Intelligence"},
                        "field": {"display_name": "Computer Science"},
                        "score": 0.95,
                    },
                    {
                        "display_name": "Deep Learning",
                        "subfield": {"display_name": "Artificial Intelligence"},
                        "field": {"display_name": "Computer Science"},
                        "score": 0.80,
                    },
                ]
            ),
        ],
    }
    mock_get.return_value = mock_resp

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(settings)

    assert len(papers) == 1
    assert "Artificial Intelligence" in papers[0].fields_of_study


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_field_filter(mock_get, make_settings):
    """When openalex_fields is set, the filter includes field IDs."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_fields=["Computer Science", "Mathematics"],
    )
    fetch_openalex_papers(settings)

    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params", {})
    filter_str = params.get("filter", "")
    assert "primary_topic.field.id" in filter_str


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_skips_no_title(mock_get, make_settings):
    """Papers without titles are skipped."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meta": {"count": 2},
        "results": [
            _make_openalex_work(openalex_id="W1", title=""),
            _make_openalex_work(openalex_id="W2", title="Good Paper"),
        ],
    }
    mock_get.return_value = mock_resp

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(settings)

    assert len(papers) == 1
    assert papers[0].title == "Good Paper"


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_skips_no_id(mock_get, make_settings):
    """Papers without an OpenAlex ID are skipped."""
    no_id_work = _make_openalex_work(openalex_id="W1")
    no_id_work["id"] = ""
    good_work = _make_openalex_work(openalex_id="W2", title="Good Paper")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meta": {"count": 2},
        "results": [no_id_work, good_work],
    }
    mock_get.return_value = mock_resp

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(settings)

    assert len(papers) == 1
    assert papers[0].title == "Good Paper"


def test_openalex_fields_has_26_entries():
    """Verify all 26 OpenAlex fields are mapped."""
    assert len(OPENALEX_FIELDS) == 26
