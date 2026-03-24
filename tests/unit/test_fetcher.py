"""Tests for the arXiv fetcher module."""

from unittest.mock import MagicMock, patch

import pytest

from digest_pipeline.fetcher import (
    _extract_abstract,
    _parse_arxiv_id,
    download_pdf,
    fetch_papers,
)


# ── download_pdf ─────────────────────────────────────────────────


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


# ── _extract_abstract ────────────────────────────────────────────


def test_extract_abstract_with_announce_type():
    desc = "arXiv:2603.12345v1 Announce Type: new \n Abstract: This paper studies transformers."
    assert _extract_abstract(desc) == "This paper studies transformers."


def test_extract_abstract_replace_cross():
    desc = "arXiv:2203.01250v3 Announce Type: replace-cross \n Abstract: We study concentration."
    assert _extract_abstract(desc) == "We study concentration."


def test_extract_abstract_empty():
    assert _extract_abstract("") == ""


def test_extract_abstract_no_abstract_marker():
    assert _extract_abstract("Some random text without abstract marker") == ""


# ── _parse_arxiv_id ──────────────────────────────────────────────


def test_parse_arxiv_id_standard():
    assert _parse_arxiv_id("https://arxiv.org/abs/2603.12345") == "2603.12345"


def test_parse_arxiv_id_with_version():
    assert _parse_arxiv_id("https://arxiv.org/abs/2603.12345v2") == "2603.12345v2"


def test_parse_arxiv_id_trailing_slash():
    assert _parse_arxiv_id("https://arxiv.org/abs/2603.12345/") == "2603.12345"


# ── fetch_papers (RSS-based) ─────────────────────────────────────

_SAMPLE_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:arxiv="http://arxiv.org/schemas/atom"
     version="2.0">
  <channel>
    <title>cs.AI updates on arXiv.org</title>
    <link>http://rss.arxiv.org/rss/cs.AI</link>
    <item>
      <title>Test Paper Title</title>
      <link>https://arxiv.org/abs/2603.00001</link>
      <description>arXiv:2603.00001v1 Announce Type: new
Abstract: This is a test abstract about transformers.</description>
      <guid isPermaLink="false">oai:arXiv.org:2603.00001v1</guid>
      <category>cs.AI</category>
      <dc:creator>Alice Smith, Bob Jones</dc:creator>
      <arxiv:announce_type>new</arxiv:announce_type>
    </item>
    <item>
      <title>Another Paper</title>
      <link>https://arxiv.org/abs/2603.00002</link>
      <description>arXiv:2603.00002v1 Announce Type: new
Abstract: Second paper abstract.</description>
      <guid isPermaLink="false">oai:arXiv.org:2603.00002v1</guid>
      <category>cs.LG</category>
      <category>cs.AI</category>
      <dc:creator>Charlie Brown</dc:creator>
      <arxiv:announce_type>new</arxiv:announce_type>
    </item>
  </channel>
</rss>
"""


@patch("digest_pipeline.fetcher.download_pdf", return_value=True)
@patch("digest_pipeline.fetcher.requests.get")
def test_fetch_papers_rss(mock_get, mock_dl):
    """fetch_papers should parse RSS 2.0 XML and return Paper objects."""
    mock_resp = MagicMock()
    mock_resp.content = _SAMPLE_RSS.encode()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    from digest_pipeline.config import Settings

    settings = Settings(
        _env_file=None,
        arxiv_topics=["cs.AI", "cs.LG"],
        arxiv_max_results=10,
        llm_api_key="k",
        smtp_user="u",
        email_from="a@b.com",
        email_to="c@d.com",
    )
    papers = fetch_papers(settings)

    assert len(papers) == 2
    assert papers[0].paper_id == "2603.00001"
    assert papers[0].title == "Test Paper Title"
    assert papers[0].authors == ["Alice Smith", "Bob Jones"]
    assert papers[0].abstract == "This is a test abstract about transformers."
    assert papers[0].source == "arxiv"
    assert papers[0].categories == ["cs.AI"]
    assert papers[1].paper_id == "2603.00002"
    assert papers[1].categories == ["cs.LG", "cs.AI"]


@patch("digest_pipeline.fetcher.download_pdf", return_value=True)
@patch("digest_pipeline.fetcher.requests.get")
def test_fetch_papers_respects_max_results(mock_get, mock_dl):
    """fetch_papers should stop after max_results papers."""
    mock_resp = MagicMock()
    mock_resp.content = _SAMPLE_RSS.encode()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    from digest_pipeline.config import Settings

    settings = Settings(
        _env_file=None,
        arxiv_topics=["cs.AI"],
        arxiv_max_results=1,
        llm_api_key="k",
        smtp_user="u",
        email_from="a@b.com",
        email_to="c@d.com",
    )
    papers = fetch_papers(settings)

    assert len(papers) == 1
    assert papers[0].paper_id == "2603.00001"


@patch("digest_pipeline.fetcher.requests.get")
def test_fetch_papers_empty_feed(mock_get):
    """fetch_papers should return empty list for feed with no items."""
    empty_rss = (
        '<?xml version="1.0"?>'
        '<rss version="2.0"><channel><title>test</title></channel></rss>'
    )
    mock_resp = MagicMock()
    mock_resp.content = empty_rss.encode()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    from digest_pipeline.config import Settings

    settings = Settings(
        _env_file=None,
        arxiv_topics=["cs.AI"],
        llm_api_key="k",
        smtp_user="u",
        email_from="a@b.com",
        email_to="c@d.com",
    )
    papers = fetch_papers(settings)
    assert papers == []


@patch("digest_pipeline.fetcher.requests.get", side_effect=Exception("connection refused"))
def test_fetch_papers_network_error(mock_get):
    """fetch_papers should propagate network errors."""
    from digest_pipeline.config import Settings

    settings = Settings(
        _env_file=None,
        arxiv_topics=["cs.AI"],
        llm_api_key="k",
        smtp_user="u",
        email_from="a@b.com",
        email_to="c@d.com",
    )
    with pytest.raises(Exception, match="connection refused"):
        fetch_papers(settings)
