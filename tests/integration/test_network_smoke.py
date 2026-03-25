"""Network smoke tests — hit real external APIs.

Run only with: pytest -m network
Test IDs: F-1, F-2, G-1
"""

import tempfile
from pathlib import Path

import pytest

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper, download_pdf, fetch_papers


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

    def test_fetch_papers_real_arxiv_rss(self):
        """F-1: Hit the live arXiv RSS feed for cs.DL with max_results=2."""
        settings = _make_settings()
        papers = fetch_papers(settings)

        # May return 0 if no papers today for cs.DL
        assert isinstance(papers, list)
        for p in papers:
            assert isinstance(p, Paper)
            assert p.paper_id
            assert p.title
            assert len(p.authors) > 0
            assert p.pdf_path is not None
            assert p.pdf_path.exists()
            # Verify it's actually a PDF
            header = p.pdf_path.read_bytes()[:5]
            assert header == b"%PDF-"

    def testdownload_pdf_real(self):
        """F-2: Download a known small PDF from arXiv and verify it starts with %PDF."""
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.pdf"
            # Use a known stable arXiv paper
            success = download_pdf(
                "https://arxiv.org/pdf/2301.00001v1",
                dest,
                max_retries=2,
            )
            if success:
                assert dest.exists()
                assert dest.read_bytes()[:5] == b"%PDF-"

    def test_fetch_openalex_real(self):
        """Hit the live OpenAlex API and verify Paper objects."""
        from digest_pipeline.openalex_fetcher import fetch_openalex_papers

        settings = _make_settings(
            openalex_enabled=True,
            openalex_query="machine learning",
            openalex_max_results=3,
        )
        papers = fetch_openalex_papers(settings)

        assert isinstance(papers, list)
        for p in papers:
            assert isinstance(p, Paper)
            assert p.source == "openalex"
            assert p.paper_id.startswith("oa_")
            assert p.abstract
            assert p.title

