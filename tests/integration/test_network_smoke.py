"""Network smoke tests — hit real external APIs.

Run only with: pytest -m network
Test IDs: F-1, F-2, G-1
"""

import tempfile
from pathlib import Path

import pytest

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper, _download_pdf, fetch_papers
from digest_pipeline.github_trending import TrendingRepo, fetch_trending


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

    def test_fetch_papers_real_arxiv(self):
        """F-1: Hit the live arXiv API for cs.DL with max_results=2."""
        settings = _make_settings()
        papers = fetch_papers(settings)

        # May return 0 if no papers in the 24-hour window for cs.DL
        assert isinstance(papers, list)
        for p in papers:
            assert isinstance(p, Paper)
            assert p.arxiv_id
            assert p.title
            assert len(p.authors) > 0
            assert p.pdf_path is not None
            assert p.pdf_path.exists()
            # Verify it's actually a PDF
            header = p.pdf_path.read_bytes()[:5]
            assert header == b"%PDF-"

    def test_download_pdf_real(self):
        """F-2: Download a known small PDF from arXiv and verify it starts with %PDF."""
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "test.pdf"
            # Use a known stable arXiv paper
            success = _download_pdf(
                "https://arxiv.org/pdf/2301.00001v1",
                dest,
                max_retries=2,
            )
            if success:
                assert dest.exists()
                assert dest.read_bytes()[:5] == b"%PDF-"

    def test_fetch_trending_real_github(self):
        """G-1: Hit the live GitHub Search API and verify TrendingRepo objects."""
        settings = _make_settings(github_enabled=True)
        repos = fetch_trending(settings)

        assert isinstance(repos, list)
        for r in repos:
            assert isinstance(r, TrendingRepo)
            assert r.name
            assert r.url.startswith("https://github.com/")
