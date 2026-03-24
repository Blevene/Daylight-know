"""Shared pytest configuration, markers, and fixtures."""

import pytest
from datetime import datetime, timezone

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, no external dependencies")
    config.addinivalue_line(
        "markers", "integration: real local dependencies (ChromaDB, Chonkie, pypdf)"
    )
    config.addinivalue_line("markers", "e2e: full pipeline runs")
    config.addinivalue_line("markers", "network: requires internet access (arXiv, GitHub)")


@pytest.fixture
def make_paper():
    """Factory fixture for building Paper with sensible test defaults."""

    def _factory(**overrides) -> Paper:
        defaults = dict(
            paper_id="2401.00001",
            title="Test Paper",
            authors=["Alice", "Bob"],
            abstract="This paper explores testing.",
            url="https://arxiv.org/abs/2401.00001",
            published=datetime(2025, 1, 15, tzinfo=timezone.utc),
            source="arxiv",
            pdf_path=None,
        )
        defaults.update(overrides)
        return Paper(**defaults)

    return _factory


@pytest.fixture
def make_settings():
    """Factory fixture for building Settings with sensible test defaults."""

    def _factory(**overrides) -> Settings:
        defaults = dict(
            _env_file=None,
            llm_api_key="test-key",
            smtp_user="u",
            smtp_password="p",
            email_from="a@b.com",
            email_to="c@d.com",
        )
        defaults.update(overrides)
        return Settings(**defaults)

    return _factory
