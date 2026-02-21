"""Shared fixtures for unit tests."""

import pytest

from digest_pipeline.config import Settings


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
            dry_run=True,
            github_enabled=False,
        )
        defaults.update(overrides)
        return Settings(**defaults)

    return _factory
