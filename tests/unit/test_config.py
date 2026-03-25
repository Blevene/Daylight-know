"""Tests for the configuration module."""

from digest_pipeline.config import Settings, get_settings


def test_defaults():
    s = Settings(
        _env_file=None,
        llm_api_key="test-key",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b.com",
        email_to="c@d.com",
    )
    assert s.arxiv_topics == ["cs.AI", "cs.LG"]
    assert s.dry_run is True
    assert s.pdf_download_max_retries == 3


def test_get_settings_returns_instance():
    s = get_settings()
    assert isinstance(s, Settings)
