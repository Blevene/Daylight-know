"""Integration tests for config.py — real .env loading and env var overrides.

Test IDs: CFG-1, CFG-2, CFG-3, CFG-4, CFG-5
"""

import os

import pytest

from digest_pipeline.config import Settings


@pytest.mark.integration
class TestConfigIntegration:
    """Tests that exercise real Settings behavior with env files and vars."""

    def test_env_file_loading(self, tmp_path):
        """CFG-1: Create a temp .env file with overrides, verify values are read."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "LLM_API_KEY=from-env-file\n"
            "DRY_RUN=false\n"
            "ARXIV_MAX_RESULTS=99\n"
            "SMTP_USER=envuser\n"
            "SMTP_PASSWORD=envpass\n"
            "EMAIL_FROM=env@test.com\n"
            "EMAIL_TO=envto@test.com\n"
        )
        s = Settings(_env_file=str(env_file))

        assert s.llm_api_key == "from-env-file"
        assert s.dry_run is False
        assert s.arxiv_max_results == 99

    def test_environment_variable_override(self, monkeypatch):
        """CFG-2: os.environ overrides take effect in Settings."""
        monkeypatch.setenv("ARXIV_MAX_RESULTS", "10")
        monkeypatch.setenv("LLM_API_KEY", "env-override-key")
        monkeypatch.setenv("SMTP_USER", "u")
        monkeypatch.setenv("SMTP_PASSWORD", "p")
        monkeypatch.setenv("EMAIL_FROM", "a@b.com")
        monkeypatch.setenv("EMAIL_TO", "c@d.com")

        s = Settings(_env_file=None)
        assert s.arxiv_max_results == 10

    def test_list_parsing(self, monkeypatch):
        """CFG-3: ARXIV_TOPICS parsed into a list."""
        monkeypatch.setenv("ARXIV_TOPICS", '["cs.AI","cs.CL","stat.ML"]')
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("SMTP_USER", "u")
        monkeypatch.setenv("SMTP_PASSWORD", "p")
        monkeypatch.setenv("EMAIL_FROM", "a@b.com")
        monkeypatch.setenv("EMAIL_TO", "c@d.com")

        s = Settings(_env_file=None)
        assert s.arxiv_topics == ["cs.AI", "cs.CL", "stat.ML"]

    def test_litellm_model_string(self, monkeypatch):
        """CFG-4: LLM_MODEL accepts provider/model format."""
        monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("SMTP_USER", "u")
        monkeypatch.setenv("SMTP_PASSWORD", "p")
        monkeypatch.setenv("EMAIL_FROM", "a@b.com")
        monkeypatch.setenv("EMAIL_TO", "c@d.com")

        s = Settings(_env_file=None)
        assert s.llm_model == "anthropic/claude-sonnet-4-20250514"

    def test_llm_api_base_nullable(self):
        """CFG-5: llm_api_base defaults to None and accepts a URL when set."""
        s = Settings(
            _env_file=None,
            llm_api_key="k",
            smtp_user="u",
            smtp_password="p",
            email_from="a@b.com",
            email_to="c@d.com",
        )
        assert s.llm_api_base is None

        s2 = Settings(
            _env_file=None,
            llm_api_key="k",
            llm_api_base="http://localhost:8080/v1",
            smtp_user="u",
            smtp_password="p",
            email_from="a@b.com",
            email_to="c@d.com",
        )
        assert s2.llm_api_base == "http://localhost:8080/v1"
