"""Unit tests for the setup wizard."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from digest_pipeline.setup import (
    _collect_arxiv_topics,
    _handle_existing_env,
    _prompt,
    _prompt_bool,
    _prompt_choice,
    _read_existing_env,
    _test_llm_connection,
    _test_smtp_connection,
    _write_env_file,
)


@pytest.fixture(autouse=True)
def _capture_console(monkeypatch):
    """Redirect Rich console output to a StringIO."""
    buf = io.StringIO()
    captured = Console(file=buf, width=120, force_terminal=False)
    monkeypatch.setattr("digest_pipeline.setup.console", captured)
    yield buf


class TestPrompt:
    def test_returns_user_input(self, monkeypatch):
        monkeypatch.setattr("digest_pipeline.setup.console.input", lambda *a, **kw: "hello")
        result = _prompt("Label")
        assert result == "hello"

    def test_returns_default_on_empty(self, monkeypatch):
        monkeypatch.setattr("digest_pipeline.setup.console.input", lambda *a, **kw: "")
        result = _prompt("Label", default="fallback")
        assert result == "fallback"


class TestPromptBool:
    def test_yes(self, monkeypatch):
        monkeypatch.setattr("digest_pipeline.setup.console.input", lambda *a, **kw: "y")
        assert _prompt_bool("Continue?") is True

    def test_no(self, monkeypatch):
        monkeypatch.setattr("digest_pipeline.setup.console.input", lambda *a, **kw: "n")
        assert _prompt_bool("Continue?") is False

    def test_default_true(self, monkeypatch):
        monkeypatch.setattr("digest_pipeline.setup.console.input", lambda *a, **kw: "")
        assert _prompt_bool("Continue?", default=True) is True

    def test_default_false(self, monkeypatch):
        monkeypatch.setattr("digest_pipeline.setup.console.input", lambda *a, **kw: "")
        assert _prompt_bool("Continue?", default=False) is False


class TestPromptChoice:
    def test_select_by_number(self, monkeypatch):
        monkeypatch.setattr("digest_pipeline.setup.console.input", lambda *a, **kw: "2")
        result = _prompt_choice("Pick one", ["alpha", "beta", "gamma"])
        assert result == "beta"

    def test_returns_default_on_empty(self, monkeypatch):
        monkeypatch.setattr("digest_pipeline.setup.console.input", lambda *a, **kw: "")
        result = _prompt_choice("Pick one", ["alpha", "beta"], default="alpha")
        assert result == "alpha"

    def test_raw_value_fallback(self, monkeypatch):
        monkeypatch.setattr("digest_pipeline.setup.console.input", lambda *a, **kw: "custom")
        result = _prompt_choice("Pick one", ["alpha", "beta"])
        assert result == "custom"

    def test_out_of_range_returns_raw(self, monkeypatch):
        monkeypatch.setattr("digest_pipeline.setup.console.input", lambda *a, **kw: "99")
        result = _prompt_choice("Pick one", ["alpha", "beta"])
        assert result == "99"


class TestCollectArxivTopics:
    def test_manual_entry(self, monkeypatch):
        """Type codes manually then done."""
        inputs = iter(["t", "cs.AI, cs.LG", "d"])
        monkeypatch.setattr(
            "digest_pipeline.setup.console.input", lambda *a, **kw: next(inputs)
        )
        result = _collect_arxiv_topics()
        assert "cs.AI" in result
        assert "cs.LG" in result

    def test_rejects_invalid_and_continues(self, monkeypatch):
        """Invalid codes are rejected, valid ones kept."""
        inputs = iter(["t", "fake.XX, cs.AI", "d"])
        monkeypatch.setattr(
            "digest_pipeline.setup.console.input", lambda *a, **kw: next(inputs)
        )
        result = _collect_arxiv_topics()
        assert "cs.AI" in result
        assert "fake.XX" not in result


class TestSmtpConnection:
    @patch("digest_pipeline.setup.smtplib.SMTP")
    def test_success(self, mock_smtp_cls, _capture_console):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_server
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        config = {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "user@test.com",
            "SMTP_PASSWORD": "pass",
        }
        result = _test_smtp_connection(config)
        assert result is True
        output = _capture_console.getvalue()
        assert "successful" in output

    @patch("digest_pipeline.setup.smtplib.SMTP")
    def test_failure(self, mock_smtp_cls, _capture_console):
        mock_smtp_cls.side_effect = ConnectionRefusedError("refused")

        config = {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "user@test.com",
            "SMTP_PASSWORD": "pass",
        }
        result = _test_smtp_connection(config)
        assert result is False
        output = _capture_console.getvalue()
        assert "failed" in output


class TestLlmConnection:
    @patch("litellm.completion")
    def test_success(self, mock_completion, _capture_console):
        mock_completion.return_value = MagicMock()

        config = {"LLM_MODEL": "openai/gpt-4o-mini", "LLM_API_KEY": "sk-test"}
        result = _test_llm_connection(config)
        assert result is True
        output = _capture_console.getvalue()
        assert "successful" in output

    @patch("litellm.completion")
    def test_failure(self, mock_completion, _capture_console):
        mock_completion.side_effect = Exception("auth error")

        config = {"LLM_MODEL": "openai/gpt-4o-mini", "LLM_API_KEY": "bad-key"}
        result = _test_llm_connection(config)
        assert result is False
        output = _capture_console.getvalue()
        assert "failed" in output


class TestEnvFile:
    def test_write_creates_correct_content(self, tmp_path):
        path = tmp_path / ".env"
        config = {
            "ARXIV_TOPICS": "cs.AI,cs.LG",
            "ARXIV_MAX_RESULTS": "50",
            "LLM_MODEL": "openai/gpt-4o-mini",
            "LLM_API_KEY": "sk-test",
        }
        _write_env_file(config, path)

        content = path.read_text()
        assert 'ARXIV_TOPICS="cs.AI,cs.LG"' in content
        assert 'LLM_MODEL="openai/gpt-4o-mini"' in content
        assert "# ── arXiv Settings" in content

    def test_write_quotes_special_characters(self, tmp_path):
        path = tmp_path / ".env"
        config = {"SMTP_PASSWORD": "p@ss=w#rd with spaces"}
        _write_env_file(config, path)

        content = path.read_text()
        assert 'SMTP_PASSWORD="p@ss=w#rd with spaces"' in content

    def test_write_remaining_keys(self, tmp_path):
        path = tmp_path / ".env"
        config = {"CUSTOM_KEY": "custom_value"}
        _write_env_file(config, path)

        content = path.read_text()
        assert "# ── Other Settings" in content
        assert 'CUSTOM_KEY="custom_value"' in content

    def test_read_existing_parses_key_value(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("# comment\nFOO=bar\nBAZ=qux\n\n")

        result = _read_existing_env(path)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_read_existing_strips_quotes(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text('FOO="bar"\nBAZ=\'qux\'\n')

        result = _read_existing_env(path)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_read_existing_preserves_equals_in_value(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text("KEY=value=with=equals\n")

        result = _read_existing_env(path)
        assert result == {"KEY": "value=with=equals"}

    def test_read_nonexistent(self, tmp_path):
        path = tmp_path / "missing.env"
        result = _read_existing_env(path)
        assert result == {}


class TestHandleExistingEnv:
    def test_no_existing_file_returns_new_config(self, tmp_path):
        path = tmp_path / ".env"
        new_config = {"KEY": "value"}
        result = _handle_existing_env(path, new_config)
        assert result == new_config

    def test_overwrite_creates_backup(self, tmp_path, monkeypatch):
        path = tmp_path / ".env"
        path.write_text("OLD_KEY=old_value\n")

        # Choose "Overwrite"
        monkeypatch.setattr(
            "digest_pipeline.setup.console.input", lambda *a, **kw: "1"
        )
        new_config = {"NEW_KEY": "new_value"}
        result = _handle_existing_env(path, new_config)

        # Should return new config only
        assert result == new_config
        # Backup should exist
        backups = list(tmp_path.glob(".env.bak.*"))
        assert len(backups) == 1
        assert "OLD_KEY=old_value" in backups[0].read_text()

    def test_merge_preserves_existing_keys(self, tmp_path, monkeypatch):
        path = tmp_path / ".env"
        path.write_text("EXISTING_KEY=keep_me\nARXIV_TOPICS=old\n")

        # Choose "Merge"
        monkeypatch.setattr(
            "digest_pipeline.setup.console.input", lambda *a, **kw: "2"
        )
        new_config = {"ARXIV_TOPICS": "cs.AI", "LLM_MODEL": "test"}
        result = _handle_existing_env(path, new_config)

        assert result["EXISTING_KEY"] == "keep_me"
        assert result["ARXIV_TOPICS"] == "cs.AI"
        assert result["LLM_MODEL"] == "test"

    def test_multiple_backups_dont_overwrite(self, tmp_path, monkeypatch):
        path = tmp_path / ".env"
        path.write_text("KEY=v1\n")

        # Create a pre-existing backup
        (tmp_path / ".env.bak.20260101T000000").write_text("KEY=v0\n")

        monkeypatch.setattr(
            "digest_pipeline.setup.console.input", lambda *a, **kw: "1"
        )
        _handle_existing_env(path, {"KEY": "v2"})

        # Should now have 2 backups (old one preserved)
        backups = list(tmp_path.glob(".env.bak.*"))
        assert len(backups) == 2
