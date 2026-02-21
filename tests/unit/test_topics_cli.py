"""Unit tests for the topic browser CLI."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from digest_pipeline.topics_cli import (
    _cmd_group,
    _cmd_list,
    _cmd_search,
    _cmd_validate,
    console as cli_console,
)


@pytest.fixture(autouse=True)
def _capture_console(monkeypatch):
    """Redirect Rich console output to a StringIO for assertions."""
    buf = io.StringIO()
    captured = Console(file=buf, width=120, force_terminal=False)
    monkeypatch.setattr("digest_pipeline.topics_cli.console", captured)
    yield buf


class TestCmdList:
    def test_outputs_group_names(self, _capture_console):
        _cmd_list()
        output = _capture_console.getvalue()
        assert "cs" in output
        assert "math" in output
        assert "physics" in output
        assert "stat" in output


class TestCmdSearch:
    def test_finds_machine_learning(self, _capture_console):
        _cmd_search("machine learning")
        output = _capture_console.getvalue()
        assert "cs.LG" in output
        assert "Machine Learning" in output

    def test_nonexistent_query(self, _capture_console):
        _cmd_search("xyznonexistent999")
        output = _capture_console.getvalue()
        assert "No topics found" in output


class TestCmdGroup:
    def test_cs_group_shows_cs_ai(self, _capture_console):
        _cmd_group("cs")
        output = _capture_console.getvalue()
        assert "cs.AI" in output
        assert "Artificial Intelligence" in output

    def test_nonexistent_group(self, _capture_console):
        _cmd_group("nonexistent")
        output = _capture_console.getvalue()
        assert "Unknown group" in output


class TestCmdValidate:
    def test_all_valid(self, _capture_console):
        _cmd_validate(["cs.AI", "cs.LG"])
        output = _capture_console.getvalue()
        assert "All topic codes are valid" in output

    def test_with_invalid(self, _capture_console):
        with pytest.raises(SystemExit) as exc_info:
            _cmd_validate(["cs.AI", "fake.XX"])
        assert exc_info.value.code == 1
        output = _capture_console.getvalue()
        assert "fake.XX" in output
