"""Tests for the prompt loading module."""

import pytest

from digest_pipeline.prompts import load_prompt


def test_load_summarizer_prompt():
    prompt = load_prompt("summarizer")
    assert "expert research assistant" in prompt
    assert len(prompt) > 20


def test_load_implications_prompt():
    prompt = load_prompt("implications")
    assert "actionable insights" in prompt


def test_load_critiques_prompt():
    prompt = load_prompt("critiques")
    assert "peer reviewer" in prompt


def test_prompts_are_pure_prose():
    """Prompts should not contain JSON formatting instructions."""
    for name in ("summarizer", "implications", "critiques"):
        prompt = load_prompt(name)
        assert "Return your response as a JSON" not in prompt
        assert "Return ONLY valid JSON" not in prompt


def test_load_missing_prompt_raises():
    with pytest.raises(FileNotFoundError, match="Available prompts"):
        load_prompt("nonexistent_prompt")
