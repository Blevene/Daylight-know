"""Tests for the post-processing module (implications & critiques)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper
from digest_pipeline.postprocessor import (
    CRITIQUES_SYSTEM_PROMPT,
    IMPLICATIONS_SYSTEM_PROMPT,
    _build_user_prompt,
    extract_implications,
    generate_critiques,
)


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        arxiv_id="2401.00001",
        title="Test Paper",
        authors=["Alice", "Bob"],
        abstract="This paper explores testing.",
        url="https://arxiv.org/abs/2401.00001",
        published=datetime(2025, 1, 15, tzinfo=timezone.utc),
        pdf_path=None,
    )
    defaults.update(overrides)
    return Paper(**defaults)


def _make_settings(**overrides) -> Settings:
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


def test_build_user_prompt_single_paper():
    papers = [_make_paper()]
    prompt = _build_user_prompt(papers)
    assert "Test Paper" in prompt
    assert "Alice, Bob" in prompt
    assert "This paper explores testing." in prompt


def test_build_user_prompt_multiple_papers():
    papers = [_make_paper(), _make_paper(title="Second Paper")]
    prompt = _build_user_prompt(papers)
    assert "Test Paper" in prompt
    assert "Second Paper" in prompt
    assert "---" in prompt


@patch("digest_pipeline.postprocessor.litellm.completion")
def test_extract_implications_success(mock_completion):
    mock_choice = MagicMock()
    mock_choice.message.content = "Practitioners can apply these findings by..."
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    settings = _make_settings()
    result = extract_implications([_make_paper()], settings)

    assert result == "Practitioners can apply these findings by..."
    mock_completion.assert_called_once()

    call_kwargs = mock_completion.call_args
    messages = call_kwargs.kwargs["messages"]
    assert messages[0]["content"] == IMPLICATIONS_SYSTEM_PROMPT


@patch("digest_pipeline.postprocessor.litellm.completion")
def test_generate_critiques_success(mock_completion):
    mock_choice = MagicMock()
    mock_choice.message.content = "The methodology has several strengths..."
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    settings = _make_settings()
    result = generate_critiques([_make_paper()], settings)

    assert result == "The methodology has several strengths..."
    mock_completion.assert_called_once()

    call_kwargs = mock_completion.call_args
    messages = call_kwargs.kwargs["messages"]
    assert messages[0]["content"] == CRITIQUES_SYSTEM_PROMPT


@patch("digest_pipeline.postprocessor.litellm.completion")
def test_llm_call_respects_max_tokens(mock_completion):
    mock_choice = MagicMock()
    mock_choice.message.content = "result"
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    settings = _make_settings(llm_max_tokens=2048)
    extract_implications([_make_paper()], settings)

    call_kwargs = mock_completion.call_args
    assert call_kwargs.kwargs["max_tokens"] == 2048


@patch("digest_pipeline.postprocessor.litellm.completion")
def test_llm_call_empty_content_returns_empty_string(mock_completion):
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    settings = _make_settings()
    result = generate_critiques([_make_paper()], settings)
    assert result == ""
