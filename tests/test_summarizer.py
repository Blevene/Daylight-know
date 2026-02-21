"""Tests for the LLM summarization module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper
from digest_pipeline.summarizer import _build_user_prompt, summarize


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


def test_build_user_prompt():
    papers = [_make_paper(), _make_paper(title="Second Paper")]
    prompt = _build_user_prompt(papers)
    assert "Test Paper" in prompt
    assert "Second Paper" in prompt


def test_build_user_prompt_with_github():
    papers = [_make_paper()]
    prompt = _build_user_prompt(papers, github_section="Repo1\nRepo2")
    assert "Trending GitHub Repositories" in prompt


@patch("digest_pipeline.summarizer.OpenAI")
def test_summarize_success(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a summary."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    settings = _make_settings()
    result = summarize([_make_paper()], settings)
    assert result == "This is a summary."
    mock_client.chat.completions.create.assert_called_once()
