"""Tests for the LLM summarization module."""

import json
from unittest.mock import MagicMock, patch

from digest_pipeline.summarizer import summarize


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_summarize_success(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"paper_1": "This is a summary."})
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    result = summarize([make_paper()], make_settings())
    assert result == {"paper_1": "This is a summary."}
    mock_completion.assert_called_once()

    call_kwargs = mock_completion.call_args
    assert "response_format" in call_kwargs.kwargs
    rf = call_kwargs.kwargs["response_format"]
    assert rf["type"] == "json_schema"
    # Schema should have explicit paper_1 key
    schema = rf["json_schema"]["schema"]
    assert "paper_1" in schema["properties"]
    assert schema["additionalProperties"] is False


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_summarize_with_github_section(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"paper_1": "Summary."})
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    summarize([make_paper()], make_settings(), github_section="Repo1\nRepo2")

    call_kwargs = mock_completion.call_args
    user_msg = call_kwargs.kwargs["messages"][1]["content"]
    assert "Trending GitHub Repositories" in user_msg


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_summarize_empty_content_returns_empty_dict(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    result = summarize([make_paper()], make_settings())
    assert result == {}


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_summarize_malformed_json_returns_empty_dict(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = "This is not valid JSON at all"
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    result = summarize([make_paper()], make_settings())
    assert result == {}


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_summarize_non_object_json_returns_empty_dict(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = '["an", "array"]'
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    result = summarize([make_paper()], make_settings())
    assert result == {}


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_summarize_multiple_papers_schema(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "paper_1": "Summary 1",
        "paper_2": "Summary 2",
    })
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    papers = [make_paper(), make_paper(title="Second Paper")]
    result = summarize(papers, make_settings())
    assert result == {"paper_1": "Summary 1", "paper_2": "Summary 2"}

    # Schema should enumerate both keys
    rf = mock_completion.call_args.kwargs["response_format"]
    schema = rf["json_schema"]["schema"]
    assert "paper_1" in schema["properties"]
    assert "paper_2" in schema["properties"]
    assert schema["required"] == ["paper_1", "paper_2"]
