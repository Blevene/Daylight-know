"""Tests for the post-processing module (implications & critiques)."""

import json
from unittest.mock import MagicMock, patch

from digest_pipeline.postprocessor import (
    CRITIQUES_SYSTEM_PROMPT,
    IMPLICATIONS_SYSTEM_PROMPT,
    extract_implications,
    generate_critiques,
)


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_extract_implications_success(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"paper_1": "Practitioners can apply..."})
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    result = extract_implications([make_paper()], make_settings())

    assert result == {"paper_1": "Practitioners can apply..."}
    mock_completion.assert_called_once()

    call_kwargs = mock_completion.call_args
    messages = call_kwargs.kwargs["messages"]
    assert messages[0]["content"] == IMPLICATIONS_SYSTEM_PROMPT
    assert "response_format" in call_kwargs.kwargs


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_generate_critiques_success(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"paper_1": "The methodology has strengths..."})
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    result = generate_critiques([make_paper()], make_settings())

    assert result == {"paper_1": "The methodology has strengths..."}
    mock_completion.assert_called_once()

    call_kwargs = mock_completion.call_args
    messages = call_kwargs.kwargs["messages"]
    assert messages[0]["content"] == CRITIQUES_SYSTEM_PROMPT
    assert "response_format" in call_kwargs.kwargs


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_llm_call_respects_max_tokens(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"paper_1": "result"})
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    extract_implications([make_paper()], make_settings(llm_max_tokens=2048))

    call_kwargs = mock_completion.call_args
    assert call_kwargs.kwargs["max_tokens"] == 2048


@patch("digest_pipeline.llm_utils.time.sleep")
@patch("digest_pipeline.llm_utils.litellm.completion")
def test_llm_call_empty_content_returns_empty_dict(
    mock_completion, _mock_sleep, make_paper, make_settings
):
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    result = generate_critiques([make_paper()], make_settings())
    assert result == {}


@patch("digest_pipeline.llm_utils.time.sleep")
@patch("digest_pipeline.llm_utils.litellm.completion")
def test_llm_call_malformed_json_returns_empty_dict(
    mock_completion, _mock_sleep, make_paper, make_settings
):
    mock_choice = MagicMock()
    mock_choice.message.content = "Not JSON at all."
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    result = extract_implications([make_paper()], make_settings())
    assert result == {}


@patch("digest_pipeline.llm_utils.litellm.completion")
def test_schema_has_explicit_keys(mock_completion, make_paper, make_settings):
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"paper_1": "a", "paper_2": "b"})
    mock_completion.return_value = MagicMock(choices=[mock_choice])

    papers = [make_paper(), make_paper(title="Second")]
    extract_implications(papers, make_settings())

    rf = mock_completion.call_args.kwargs["response_format"]
    schema = rf["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert "paper_1" in schema["properties"]
    assert "paper_2" in schema["properties"]
