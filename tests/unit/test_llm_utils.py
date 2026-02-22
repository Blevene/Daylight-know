"""Tests for the shared LLM call utilities."""

import json
from unittest.mock import MagicMock, patch

import pytest

from digest_pipeline.llm_utils import build_response_format, build_user_prompt, llm_call


class TestBuildResponseFormat:
    def test_single_paper(self):
        rf = build_response_format("test", 1)
        assert rf["type"] == "json_schema"
        schema = rf["json_schema"]["schema"]
        assert schema["properties"] == {"paper_1": {"type": "string"}}
        assert schema["required"] == ["paper_1"]
        assert schema["additionalProperties"] is False

    def test_multiple_papers(self):
        rf = build_response_format("test", 3)
        schema = rf["json_schema"]["schema"]
        assert set(schema["properties"].keys()) == {"paper_1", "paper_2", "paper_3"}
        assert len(schema["required"]) == 3

    def test_strict_mode_enabled(self):
        rf = build_response_format("test", 1)
        assert rf["json_schema"]["strict"] is True

    def test_schema_name_passed_through(self):
        rf = build_response_format("my_schema", 1)
        assert rf["json_schema"]["name"] == "my_schema"


class TestBuildUserPrompt:
    def test_single_paper(self, make_paper):
        prompt = build_user_prompt([make_paper()])
        assert "Test Paper" in prompt
        assert "Alice, Bob" in prompt
        assert "This paper explores testing." in prompt

    def test_multiple_papers(self, make_paper):
        papers = [make_paper(), make_paper(title="Second Paper")]
        prompt = build_user_prompt(papers)
        assert "Test Paper" in prompt
        assert "Second Paper" in prompt
        assert "---" in prompt

    def test_includes_source(self, make_paper):
        prompt = build_user_prompt([make_paper(source="openalex")])
        assert "**Source:** openalex" in prompt

    def test_includes_categories(self, make_paper):
        prompt = build_user_prompt([make_paper(categories=["cs.AI", "cs.LG"])])
        assert "**Categories:** cs.AI, cs.LG" in prompt

    def test_includes_fields_of_study(self, make_paper):
        prompt = build_user_prompt([make_paper(fields_of_study=["Computer Science", "Biology"])])
        assert "**Fields of Study:** Computer Science, Biology" in prompt

    def test_includes_upvotes(self, make_paper):
        prompt = build_user_prompt([make_paper(upvotes=42)])
        assert "**Community Upvotes:** 42" in prompt

    def test_omits_empty_optional_fields(self, make_paper):
        prompt = build_user_prompt([make_paper()])
        assert "Categories:" not in prompt
        assert "Fields of Study:" not in prompt
        assert "Community Upvotes:" not in prompt

    def test_github_section(self, make_paper):
        prompt = build_user_prompt([make_paper()], github_section="Repo1\nRepo2")
        assert "Trending GitHub Repositories" in prompt
        assert "Repo1" in prompt


class TestLlmCall:
    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_success(self, mock_completion, make_paper, make_settings):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({"paper_1": "Result."})
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        result = llm_call([make_paper()], "system prompt", make_settings(), "test")
        assert result == {"paper_1": "Result."}

    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_empty_content_returns_empty(self, mock_completion, make_paper, make_settings):
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        result = llm_call([make_paper()], "prompt", make_settings(), "test")
        assert result == {}

    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_malformed_json_returns_empty(self, mock_completion, make_paper, make_settings):
        mock_choice = MagicMock()
        mock_choice.message.content = "not json"
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        result = llm_call([make_paper()], "prompt", make_settings(), "test")
        assert result == {}

    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_non_object_json_returns_empty(self, mock_completion, make_paper, make_settings):
        mock_choice = MagicMock()
        mock_choice.message.content = '["array"]'
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        result = llm_call([make_paper()], "prompt", make_settings(), "test")
        assert result == {}

    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_values_coerced_to_str(self, mock_completion, make_paper, make_settings):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({"paper_1": 42})
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        result = llm_call([make_paper()], "prompt", make_settings(), "test")
        assert result == {"paper_1": "42"}
