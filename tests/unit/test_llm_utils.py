"""Tests for the shared LLM call utilities."""

import json
import logging
from unittest.mock import MagicMock, call, patch

import litellm
import pytest

from digest_pipeline.llm_utils import LLM_BATCH_SIZE, build_response_format, build_user_prompt, llm_call

_MAX_ATTEMPTS = 5  # must match llm_utils.max_backoff_attempts
_EXPECTED_SLEEPS = [call(2**i) for i in range(1, _MAX_ATTEMPTS)]  # [call(2), call(4), call(8), call(16)]


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

    @patch("digest_pipeline.llm_utils.time.sleep")
    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_empty_content_returns_empty(self, mock_completion, mock_sleep, make_paper, make_settings):
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        result = llm_call([make_paper()], "prompt", make_settings(), "test")
        assert result == {}

    @patch("digest_pipeline.llm_utils.time.sleep")
    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_malformed_json_returns_empty(self, mock_completion, mock_sleep, make_paper, make_settings):
        mock_choice = MagicMock()
        mock_choice.message.content = "not json"
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        result = llm_call([make_paper()], "prompt", make_settings(), "test")
        assert result == {}

    @patch("digest_pipeline.llm_utils.time.sleep")
    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_non_object_json_returns_empty(self, mock_completion, mock_sleep, make_paper, make_settings):
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

    # ── Retry behaviour tests ──────────────────────────────────────

    @pytest.mark.parametrize("bad_content", [
        None,          # empty content
        "not json",    # invalid JSON
        '["array"]',   # non-dict JSON
    ], ids=["empty_content", "invalid_json", "non_dict_json"])
    @patch("digest_pipeline.llm_utils.time.sleep")
    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_bad_content_retries_until_exhausted(self, mock_completion, mock_sleep, bad_content, make_paper, make_settings):
        mock_choice = MagicMock()
        mock_choice.message.content = bad_content
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        result = llm_call([make_paper()], "prompt", make_settings(), "test")
        assert result == {}
        assert mock_completion.call_count == _MAX_ATTEMPTS
        assert mock_sleep.call_count == _MAX_ATTEMPTS - 1
        assert mock_sleep.call_args_list == _EXPECTED_SLEEPS

    @patch("digest_pipeline.llm_utils.time.sleep")
    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_rate_limit_returns_empty_after_exhausting_retries(self, mock_completion, mock_sleep, make_paper, make_settings):
        mock_completion.side_effect = litellm.RateLimitError(
            message="rate limited", model="test", llm_provider="test"
        )

        result = llm_call([make_paper()], "prompt", make_settings(), "test")
        assert result == {}
        assert mock_completion.call_count == _MAX_ATTEMPTS
        assert mock_sleep.call_count == _MAX_ATTEMPTS - 1
        assert mock_sleep.call_args_list == _EXPECTED_SLEEPS

    @patch("digest_pipeline.llm_utils.time.sleep")
    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_generic_exception_returns_empty_after_exhausting_retries(self, mock_completion, mock_sleep, make_paper, make_settings):
        mock_completion.side_effect = RuntimeError("something broke")

        result = llm_call([make_paper()], "prompt", make_settings(), "test")
        assert result == {}
        assert mock_completion.call_count == _MAX_ATTEMPTS
        assert mock_sleep.call_count == _MAX_ATTEMPTS - 1
        assert mock_sleep.call_args_list == _EXPECTED_SLEEPS

    @patch("digest_pipeline.llm_utils.time.sleep")
    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_missing_keys_warns_and_returns_partial(self, mock_completion, mock_sleep, caplog, make_paper, make_settings):
        papers = [make_paper(title="Paper 1"), make_paper(title="Paper 2")]
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({"paper_1": "Result 1"})
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        with caplog.at_level(logging.WARNING, logger="digest_pipeline.llm_utils"):
            result = llm_call(papers, "prompt", make_settings(), "test")

        assert result == {"paper_1": "Result 1"}
        assert mock_completion.call_count == 1  # no retry for partial results
        assert any("paper_2" in r.message for r in caplog.records if r.levelno == logging.WARNING)

    @patch("digest_pipeline.llm_utils.time.sleep")
    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_recovers_on_retry_after_empty_content(self, mock_completion, mock_sleep, make_paper, make_settings):
        empty_choice = MagicMock()
        empty_choice.message.content = None
        valid_choice = MagicMock()
        valid_choice.message.content = json.dumps({"paper_1": "Got it."})
        mock_completion.side_effect = [
            MagicMock(choices=[empty_choice]),
            MagicMock(choices=[valid_choice]),
        ]

        result = llm_call([make_paper()], "prompt", make_settings(), "test")
        assert result == {"paper_1": "Got it."}
        assert mock_completion.call_count == 2


class TestLlmCallBatching:
    """Tests for automatic batching when paper count exceeds LLM_BATCH_SIZE."""

    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_small_batch_no_splitting(self, mock_completion, make_paper, make_settings):
        """Papers <= LLM_BATCH_SIZE should result in a single LLM call."""
        papers = [make_paper(title=f"Paper {i}") for i in range(1, LLM_BATCH_SIZE + 1)]
        response_data = {f"paper_{i}": f"Summary {i}" for i in range(1, len(papers) + 1)}
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(response_data)
        mock_completion.return_value = MagicMock(choices=[mock_choice])

        result = llm_call(papers, "system prompt", make_settings(), "test")
        assert mock_completion.call_count == 1
        assert result == response_data

    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_large_batch_splits_and_merges(self, mock_completion, make_paper, make_settings):
        """Papers > LLM_BATCH_SIZE should be split into multiple calls and merged."""
        num_papers = LLM_BATCH_SIZE + 3
        papers = [make_paper(title=f"Paper {i}") for i in range(1, num_papers + 1)]

        def fake_completion(**kwargs):
            """Return correctly-keyed JSON for whatever batch was sent."""
            user_msg = kwargs["messages"][1]["content"]
            # Count papers in the user prompt
            count = user_msg.count("### Paper ")
            resp = {f"paper_{i}": f"Summary {i}" for i in range(1, count + 1)}
            mock_choice = MagicMock()
            mock_choice.message.content = json.dumps(resp)
            return MagicMock(choices=[mock_choice])

        mock_completion.side_effect = fake_completion

        result = llm_call(papers, "system prompt", make_settings(), "test")
        # Should have called LLM twice (one full batch + one remainder)
        assert mock_completion.call_count == 2
        # Final result should have all papers with globally re-keyed indices
        assert len(result) == num_papers
        for i in range(1, num_papers + 1):
            assert f"paper_{i}" in result

    @patch("digest_pipeline.llm_utils.time.sleep")
    @patch("digest_pipeline.llm_utils.litellm.completion")
    def test_one_batch_fails_others_succeed(self, mock_completion, mock_sleep, make_paper, make_settings):
        """If one batch fails all retries, other batches' results still appear."""
        num_papers = LLM_BATCH_SIZE + 3
        # Use unique titles so we can tell batches apart
        papers = [make_paper(title=f"ALPHA_{i}") for i in range(1, LLM_BATCH_SIZE + 1)]
        papers += [make_paper(title=f"BETA_{i}") for i in range(1, 4)]

        def fake_completion(**kwargs):
            user_msg = kwargs["messages"][1]["content"]
            paper_count = user_msg.count("### Paper ")
            # First batch (contains ALPHA titles) always fails
            if "ALPHA_" in user_msg:
                mock_choice = MagicMock()
                mock_choice.message.content = "not json"
                return MagicMock(choices=[mock_choice])
            # Second batch (contains BETA titles) succeeds
            resp = {f"paper_{i}": f"Summary {i}" for i in range(1, paper_count + 1)}
            mock_choice = MagicMock()
            mock_choice.message.content = json.dumps(resp)
            return MagicMock(choices=[mock_choice])

        mock_completion.side_effect = fake_completion

        result = llm_call(papers, "system prompt", make_settings(), "test")
        # Should have results from the second batch, re-keyed to global indices
        for i in range(LLM_BATCH_SIZE + 1, num_papers + 1):
            assert f"paper_{i}" in result
        # First batch papers should NOT be present (failed)
        for i in range(1, LLM_BATCH_SIZE + 1):
            assert f"paper_{i}" not in result
