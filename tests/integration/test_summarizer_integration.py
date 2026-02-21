"""Integration tests for summarizer.py -- litellm against stub LLM server.

Test IDs: S-1, S-2, S-3, S-5, S-6
"""

import litellm
import pytest

from digest_pipeline.config import Settings
from digest_pipeline.summarizer import summarize
from tests.stub_llm_server import StubConfig, run_stub_server


def _make_settings(port: int, **overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        llm_api_key="test-key",
        llm_model="openai/test-model",
        llm_api_base=f"http://127.0.0.1:{port}/v1",
        llm_max_tokens=4096,
        smtp_user="u",
        smtp_password="p",
        email_from="a@b.com",
        email_to="c@d.com",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestSummarizerIntegration:
    """Tests that exercise real litellm.completion() against a stub server."""

    def test_successful_completion(self, make_paper):
        """S-1: litellm.completion() returns a non-empty summary from stub."""
        with run_stub_server(StubConfig(response_content="Generated summary of papers.")) as srv:
            settings = _make_settings(srv.port)
            result = summarize([make_paper()], settings)

            assert result == "Generated summary of papers."
            assert len(srv.requests) == 1
            # Verify system prompt was sent
            messages = srv.requests[0]["messages"]
            assert messages[0]["role"] == "system"
            assert "expert research assistant" in messages[0]["content"]

    def test_rate_limit_retry_then_success(self, make_paper):
        """S-2: Stub returns 429 twice then 200 -- litellm retries and succeeds."""
        config = StubConfig(rate_limit_count=2, response_content="Eventually succeeded.")
        with run_stub_server(config) as srv:
            settings = _make_settings(srv.port)
            # Disable litellm's internal retries so the summarizer's own
            # backoff loop handles the 429s.
            old_retries = getattr(litellm, "num_retries", None)
            litellm.num_retries = 0
            try:
                result = summarize([make_paper()], settings)
            finally:
                litellm.num_retries = old_retries

            assert result == "Eventually succeeded."

    def test_rate_limit_exhaustion(self, make_paper):
        """S-3: Stub returns 429 for all attempts -- error is raised."""
        config = StubConfig(rate_limit_count=100)  # Always 429
        with run_stub_server(config) as srv:
            settings = _make_settings(srv.port)
            old_retries = getattr(litellm, "num_retries", None)
            litellm.num_retries = 0
            try:
                with pytest.raises(Exception):
                    summarize([make_paper()], settings)
            finally:
                litellm.num_retries = old_retries

    def test_null_content_returns_empty_string(self, make_paper):
        """S-5: Stub returns null content -> summarize returns empty string."""
        config = StubConfig(null_content=True)
        with run_stub_server(config) as srv:
            settings = _make_settings(srv.port)
            result = summarize([make_paper()], settings)

            assert result == ""

    def test_model_string_passed_to_litellm(self, make_paper):
        """S-6: Verify the full provider/model string reaches the stub server."""
        with run_stub_server() as srv:
            settings = _make_settings(srv.port, llm_model="openai/gpt-4o-mini")
            summarize([make_paper()], settings)

            assert len(srv.requests) == 1
            # litellm strips the provider prefix before sending to the API
            assert srv.requests[0]["model"] in ("gpt-4o-mini", "openai/gpt-4o-mini")
