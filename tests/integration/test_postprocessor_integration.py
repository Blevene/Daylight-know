"""Integration tests for postprocessor.py -- litellm against stub LLM server.

Test IDs: P-1, P-2
"""

import litellm
import pytest

from digest_pipeline.config import Settings
from digest_pipeline.postprocessor import (
    CRITIQUES_SYSTEM_PROMPT,
    IMPLICATIONS_SYSTEM_PROMPT,
    extract_implications,
    generate_critiques,
)
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
class TestPostprocessorIntegration:
    """Tests that exercise real litellm calls for implications and critiques."""

    def test_both_prompts_sent_correctly(self, make_paper):
        """P-1: Both IMPLICATIONS and CRITIQUES system prompts are sent via litellm."""
        with run_stub_server(StubConfig(response_content="Implications result.")) as srv:
            settings = _make_settings(srv.port)
            impl_result = extract_implications([make_paper()], settings)

            assert impl_result == "Implications result."
            assert len(srv.requests) == 1
            messages = srv.requests[0]["messages"]
            assert messages[0]["role"] == "system"
            assert "actionable insights" in messages[0]["content"]

        with run_stub_server(StubConfig(response_content="Critiques result.")) as srv:
            settings = _make_settings(srv.port)
            crit_result = generate_critiques([make_paper()], settings)

            assert crit_result == "Critiques result."
            assert len(srv.requests) == 1
            messages = srv.requests[0]["messages"]
            assert messages[0]["role"] == "system"
            assert "peer reviewer" in messages[0]["content"]

    def test_rate_limit_retry(self, make_paper):
        """P-2: Rate-limit backoff works for _llm_call (shared by both functions)."""
        config = StubConfig(rate_limit_count=1, response_content="After retry.")
        with run_stub_server(config) as srv:
            settings = _make_settings(srv.port)
            # Disable litellm's internal retries so the postprocessor's own
            # backoff loop handles the 429.
            old_retries = getattr(litellm, "num_retries", None)
            litellm.num_retries = 0
            try:
                result = extract_implications([make_paper()], settings)
            finally:
                litellm.num_retries = old_retries

            assert result == "After retry."
