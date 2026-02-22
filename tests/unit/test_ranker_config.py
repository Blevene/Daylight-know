"""Tests for ranking-related config fields."""

from digest_pipeline.config import Settings


def test_ranking_config_defaults():
    s = Settings(_env_file=None, llm_api_key="k", smtp_user="u",
                 smtp_password="p", email_from="a@b", email_to="c@d")
    assert s.openalex_interest_profile == ""
    assert s.openalex_interest_keywords == []
    assert s.openalex_fetch_pool == 100


def test_ranking_config_from_values():
    s = Settings(
        _env_file=None, llm_api_key="k", smtp_user="u",
        smtp_password="p", email_from="a@b", email_to="c@d",
        openalex_interest_profile="I study LLMs for drug discovery",
        openalex_interest_keywords=["LLM", "drug discovery"],
        openalex_fetch_pool=50,
    )
    assert s.openalex_interest_profile == "I study LLMs for drug discovery"
    assert s.openalex_interest_keywords == ["LLM", "drug discovery"]
    assert s.openalex_fetch_pool == 50
