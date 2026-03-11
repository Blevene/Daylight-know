"""Tests for ranking-related config fields."""

from digest_pipeline.config import Settings


def test_ranking_config_defaults():
    s = Settings(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b",
        email_to="c@d",
    )
    assert s.openalex_interest_profile == ""
    assert s.openalex_interest_keywords == []
    assert s.openalex_fetch_pool == 100


def test_ranking_config_from_values():
    s = Settings(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b",
        email_to="c@d",
        openalex_interest_profile="I study LLMs for drug discovery",
        openalex_interest_keywords=["LLM", "drug discovery"],
        openalex_fetch_pool=50,
    )
    assert s.openalex_interest_profile == "I study LLMs for drug discovery"
    assert s.openalex_interest_keywords == ["LLM", "drug discovery"]
    assert s.openalex_fetch_pool == 50


def test_pipeline_wide_interest_config_defaults():
    s = Settings(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b",
        email_to="c@d",
    )
    assert s.interest_profile == ""
    assert s.interest_keywords == []
    assert s.arxiv_fetch_pool == 200


def test_pipeline_wide_interest_config_from_values():
    s = Settings(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b",
        email_to="c@d",
        interest_profile="AI safety and alignment research",
        interest_keywords=["alignment", "RLHF", "interpretability"],
        arxiv_fetch_pool=150,
    )
    assert s.interest_profile == "AI safety and alignment research"
    assert s.interest_keywords == ["alignment", "RLHF", "interpretability"]
    assert s.arxiv_fetch_pool == 150


def test_openalex_interest_fields_still_exist():
    """Backward compat: openalex-prefixed fields still work."""
    s = Settings(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b",
        email_to="c@d",
        openalex_interest_profile="OpenAlex-specific profile",
        openalex_interest_keywords=["openalex-kw"],
    )
    assert s.openalex_interest_profile == "OpenAlex-specific profile"
    assert s.openalex_interest_keywords == ["openalex-kw"]


def test_llm_max_tokens_default_increased():
    s = Settings(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b",
        email_to="c@d",
    )
    assert s.llm_max_tokens == 32768
