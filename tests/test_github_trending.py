"""Tests for the GitHub trending module."""

from unittest.mock import MagicMock, patch

from digest_pipeline.config import Settings
from digest_pipeline.github_trending import TrendingRepo, fetch_trending, format_for_prompt


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b.com",
        email_to="c@d.com",
        github_enabled=True,
        github_languages=["python"],
        github_top_n=3,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_fetch_trending_disabled():
    settings = _make_settings(github_enabled=False)
    assert fetch_trending(settings) == []


@patch("digest_pipeline.github_trending.requests.get")
def test_fetch_trending_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "items": [
            {
                "full_name": "user/repo1",
                "description": "A cool repo",
                "html_url": "https://github.com/user/repo1",
                "stargazers_count": 100,
                "language": "Python",
            }
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    settings = _make_settings()
    repos = fetch_trending(settings)
    assert len(repos) == 1
    assert repos[0].name == "user/repo1"


def test_format_for_prompt_empty():
    assert format_for_prompt([]) == ""


def test_format_for_prompt():
    repos = [
        TrendingRepo(name="user/repo", description="Desc", url="https://github.com/user/repo", stars=42, language="Python")
    ]
    result = format_for_prompt(repos)
    assert "user/repo" in result
    assert "42 stars" in result
