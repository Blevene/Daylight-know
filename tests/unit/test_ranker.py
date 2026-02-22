"""Tests for the paper ranker module."""

from unittest.mock import patch, MagicMock

from digest_pipeline.ranker import compute_keyword_scores, score_batch_with_llm


def test_keyword_scoring_basic(make_paper):
    papers = [
        make_paper(title="LLM for Drug Discovery", abstract="We apply large language models to find drugs."),
        make_paper(title="Image Segmentation", abstract="A new method for segmenting images."),
    ]
    scores = compute_keyword_scores(papers, ["LLM", "drug discovery"])
    assert scores[0] > scores[1]


def test_keyword_scoring_case_insensitive(make_paper):
    papers = [make_paper(title="llm research", abstract="about llm")]
    scores = compute_keyword_scores(papers, ["LLM"])
    assert scores[0] == 4  # +2 for title match, +2 for abstract match


def test_keyword_scoring_empty_keywords(make_paper):
    papers = [make_paper()]
    scores = compute_keyword_scores(papers, [])
    assert scores == [0]


def test_keyword_scoring_no_match(make_paper):
    papers = [make_paper(title="Unrelated", abstract="Nothing here")]
    scores = compute_keyword_scores(papers, ["quantum"])
    assert scores == [0]


@patch("digest_pipeline.ranker.litellm.completion")
def test_llm_batch_scoring(mock_completion, make_paper, make_settings):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = '{"paper_1": 8, "paper_2": 3}'
    mock_completion.return_value = mock_resp

    papers = [
        make_paper(title="LLM Drug Discovery"),
        make_paper(title="Image Segmentation"),
    ]
    settings = make_settings(
        openalex_interest_profile="I study LLMs for drug discovery",
    )
    scores = score_batch_with_llm(papers, settings)
    assert scores == [8, 3]


@patch("digest_pipeline.ranker.litellm.completion")
def test_llm_batch_scoring_handles_failure(mock_completion, make_paper, make_settings):
    mock_completion.side_effect = Exception("API error")

    papers = [make_paper(), make_paper()]
    settings = make_settings(openalex_interest_profile="anything")
    scores = score_batch_with_llm(papers, settings)
    assert scores == [0, 0]  # graceful degradation


@patch("digest_pipeline.ranker.litellm.completion")
def test_llm_batch_scoring_handles_invalid_json(mock_completion, make_paper, make_settings):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "not json"
    mock_completion.return_value = mock_resp

    papers = [make_paper()]
    settings = make_settings(openalex_interest_profile="anything")
    scores = score_batch_with_llm(papers, settings)
    assert scores == [0]
