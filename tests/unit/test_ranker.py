"""Tests for the paper ranker module."""

from unittest.mock import patch, MagicMock

from digest_pipeline.ranker import compute_keyword_scores, score_batch_with_llm, rank_papers


def test_keyword_scoring_basic(make_paper):
    papers = [
        make_paper(
            title="LLM for Drug Discovery", abstract="We apply large language models to find drugs."
        ),
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
    settings = make_settings()
    scores = score_batch_with_llm(
        papers, settings, interest_profile="I study LLMs for drug discovery"
    )
    assert scores == [8, 3]


@patch("digest_pipeline.ranker.litellm.completion")
def test_llm_batch_scoring_handles_failure(mock_completion, make_paper, make_settings):
    mock_completion.side_effect = Exception("API error")

    papers = [make_paper(), make_paper()]
    settings = make_settings()
    scores = score_batch_with_llm(papers, settings, interest_profile="anything")
    assert scores == [0, 0]  # graceful degradation


@patch("digest_pipeline.ranker.litellm.completion")
def test_llm_batch_scoring_handles_invalid_json(mock_completion, make_paper, make_settings):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "not json"
    mock_completion.return_value = mock_resp

    papers = [make_paper()]
    settings = make_settings()
    scores = score_batch_with_llm(papers, settings, interest_profile="anything")
    assert scores == [0]


@patch("digest_pipeline.ranker.score_batch_with_llm")
def test_rank_papers_combines_scores(mock_llm_score, make_paper, make_settings):
    """Papers are ranked by combined keyword + LLM score."""
    papers = [
        make_paper(paper_id="low", title="Unrelated Topic", abstract="Nothing relevant here"),
        make_paper(
            paper_id="high", title="LLM for Drug Discovery", abstract="We use LLM to find drugs"
        ),
    ]
    # LLM gives both a 5, but keywords will differentiate
    mock_llm_score.return_value = [5, 5]

    settings = make_settings(
        interest_profile="LLMs for drug discovery",
        interest_keywords=["LLM", "drug"],
    )
    ranked = rank_papers(papers, settings, max_results=1)
    assert len(ranked) == 1
    assert ranked[0].paper_id == "high"


def test_rank_papers_no_profile_returns_unchanged(make_paper, make_settings):
    """When no interest profile or keywords, return papers unchanged."""
    papers = [make_paper(paper_id="a"), make_paper(paper_id="b")]
    settings = make_settings()
    ranked = rank_papers(papers, settings, max_results=20)
    assert len(ranked) == 2
    assert ranked[0].paper_id == "a"


@patch("digest_pipeline.ranker.score_batch_with_llm")
def test_rank_papers_batches_correctly(mock_llm_score, make_paper, make_settings):
    """Papers are scored in batches of BATCH_SIZE."""
    papers = [make_paper(paper_id=f"p{i}") for i in range(25)]
    mock_llm_score.side_effect = [
        [5] * 20,  # first batch
        [5] * 5,  # second batch
    ]
    settings = make_settings(
        interest_profile="test",
    )
    ranked = rank_papers(papers, settings, max_results=10)
    assert len(ranked) == 10
    assert mock_llm_score.call_count == 2
