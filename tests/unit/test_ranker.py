"""Tests for the paper ranker module."""

from digest_pipeline.ranker import compute_keyword_scores


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
