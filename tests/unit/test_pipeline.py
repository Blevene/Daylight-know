"""Tests for the pipeline orchestrator."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from digest_pipeline.extractor import ExtractionResult
from digest_pipeline.fetcher import Paper
from digest_pipeline.pipeline import PaperAnalysis, _build_analyses, run


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        paper_id="2401.00001",
        title="Test Paper",
        authors=["Alice"],
        abstract="Abstract text.",
        url="https://arxiv.org/abs/2401.00001",
        published=datetime(2025, 1, 15, tzinfo=timezone.utc),
        source="arxiv",
        pdf_path=Path("/tmp/fake.pdf"),
    )
    defaults.update(overrides)
    return Paper(**defaults)


def test_build_analyses():
    papers = [
        _make_paper(title="Paper 1", url="http://1", authors=["A"]),
        _make_paper(title="Paper 2", url="http://2", authors=["B"]),
    ]
    summaries = {"paper_1": "Sum 1", "paper_2": "Sum 2"}
    implications = {"paper_1": "Impl 1"}
    critiques = {"paper_2": "Crit 2"}

    analyses = _build_analyses(papers, summaries, implications, critiques)

    assert len(analyses) == 2
    assert analyses[0].title == "Paper 1"
    assert analyses[0].summary == "Sum 1"
    assert analyses[0].implications == "Impl 1"
    assert analyses[0].critique == ""
    assert analyses[1].title == "Paper 2"
    assert analyses[1].summary == "Sum 2"
    assert analyses[1].implications == ""
    assert analyses[1].critique == "Crit 2"


def test_build_analyses_passes_categories():
    papers = [
        _make_paper(title="Paper 1", url="http://1", authors=["A"], categories=["cs.AI", "cs.LG"]),
    ]
    summaries = {"paper_1": "Sum 1"}

    analyses = _build_analyses(papers, summaries, {}, {})

    assert analyses[0].categories == ["cs.AI", "cs.LG"]


def test_build_analyses_passes_upvotes_and_fields_of_study():
    papers = [
        _make_paper(
            title="HF Paper",
            url="http://hf",
            authors=["A"],
            source="huggingface",
            upvotes=42,
            fields_of_study=["Computer Science", "Mathematics"],
        ),
    ]
    summaries = {"paper_1": "Sum 1"}

    analyses = _build_analyses(papers, summaries, {}, {})

    assert analyses[0].upvotes == 42
    assert analyses[0].fields_of_study == ["Computer Science", "Mathematics"]


@patch("digest_pipeline.pipeline.fetch_papers", return_value=[])
def test_run_no_papers(mock_fetch, make_settings):
    """Pipeline exits gracefully when no papers are found."""
    run(make_settings())
    mock_fetch.assert_called_once()


@patch("digest_pipeline.pipeline.save_seen")
@patch("digest_pipeline.pipeline.load_seen", return_value={})
@patch("digest_pipeline.pipeline.send_digest")
@patch("digest_pipeline.pipeline.generate_critiques", return_value={"paper_1": "Critiques text"})
@patch(
    "digest_pipeline.pipeline.extract_implications", return_value={"paper_1": "Implications text"}
)
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "Summary"})
@patch("digest_pipeline.pipeline.store_chunks", return_value=[])
@patch("digest_pipeline.pipeline.chunk_text", return_value=[])
@patch(
    "digest_pipeline.pipeline.extract_text",
    return_value=ExtractionResult(paper_id="2401.00001", text="content", parseable=True),
)
@patch("digest_pipeline.pipeline.fetch_papers")
def test_run_full_pipeline(
    mock_fetch,
    mock_extract,
    mock_chunk,
    mock_store,
    mock_summarize,
    mock_implications,
    mock_critiques,
    mock_email,
    mock_load_seen,
    mock_save_seen,
    make_settings,
):
    paper = _make_paper()
    mock_fetch.return_value = [paper]
    settings = make_settings()

    run(settings)

    mock_fetch.assert_called_once_with(settings, max_results=settings.arxiv_max_results)
    mock_extract.assert_called_once()
    mock_chunk.assert_called_once_with("content")
    mock_store.assert_called_once()
    mock_summarize.assert_called_once()
    mock_implications.assert_called_once()
    mock_critiques.assert_called_once()
    mock_email.assert_called_once()
    # Verify PaperAnalysis objects are passed to send_digest
    call_args = mock_email.call_args
    analyses = call_args.args[0]
    assert len(analyses) == 1
    assert isinstance(analyses[0], PaperAnalysis)
    assert analyses[0].summary == "Summary"
    assert analyses[0].implications == "Implications text"
    assert analyses[0].critique == "Critiques text"
    assert analyses[0].title == "Test Paper"
    assert analyses[0].url == "https://arxiv.org/abs/2401.00001"


@patch("digest_pipeline.pipeline.save_seen")
@patch("digest_pipeline.pipeline.load_seen", return_value={})
@patch("digest_pipeline.pipeline.send_digest")
@patch("digest_pipeline.pipeline.generate_critiques")
@patch("digest_pipeline.pipeline.extract_implications")
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "Summary"})
@patch("digest_pipeline.pipeline.store_chunks", return_value=[])
@patch("digest_pipeline.pipeline.chunk_text", return_value=[])
@patch(
    "digest_pipeline.pipeline.extract_text",
    return_value=ExtractionResult(paper_id="2401.00001", text="content", parseable=True),
)
@patch("digest_pipeline.pipeline.fetch_papers")
def test_run_postprocessing_disabled(
    mock_fetch,
    mock_extract,
    mock_chunk,
    mock_store,
    mock_summarize,
    mock_implications,
    mock_critiques,
    mock_email,
    mock_load_seen,
    mock_save_seen,
    make_settings,
):
    paper = _make_paper()
    mock_fetch.return_value = [paper]
    settings = make_settings(
        postprocessing_implications=False,
        postprocessing_critiques=False,
    )

    run(settings)

    mock_implications.assert_not_called()
    mock_critiques.assert_not_called()
    call_args = mock_email.call_args
    analyses = call_args.args[0]
    assert analyses[0].implications == ""
    assert analyses[0].critique == ""


def test_build_analyses_warns_on_missing_keys(caplog):
    """Warnings are logged when LLM result dicts are missing paper keys."""
    papers = [
        _make_paper(title="Paper 1", url="http://1", authors=["A"]),
        _make_paper(title="Paper 2", url="http://2", authors=["B"]),
    ]
    summaries = {"paper_1": "Sum 1"}  # missing paper_2
    implications = {"paper_1": "Impl 1", "paper_2": "Impl 2"}
    critiques = {"paper_1": "Crit 1"}  # missing paper_2

    with caplog.at_level(logging.WARNING, logger="digest_pipeline.pipeline"):
        analyses = _build_analyses(papers, summaries, implications, critiques)

    assert len(analyses) == 2
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Missing summary" in w and "paper_2" in w for w in warnings)
    assert any("Missing critique" in w and "paper_2" in w for w in warnings)
    # paper_2 implications is present, so no warning for it
    assert not any("Missing implications" in w and "paper_2" in w for w in warnings)


@patch("digest_pipeline.pipeline.save_seen")
@patch("digest_pipeline.pipeline.load_seen", return_value={})
@patch("digest_pipeline.pipeline.store_unparseable")
@patch(
    "digest_pipeline.pipeline.extract_text",
    return_value=ExtractionResult(paper_id="2401.00001", text="", parseable=False),
)
@patch("digest_pipeline.pipeline.fetch_papers")
def test_run_unparseable_paper(
    mock_fetch, mock_extract, mock_store_unparse, mock_load_seen, mock_save_seen, make_settings
):
    paper = _make_paper()
    mock_fetch.return_value = [paper]

    run(make_settings())

    mock_store_unparse.assert_called_once()


@patch("digest_pipeline.pipeline.save_seen")
@patch("digest_pipeline.pipeline.load_seen", return_value={})
@patch("digest_pipeline.pipeline.send_digest")
@patch("digest_pipeline.pipeline.generate_critiques", return_value={})
@patch(
    "digest_pipeline.pipeline.extract_implications", return_value={"paper_1": "Implications text"}
)
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "Summary"})
@patch("digest_pipeline.pipeline.store_chunks", return_value=[])
@patch("digest_pipeline.pipeline.chunk_text", return_value=[])
@patch(
    "digest_pipeline.pipeline.extract_text",
    return_value=ExtractionResult(paper_id="2401.00001", text="content", parseable=True),
)
@patch("digest_pipeline.pipeline.fetch_papers")
def test_run_logs_error_when_critiques_empty(
    mock_fetch,
    mock_extract,
    mock_chunk,
    mock_store,
    mock_summarize,
    mock_implications,
    mock_critiques,
    mock_email,
    mock_load_seen,
    mock_save_seen,
    caplog,
    make_settings,
):
    """Pipeline logs error when critique generation returns empty and still sends digest."""
    paper = _make_paper()
    mock_fetch.return_value = [paper]
    settings = make_settings()

    with caplog.at_level(logging.ERROR, logger="digest_pipeline.pipeline"):
        run(settings)

    assert any(
        "Critique generation returned no results" in r.message and r.levelno == logging.ERROR
        for r in caplog.records
    )
    mock_email.assert_called_once()


@patch("digest_pipeline.pipeline.save_seen")
@patch("digest_pipeline.pipeline.load_seen", return_value={})
@patch("digest_pipeline.pipeline.send_digest")
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "Summary"})
@patch("digest_pipeline.pipeline.store_chunks")
@patch("digest_pipeline.pipeline.chunk_text", return_value=[])
@patch("digest_pipeline.pipeline.rank_papers")
@patch("digest_pipeline.pipeline.fetch_openalex_papers")
@patch("digest_pipeline.pipeline.fetch_papers", return_value=[])
def test_pipeline_calls_ranker_for_openalex(
    mock_fetch,
    mock_oa_fetch,
    mock_rank,
    mock_chunk,
    mock_store,
    mock_summarize,
    mock_send,
    mock_load_seen,
    mock_save_seen,
    make_settings,
):
    """rank_papers is called on OpenAlex results before adding to pipeline."""
    oa_paper = _make_paper(
        paper_id="oa_W1", source="openalex", pdf_path=None, abstract="Test abstract"
    )
    mock_oa_fetch.return_value = [oa_paper]
    mock_rank.return_value = [oa_paper]

    settings = make_settings(
        openalex_enabled=True,
        openalex_interest_profile="test",
        postprocessing_implications=False,
        postprocessing_critiques=False,
    )
    run(settings)

    mock_rank.assert_called_once()
    assert mock_rank.call_args[0][0] == [oa_paper]


@patch("digest_pipeline.pipeline.save_seen")
@patch("digest_pipeline.pipeline.record_papers")
@patch("digest_pipeline.pipeline.filter_unseen", side_effect=lambda papers, seen: papers)
@patch("digest_pipeline.pipeline.load_seen", return_value={})
@patch("digest_pipeline.pipeline.send_digest")
@patch("digest_pipeline.pipeline.store_chunks")
@patch("digest_pipeline.pipeline.chunk_text", return_value=["chunk1"])
@patch("digest_pipeline.pipeline.generate_critiques", return_value={})
@patch("digest_pipeline.pipeline.extract_implications", return_value={})
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "summary", "paper_2": "summary"})
@patch("digest_pipeline.pipeline.fetch_papers")
@patch("digest_pipeline.pipeline.rank_papers")
def test_arxiv_papers_ranked_when_interest_configured(
    mock_rank,
    mock_fetch,
    mock_summarize,
    mock_impl,
    mock_critiques,
    mock_chunk,
    mock_store,
    mock_send,
    mock_load_seen,
    mock_filter,
    mock_record,
    mock_save,
    make_paper,
    make_settings,
):
    """arXiv papers should be ranked when interest profile is set."""
    arxiv_papers = [make_paper(paper_id=f"p{i}", abstract=f"abstract {i}") for i in range(5)]
    mock_fetch.return_value = arxiv_papers
    mock_rank.return_value = arxiv_papers[:2]  # ranked down to 2

    settings = make_settings(
        interest_profile="AI safety research",
        interest_keywords=["alignment"],
        arxiv_max_results=2,
        arxiv_fetch_pool=50,
        postprocessing_implications=False,
        postprocessing_critiques=False,
    )
    run(settings)

    mock_rank.assert_called_once()
    call_args = mock_rank.call_args
    assert len(call_args[0][0]) == 5  # all papers passed to ranker
    assert call_args[1]["max_results"] == 2
