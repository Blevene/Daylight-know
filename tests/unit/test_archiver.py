"""Tests for the PDF archiver module."""

from datetime import datetime, timezone

from digest_pipeline.archiver import _init_db, _sanitize_filename, _upsert_papers, _generate_markdown_index
from digest_pipeline.fetcher import Paper


def test_sanitize_basic_title():
    result = _sanitize_filename("Attention Is All You Need", "2603.20213")
    assert result == "Attention-Is-All-You-Need_2603.20213.pdf"


def test_sanitize_special_characters():
    result = _sanitize_filename("A (New) Approach: To LLMs!", "2603.00001")
    assert result == "A-New-Approach-To-LLMs_2603.00001.pdf"


def test_sanitize_truncates_long_title():
    long_title = "A" * 120
    result = _sanitize_filename(long_title, "2603.00001")
    title_part = result.split("_")[0]
    assert len(title_part) <= 80


def test_sanitize_old_arxiv_id_with_slash():
    result = _sanitize_filename("Some Paper", "hep-ph/0301001")
    assert result == "Some-Paper_hep-ph-0301001.pdf"


def test_sanitize_hf_prefix_preserved():
    result = _sanitize_filename("HF Paper", "hf_2603.00001")
    assert result == "HF-Paper_hf_2603.00001.pdf"


def test_sanitize_empty_title():
    result = _sanitize_filename("", "2603.00001")
    assert result == "_2603.00001.pdf"


def test_sanitize_whitespace_collapsed():
    result = _sanitize_filename("  Multiple   Spaces  ", "2603.00001")
    assert result == "Multiple-Spaces_2603.00001.pdf"



def _make_paper(**kwargs):
    defaults = dict(
        paper_id="2603.00001",
        title="Test Paper",
        authors=["Alice Smith", "Bob Jones"],
        abstract="An abstract.",
        url="https://arxiv.org/abs/2603.00001",
        published=datetime(2026, 3, 24, tzinfo=timezone.utc),
        source="arxiv",
        categories=["cs.AI"],
    )
    defaults.update(kwargs)
    return Paper(**defaults)


def test_markdown_index_basic():
    papers = [_make_paper()]
    filenames = {"2603.00001": "Test-Paper_2603.00001.pdf"}
    md = _generate_markdown_index(papers, "2026-03-24", filenames)
    assert "# Papers — 2026-03-24" in md
    assert "Test Paper" in md
    assert "[PDF](Test-Paper_2603.00001.pdf)" in md
    assert "1 papers archived." in md


def test_markdown_index_author_truncation():
    papers = [_make_paper(authors=["Alice", "Bob", "Charlie"])]
    filenames = {"2603.00001": "Test-Paper_2603.00001.pdf"}
    md = _generate_markdown_index(papers, "2026-03-24", filenames)
    assert "Alice et al." in md


def test_markdown_index_two_authors_not_truncated():
    papers = [_make_paper(authors=["Alice", "Bob"])]
    filenames = {"2603.00001": "Test-Paper_2603.00001.pdf"}
    md = _generate_markdown_index(papers, "2026-03-24", filenames)
    assert "Alice, Bob" in md


def test_markdown_index_no_pdf():
    papers = [_make_paper()]
    filenames = {}  # no PDF archived
    md = _generate_markdown_index(papers, "2026-03-24", filenames)
    assert "\u2014" in md


def test_markdown_index_includes_url():
    papers = [_make_paper(url="https://arxiv.org/abs/2603.00001")]
    filenames = {"2603.00001": "Test-Paper_2603.00001.pdf"}
    md = _generate_markdown_index(papers, "2026-03-24", filenames)
    assert "[arXiv](https://arxiv.org/abs/2603.00001)" in md


def test_init_db_creates_table(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = _init_db(db_path)
    tables = con.execute("SHOW TABLES").fetchall()
    assert any("papers" in str(t) for t in tables)
    con.close()


def test_upsert_inserts_new_paper(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = _init_db(db_path)
    papers = [_make_paper()]
    filenames = {"2603.00001": "2026-03-24/Test-Paper_2603.00001.pdf"}
    _upsert_papers(con, papers, "2026-03-24", filenames)

    rows = con.execute("SELECT * FROM papers").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "2603.00001"  # paper_id
    assert rows[0][1] == "Test Paper"  # title
    con.close()


def test_upsert_preserves_pdf_path_on_rerun(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = _init_db(db_path)
    papers = [_make_paper()]

    # First run: paper has a PDF
    _upsert_papers(con, papers, "2026-03-24", {"2603.00001": "2026-03-24/Test.pdf"})
    # Second run: paper has no PDF (re-run, temp file gone)
    _upsert_papers(con, papers, "2026-03-24", {})

    rows = con.execute("SELECT pdf_path FROM papers WHERE paper_id = '2603.00001'").fetchall()
    assert rows[0][0] == "2026-03-24/Test.pdf"  # preserved, not overwritten with NULL
    con.close()


def test_upsert_updates_metadata_on_rerun(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = _init_db(db_path)

    paper_v1 = _make_paper(title="Original Title")
    _upsert_papers(con, [paper_v1], "2026-03-24", {"2603.00001": "2026-03-24/Test.pdf"})

    paper_v2 = _make_paper(title="Updated Title")
    _upsert_papers(con, [paper_v2], "2026-03-24", {"2603.00001": "2026-03-24/Test.pdf"})

    rows = con.execute("SELECT title FROM papers WHERE paper_id = '2603.00001'").fetchall()
    assert rows[0][0] == "Updated Title"
    con.close()
