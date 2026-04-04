"""Integration tests for archiver.py — real file copies and DuckDB."""

import duckdb
import pytest
from datetime import datetime, timezone

from digest_pipeline.archiver import archive_papers
from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper


def _make_paper(paper_id, title, tmp_dir):
    """Create a Paper with a real temp PDF file."""
    pdf_path = tmp_dir / f"{paper_id}.pdf"
    pdf_path.write_bytes(b"%PDF-fake-content")
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=["Alice Smith", "Bob Jones", "Charlie Brown"],
        abstract="Test abstract.",
        url=f"https://arxiv.org/abs/{paper_id}",
        published=datetime(2026, 3, 24, tzinfo=timezone.utc),
        source="arxiv",
        categories=["cs.AI", "cs.LG"],
        pdf_path=pdf_path,
    )


@pytest.mark.integration
@pytest.mark.timeout(15)
class TestArchiverIntegration:

    def test_full_archive_flow(self, tmp_path):
        """Archive papers, verify directory structure, DuckDB, and markdown."""
        archive_dir = tmp_path / "archive"
        pdf_tmp = tmp_path / "pdfs"
        pdf_tmp.mkdir()

        papers = [
            _make_paper("2603.00001", "First Paper", pdf_tmp),
            _make_paper("2603.00002", "Second Paper", pdf_tmp),
        ]

        settings = Settings(
            _env_file=None,
            pdf_archive_dir=str(archive_dir),
            llm_api_key="k",
            smtp_user="u",
            email_from="a@b.com",
            email_to="c@d.com",
        )

        archive_papers(papers, "2026-03-24", settings)

        # Verify directory structure
        day_dir = archive_dir / "2026-03-24"
        assert day_dir.is_dir()

        # Verify PDFs were copied
        pdf_files = list(day_dir.glob("*.pdf"))
        assert len(pdf_files) == 2

        # Verify markdown index
        index_md = day_dir / "index.md"
        assert index_md.exists()
        content = index_md.read_text()
        assert "First Paper" in content
        assert "Second Paper" in content
        assert "Alice Smith et al." in content  # 3 authors -> truncated

        # Verify DuckDB
        db_path = archive_dir / "archive.duckdb"
        assert db_path.exists()
        con = duckdb.connect(str(db_path))
        rows = con.execute("SELECT paper_id, title FROM papers ORDER BY paper_id").fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "2603.00001"
        assert rows[1][1] == "Second Paper"
        con.close()

        # Verify original pdf_path is NOT mutated
        assert papers[0].pdf_path == pdf_tmp / "2603.00001.pdf"

    def test_archive_noop_when_disabled(self, tmp_path):
        """archive_papers does nothing when pdf_archive_dir is empty."""
        settings = Settings(
            _env_file=None,
            pdf_archive_dir="",
            llm_api_key="k",
            smtp_user="u",
            email_from="a@b.com",
            email_to="c@d.com",
        )
        archive_papers([], "2026-03-24", settings)
        # No directories created
        assert not (tmp_path / "archive").exists()

    def test_archive_skips_missing_pdf(self, tmp_path):
        """Papers whose pdf_path file doesn't exist are skipped gracefully."""
        archive_dir = tmp_path / "archive"
        paper = Paper(
            paper_id="2603.99999",
            title="Ghost Paper",
            authors=["Nobody"],
            abstract="Gone.",
            url="https://arxiv.org/abs/2603.99999",
            published=datetime(2026, 3, 24, tzinfo=timezone.utc),
            source="arxiv",
            categories=["cs.AI"],
            pdf_path=tmp_path / "nonexistent.pdf",
        )

        settings = Settings(
            _env_file=None,
            pdf_archive_dir=str(archive_dir),
            llm_api_key="k",
            smtp_user="u",
            email_from="a@b.com",
            email_to="c@d.com",
        )

        archive_papers([paper], "2026-03-24", settings)

        # Paper should still be in DuckDB index (without pdf_path)
        db_path = archive_dir / "archive.duckdb"
        con = duckdb.connect(str(db_path))
        rows = con.execute("SELECT pdf_path FROM papers WHERE paper_id = '2603.99999'").fetchall()
        assert rows[0][0] is None
        con.close()

        # No PDF files copied
        day_dir = archive_dir / "2026-03-24"
        assert len(list(day_dir.glob("*.pdf"))) == 0
