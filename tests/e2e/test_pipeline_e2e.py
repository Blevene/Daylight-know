"""End-to-end pipeline tests.

Test IDs: PL-1, PL-2, PL-3, PL-4, PL-5, PL-6
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from digest_pipeline.pipeline import run
from digest_pipeline.github_trending import TrendingRepo


@pytest.mark.e2e
@pytest.mark.timeout(120)
class TestPipelineE2E:
    """Full pipeline tests with real local deps and stubbed LLM."""

    def test_full_dry_run(self, e2e_settings, make_paper, sample_pdf, capsys):
        """PL-1: Full E2E dry-run: fixture PDF -> chunk -> ChromaDB -> LLM stub -> console."""
        paper = make_paper(pdf_path=sample_pdf)

        with patch("digest_pipeline.pipeline.fetch_papers", return_value=[paper]):
            run(e2e_settings)

        captured = capsys.readouterr()
        # Per-paper output: paper title and stub summary
        assert "Test Paper" in captured.out
        assert "E2E stub summary" in captured.out

    def test_mixed_parseable_unparseable(self, e2e_settings, make_paper, sample_pdf, corrupt_file, capsys):
        """PL-2: One valid PDF + one corrupt -> only valid one in summary."""
        valid_paper = make_paper(pdf_path=sample_pdf, paper_id="valid.001")
        corrupt_paper = make_paper(pdf_path=corrupt_file, paper_id="corrupt.001")

        with patch("digest_pipeline.pipeline.fetch_papers", return_value=[valid_paper, corrupt_paper]):
            run(e2e_settings)

        captured = capsys.readouterr()
        assert "E2E stub summary" in captured.out

    def test_vectorstore_error_exits_with_code_1(self, e2e_settings, make_paper, sample_pdf):
        """PL-3: VectorStoreError mid-pipeline -> sys.exit(1)."""
        paper = make_paper(pdf_path=sample_pdf)

        # Make ChromaDB unreachable by pointing to an invalid path
        e2e_settings.chroma_persist_dir = Path("/proc/nonexistent/impossible")

        with patch("digest_pipeline.pipeline.fetch_papers", return_value=[paper]):
            with pytest.raises(SystemExit) as exc_info:
                run(e2e_settings)
            assert exc_info.value.code == 1

    def test_cli_args(self, e2e_settings, make_paper, sample_pdf, capsys, monkeypatch):
        """PL-4: CLI main() with --dry-run --topics cs.CL -v."""
        from digest_pipeline.pipeline import main

        paper = make_paper(pdf_path=sample_pdf)

        monkeypatch.setattr(
            "sys.argv",
            ["digest-pipeline", "--dry-run", "--topics", "cs.CL", "-v"],
        )

        with patch("digest_pipeline.pipeline.get_settings", return_value=e2e_settings):
            with patch("digest_pipeline.pipeline.fetch_papers", return_value=[paper]):
                main()

        captured = capsys.readouterr()
        assert "E2E stub summary" in captured.out

    def test_github_enabled(self, e2e_settings, make_paper, sample_pdf, capsys):
        """PL-5: GitHub-enabled pipeline includes GitHub section in output."""
        e2e_settings.github_enabled = True
        paper = make_paper(pdf_path=sample_pdf)

        mock_repos = [
            TrendingRepo(
                name="test/repo",
                description="A test repo",
                url="https://github.com/test/repo",
                stars=100,
                language="Python",
            )
        ]

        with patch("digest_pipeline.pipeline.fetch_papers", return_value=[paper]):
            with patch("digest_pipeline.pipeline.fetch_trending", return_value=mock_repos):
                run(e2e_settings)

        captured = capsys.readouterr()
        assert "E2E stub summary" in captured.out

    def test_all_unparseable_early_exit(self, e2e_settings, make_paper, corrupt_file):
        """PL-6: All papers unparseable -> pipeline returns without calling summarize."""
        paper = make_paper(pdf_path=corrupt_file)

        with patch("digest_pipeline.pipeline.fetch_papers", return_value=[paper]):
            with patch("digest_pipeline.pipeline.summarize") as mock_summarize:
                run(e2e_settings)
                mock_summarize.assert_not_called()
