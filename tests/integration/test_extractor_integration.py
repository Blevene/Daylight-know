"""Integration tests for extractor.py — real PyMuPDF.

Test IDs: E-1, E-2, E-3
"""

import pytest

from digest_pipeline.extractor import extract_text


@pytest.mark.integration
@pytest.mark.timeout(15)
class TestExtractorIntegration:
    """Tests that exercise real PyMuPDF on actual files."""

    def test_real_pdf_extraction(self, sample_pdf):
        """E-1: Real PDF → text extraction with parseable=True."""
        result = extract_text(sample_pdf, "test.00001")

        assert result.parseable is True
        assert len(result.text) > 0
        assert "Introduction" in result.text or "Semantic Chunking" in result.text

    def test_multi_page_pdf(self, sample_pdf):
        """E-2: Multi-page PDF contains content from all pages joined by newline."""
        result = extract_text(sample_pdf, "test.00002")

        assert result.parseable is True
        # Page 1 content
        assert "Page 1" in result.text or "Introduction" in result.text
        # Page 2 content
        assert "Page 2" in result.text or "Results" in result.text

    def test_corrupt_file(self, corrupt_file):
        """E-3: Non-PDF file → parseable=False."""
        result = extract_text(corrupt_file, "test.corrupt")

        assert result.parseable is False
        assert result.text == ""
