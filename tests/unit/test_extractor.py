"""Tests for the PDF text extraction module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from digest_pipeline.extractor import extract_text


@patch("digest_pipeline.extractor.PdfReader")
def test_extract_text_success(mock_reader_cls):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Hello world"
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]
    mock_reader_cls.return_value = mock_reader

    result = extract_text(Path("/fake/path.pdf"), "1234.56789")
    assert result.parseable is True
    assert "Hello world" in result.text


@patch("digest_pipeline.extractor.PdfReader")
def test_extract_text_empty_marks_unparseable(mock_reader_cls):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]
    mock_reader_cls.return_value = mock_reader

    result = extract_text(Path("/fake/path.pdf"), "1234.56789")
    assert result.parseable is False
    assert result.text == ""


@patch("digest_pipeline.extractor.PdfReader", side_effect=RuntimeError("corrupt"))
def test_extract_text_exception_marks_unparseable(mock_reader_cls):
    result = extract_text(Path("/fake/path.pdf"), "1234.56789")
    assert result.parseable is False
