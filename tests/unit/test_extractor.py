"""Tests for the PDF text extraction module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from digest_pipeline.extractor import extract_text


@patch("digest_pipeline.extractor.fitz.open")
def test_extract_text_success(mock_open):
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Hello world"
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.close = MagicMock()
    mock_open.return_value = mock_doc

    result = extract_text(Path("/fake/path.pdf"), "1234.56789")
    assert result.parseable is True
    assert "Hello world" in result.text


@patch("digest_pipeline.extractor.fitz.open")
def test_extract_text_empty_marks_unparseable(mock_open):
    mock_page = MagicMock()
    mock_page.get_text.return_value = ""
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.close = MagicMock()
    mock_open.return_value = mock_doc

    result = extract_text(Path("/fake/path.pdf"), "1234.56789")
    assert result.parseable is False
    assert result.text == ""


@patch("digest_pipeline.extractor.fitz.open", side_effect=RuntimeError("corrupt"))
def test_extract_text_exception_marks_unparseable(mock_open):
    result = extract_text(Path("/fake/path.pdf"), "1234.56789")
    assert result.parseable is False
