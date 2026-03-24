"""Tests for the PDF archiver module."""

from digest_pipeline.archiver import _sanitize_filename


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
