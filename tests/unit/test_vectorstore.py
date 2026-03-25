"""Tests for vectorstore collection parameter passthrough."""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from digest_pipeline.chunker import TextChunk
from digest_pipeline.fetcher import Paper
from digest_pipeline.vectorstore import store_chunks, store_unparseable


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        paper_id="2401.00001",
        title="Test",
        authors=["Alice"],
        abstract="Abstract.",
        url="https://arxiv.org/abs/2401.00001",
        published=datetime(2025, 1, 15, tzinfo=timezone.utc),
        source="arxiv",
        pdf_path=None,
    )
    defaults.update(overrides)
    return Paper(**defaults)


@patch("digest_pipeline.vectorstore._get_collection")
def test_store_chunks_uses_passed_collection(mock_get_coll, make_settings):
    """When a collection is passed, _get_collection should NOT be called."""
    mock_coll = MagicMock()
    chunks = [TextChunk(text="hello", chunk_index=0)]

    store_chunks(_make_paper(), chunks, make_settings(), collection=mock_coll)

    mock_get_coll.assert_not_called()
    mock_coll.upsert.assert_called_once()


@patch("digest_pipeline.vectorstore._get_collection")
def test_store_chunks_falls_back_to_get_collection(mock_get_coll, make_settings):
    """Without a collection param, _get_collection is called as before."""
    mock_coll = MagicMock()
    mock_get_coll.return_value = mock_coll
    chunks = [TextChunk(text="hello", chunk_index=0)]

    store_chunks(_make_paper(), chunks, make_settings())

    mock_get_coll.assert_called_once()
    mock_coll.upsert.assert_called_once()


@patch("digest_pipeline.vectorstore._get_collection")
def test_store_unparseable_uses_passed_collection(mock_get_coll, make_settings):
    """When a collection is passed, _get_collection should NOT be called."""
    mock_coll = MagicMock()

    store_unparseable(_make_paper(), make_settings(), collection=mock_coll)

    mock_get_coll.assert_not_called()
    mock_coll.upsert.assert_called_once()
