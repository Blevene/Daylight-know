"""Tests for chunker thread safety."""

import threading
from unittest.mock import patch, MagicMock

from digest_pipeline.chunker import _get_chunker


@patch("digest_pipeline.chunker.StaticModel.from_pretrained")
@patch("digest_pipeline.chunker.Model2VecEmbeddings")
@patch("digest_pipeline.chunker.SemanticChunker")
def test_get_chunker_concurrent_calls_create_single_instance(
    mock_chunker_cls, mock_embeddings_cls, mock_model,
):
    """Multiple threads calling _get_chunker() should create only one instance."""
    import digest_pipeline.chunker as mod
    mod._chunker = None  # reset singleton

    mock_model.return_value = MagicMock()
    mock_instance = MagicMock()
    mock_chunker_cls.return_value = mock_instance

    results = []
    def call_get_chunker():
        results.append(_get_chunker())

    threads = [threading.Thread(target=call_get_chunker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 10
    # All threads should get the same instance
    assert all(r is results[0] for r in results)
    # Model should only be loaded once
    mock_model.assert_called_once()

    mod._chunker = None  # cleanup
