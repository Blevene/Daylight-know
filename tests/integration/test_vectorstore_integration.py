"""Integration tests for vectorstore.py — real ChromaDB.

Test IDs: V-1, V-2, V-3, V-4, V-5, V-6
"""

import pytest

from digest_pipeline.chunker import TextChunk
from digest_pipeline.config import Settings
from digest_pipeline.vectorstore import (
    StoredChunk,
    VectorStoreError,
    _get_collection,
    store_chunks,
    store_unparseable,
)


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestVectorstoreIntegration:
    """Tests that exercise real ChromaDB with temp directories."""

    def test_get_collection_happy_path(self, test_settings):
        """V-1: Create a PersistentClient with a temp dir, get-or-create collection."""
        collection = _get_collection(test_settings)
        assert collection is not None
        assert collection.name == "test_collection"

    def test_get_collection_failure(self):
        """V-2: Invalid/unwritable path raises VectorStoreError."""
        bad_settings = Settings(
            _env_file=None,
            llm_api_key="k",
            smtp_user="u",
            smtp_password="p",
            email_from="a@b.com",
            email_to="c@d.com",
            chroma_persist_dir="/proc/nonexistent/impossible/path",
            chroma_collection="test",
        )
        with pytest.raises(VectorStoreError):
            _get_collection(bad_settings)

    def test_store_chunks_happy_path(self, test_settings, make_paper):
        """V-3: Store 3 chunks, then query ChromaDB to verify documents, IDs, and metadata."""
        paper = make_paper()
        chunks = [
            TextChunk(text="First chunk about neural networks.", chunk_index=0),
            TextChunk(text="Second chunk about transformers.", chunk_index=1),
            TextChunk(text="Third chunk about attention mechanisms.", chunk_index=2),
        ]

        stored = store_chunks(paper, chunks, test_settings)

        assert len(stored) == 3
        assert all(isinstance(s, StoredChunk) for s in stored)

        # Verify IDs follow the pattern <paper_id>_chunk_<index>
        expected_ids = [
            "2401.00001_chunk_0",
            "2401.00001_chunk_1",
            "2401.00001_chunk_2",
        ]
        assert [s.doc_id for s in stored] == expected_ids

        # Read back from ChromaDB to verify persistence
        collection = _get_collection(test_settings)
        result = collection.get(ids=expected_ids, include=["documents", "metadatas"])

        assert len(result["ids"]) == 3
        assert result["documents"][0] == "First chunk about neural networks."
        assert result["documents"][2] == "Third chunk about attention mechanisms."

        # Verify all 6 metadata fields
        meta = result["metadatas"][0]
        assert meta["source"] == "arxiv"
        assert meta["title"] == paper.title
        assert meta["authors"] == "Alice Researcher, Bob Scientist"
        assert meta["url"] == paper.url
        assert meta["published_date"] == paper.published.isoformat()
        assert meta["chunk_index"] == 0

    def test_store_chunks_upsert_dedup(self, test_settings, make_paper):
        """V-4: Calling store_chunks twice with same paper produces no duplicates."""
        paper = make_paper()
        chunks = [TextChunk(text="Only chunk.", chunk_index=0)]

        store_chunks(paper, chunks, test_settings)
        store_chunks(paper, chunks, test_settings)  # second call

        collection = _get_collection(test_settings)
        result = collection.get(ids=["2401.00001_chunk_0"])
        # Should be exactly 1 document, not 2
        assert len(result["ids"]) == 1

    def test_store_unparseable(self, test_settings, make_paper):
        """V-5: Store unparseable paper and verify [unparseable] text + metadata."""
        paper = make_paper()
        store_unparseable(paper, test_settings)

        collection = _get_collection(test_settings)
        result = collection.get(
            ids=["2401.00001_unparseable"],
            include=["documents", "metadatas"],
        )

        assert len(result["ids"]) == 1
        assert result["documents"][0] == "[unparseable]"

        meta = result["metadatas"][0]
        assert meta["chunk_index"] == -1
        assert meta["unparseable"] is True
        assert meta["source"] == "arxiv"
        assert meta["title"] == paper.title

    def test_store_chunks_empty_list(self, test_settings, make_paper):
        """V-6: Passing chunks=[] returns empty StoredChunk list without crashing."""
        paper = make_paper()
        stored = store_chunks(paper, [], test_settings)

        assert stored == []
