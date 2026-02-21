"""ChromaDB vector store for chunk storage and retrieval.

EARS coverage
─────────────
- Ubiquitous 2.1-3: store chunks with embeddings and metadata in ChromaDB.
- Unwanted 2.4-3: halt and alert on ChromaDB connection failure.
- Data schema §3: enforce the defined metadata mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb

from digest_pipeline.chunker import TextChunk
from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when ChromaDB is unreachable (EARS 2.4-3)."""


@dataclass
class StoredChunk:
    doc_id: str
    source: str
    title: str
    authors: str
    url: str
    published_date: str
    chunk_index: int
    text: str


def _get_collection(settings: Settings) -> chromadb.Collection:
    try:
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        collection = client.get_or_create_collection(name=settings.chroma_collection)
    except Exception as exc:
        msg = f"ChromaDB connection failed: {exc}"
        logger.critical(msg)
        raise VectorStoreError(msg) from exc
    return collection


def store_chunks(
    paper: Paper,
    chunks: list[TextChunk],
    settings: Settings,
    *,
    source: str = "arxiv",
) -> list[StoredChunk]:
    """Persist *chunks* for *paper* into ChromaDB.

    Each chunk is stored with the metadata schema defined in §3 of the
    design document.  If ChromaDB is unreachable, ``VectorStoreError``
    is raised (EARS 2.4-3).
    """
    if not chunks:
        return []

    collection = _get_collection(settings)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    stored: list[StoredChunk] = []

    for chunk in chunks:
        doc_id = f"{paper.arxiv_id}_chunk_{chunk.chunk_index}"
        metadata = {
            "source": source,
            "title": paper.title,
            "authors": ", ".join(paper.authors),
            "url": paper.url,
            "published_date": paper.published.isoformat(),
            "chunk_index": chunk.chunk_index,
        }
        ids.append(doc_id)
        documents.append(chunk.text)
        metadatas.append(metadata)
        stored.append(
            StoredChunk(
                doc_id=doc_id,
                source=source,
                title=paper.title,
                authors=metadata["authors"],
                url=paper.url,
                published_date=metadata["published_date"],
                chunk_index=chunk.chunk_index,
                text=chunk.text,
            )
        )

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("Stored %d chunks for paper %s.", len(stored), paper.arxiv_id)
    return stored


def store_unparseable(paper: Paper, settings: Settings) -> None:
    """Record an unparseable document in ChromaDB with a flag (EARS 2.4-2)."""
    collection = _get_collection(settings)
    doc_id = f"{paper.arxiv_id}_unparseable"
    collection.upsert(
        ids=[doc_id],
        documents=["[unparseable]"],
        metadatas=[
            {
                "source": "arxiv",
                "title": paper.title,
                "authors": ", ".join(paper.authors),
                "url": paper.url,
                "published_date": paper.published.isoformat(),
                "chunk_index": -1,
                "unparseable": True,
            }
        ],
    )
    logger.warning("Flagged paper %s as unparseable in ChromaDB.", paper.arxiv_id)
