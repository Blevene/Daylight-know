"""Semantic text chunking via Chonkie.

EARS coverage
─────────────
- Ubiquitous 2.1-2: use Chonkie ``SemanticChunker`` for semantic-boundary
  segmentation.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from chonkie import SemanticChunker
from chonkie.embeddings.model2vec import Model2VecEmbeddings
from model2vec import StaticModel

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "minishlab/potion-base-32M"

# Module-level singleton — avoids re-downloading the embedding model on every call.
_chunker_lock = threading.Lock()
_chunker: SemanticChunker | None = None


def _get_chunker() -> SemanticChunker:
    global _chunker
    if _chunker is not None:
        return _chunker
    with _chunker_lock:
        if _chunker is None:  # double-checked locking
            static_model = StaticModel.from_pretrained(_EMBEDDING_MODEL, force_download=False)
            embeddings = Model2VecEmbeddings(model=static_model)
            _chunker = SemanticChunker(
                embedding_model=embeddings,
                threshold=0.8,
                chunk_size=2048,
                similarity_window=3,
                skip_window=0,
            )
    return _chunker


@dataclass
class TextChunk:
    """A single semantically-bounded text chunk."""

    text: str
    chunk_index: int


def chunk_text(text: str) -> list[TextChunk]:
    """Split *text* into semantic chunks using Chonkie's ``SemanticChunker``.

    Returns an ordered list of ``TextChunk`` objects.
    """
    chunker = _get_chunker()
    raw_chunks = chunker.chunk(text)

    chunks = [TextChunk(text=c.text, chunk_index=idx) for idx, c in enumerate(raw_chunks)]
    logger.info("Produced %d semantic chunks.", len(chunks))
    return chunks
