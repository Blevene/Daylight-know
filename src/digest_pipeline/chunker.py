"""Semantic text chunking via Chonkie.

EARS coverage
─────────────
- Ubiquitous 2.1-2: use Chonkie ``SemanticChunker`` for semantic-boundary
  segmentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chonkie import SemanticChunker

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single semantically-bounded text chunk."""

    text: str
    chunk_index: int


def chunk_text(text: str) -> list[TextChunk]:
    """Split *text* into semantic chunks using Chonkie's ``SemanticChunker``.

    Returns an ordered list of ``TextChunk`` objects.
    """
    chunker = SemanticChunker(
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=512,
        similarity_threshold=0.5,
    )
    raw_chunks = chunker.chunk(text)

    chunks = [
        TextChunk(text=c.text, chunk_index=idx)
        for idx, c in enumerate(raw_chunks)
    ]
    logger.info("Produced %d semantic chunks.", len(chunks))
    return chunks
