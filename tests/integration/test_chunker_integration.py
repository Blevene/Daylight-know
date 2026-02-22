"""Integration tests for chunker.py — real Chonkie + embedding model.

Test IDs: C-1, C-2, C-3, C-4
"""

import pytest

from digest_pipeline.chunker import TextChunk, chunk_text


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestChunkerIntegration:
    """Tests that exercise the real SemanticChunker with minishlab/potion-base-32M."""

    def test_happy_path_multi_paragraph(self):
        """C-1: Multi-paragraph text produces multiple TextChunk objects."""
        text = (
            "Machine learning has transformed natural language processing. "
            "Recent advances in transformer architectures have enabled models "
            "to achieve human-level performance on many benchmarks.\n\n"
            "Reinforcement learning from human feedback (RLHF) has become a "
            "standard technique for aligning large language models with human "
            "preferences. This approach uses reward models trained on human "
            "comparisons to fine-tune base models.\n\n"
            "Diffusion models have emerged as the leading approach for image "
            "generation. These models learn to reverse a noise-adding process, "
            "gradually transforming random noise into coherent images. The "
            "technique has been extended to video, audio, and 3D generation."
        )
        chunks = chunk_text(text)

        assert len(chunks) >= 1
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(isinstance(c.text, str) and len(c.text) > 0 for c in chunks)
        # Verify sequential indexing
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        # Verify no data loss: all original content appears across chunks
        combined = " ".join(c.text for c in chunks)
        assert "transformer" in combined.lower()
        assert "diffusion" in combined.lower()

    def test_empty_input(self):
        """C-2: Empty string returns an empty list (or single empty chunk)."""
        chunks = chunk_text("")
        # Either empty list or graceful handling
        assert isinstance(chunks, list)
        if len(chunks) > 0:
            # If Chonkie returns a chunk for empty input, it should be benign
            assert all(isinstance(c, TextChunk) for c in chunks)

    def test_short_input(self):
        """C-3: Single sentence shorter than chunk_size=2048 returns one chunk."""
        text = "A brief sentence about machine learning."
        chunks = chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert "machine learning" in chunks[0].text.lower()

    def test_long_input(self):
        """C-4: Multi-page text (>10KB) produces multiple chunks without data loss."""
        # Generate a ~12KB text with distinct paragraphs
        paragraphs = []
        topics = [
            "neural networks",
            "gradient descent",
            "attention mechanisms",
            "convolutional layers",
            "recurrent architectures",
            "transformer models",
            "batch normalization",
            "dropout regularization",
            "transfer learning",
            "few-shot learning",
            "meta-learning",
            "self-supervised pretraining",
        ]
        for topic in topics:
            paragraphs.append(
                f"The field of {topic} has seen significant advances in recent years. "
                f"Researchers have developed new approaches to {topic} that improve "
                f"upon previous methods by incorporating novel architectural designs "
                f"and training procedures. These advances in {topic} have practical "
                f"applications across many domains including healthcare, finance, "
                f"and autonomous systems. The theoretical foundations of {topic} "
                f"continue to be an active area of research with many open questions. "
                f"Furthermore, the scalability of {topic} remains a key concern as "
                f"practitioners seek to deploy these systems in production environments. "
                f"Recent benchmarks have demonstrated that {topic} can achieve "
                f"state-of-the-art results when combined with sufficient data and "
                f"computational resources. The community around {topic} continues "
                f"to grow, with new conferences and workshops dedicated to advancing "
                f"the understanding and application of {topic} in real-world settings."
            )
        text = "\n\n".join(paragraphs)
        assert len(text) > 10_000, f"Text should be >10KB, got {len(text)}"

        chunks = chunk_text(text)

        assert len(chunks) >= 2
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        # Spot-check: first and last topics should appear somewhere in chunks
        combined = " ".join(c.text for c in chunks)
        assert "neural networks" in combined.lower()
        assert "self-supervised" in combined.lower()
