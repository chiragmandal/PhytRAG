"""
Unit tests for chunking and retrieval logic.
No running services required.
"""
import pytest

from ingestion.chunk_and_embed import chunk_text


class TestChunkText:
    def test_single_chunk_short_text(self):
        text = "The GA20 oxidase enzyme catalyses gibberellin biosynthesis in Arabidopsis."
        chunks = chunk_text(text, chunk_size=400, overlap=40)
        assert len(chunks) == 1
        assert "GA20" in chunks[0]

    def test_multiple_chunks_long_text(self):
        # 1000 words
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=400, overlap=40)
        # Expected: ceil((1000 - 400) / (400 - 40)) + 1 = ~3 chunks
        assert len(chunks) >= 2

    def test_overlap_creates_shared_content(self):
        words = [f"w{i}" for i in range(100)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # The last 10 words of chunk 0 should appear in chunk 1
        chunk0_words = set(chunks[0].split())
        chunk1_words = set(chunks[1].split())
        assert len(chunk0_words & chunk1_words) > 0

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\t  ") == []

    def test_chunk_size_respected(self):
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 100, f"Chunk too large: {word_count} words"
