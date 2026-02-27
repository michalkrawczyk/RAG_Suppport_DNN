"""Tests for TextEmbedder class, including create_embeddings_batched."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _mock_sentence_transformer(dim: int = 8):
    """Return a minimal SentenceTransformer-compatible mock."""
    mock = MagicMock(spec=["encode"])
    mock.encode.side_effect = lambda texts, **kw: np.random.rand(len(texts), dim).astype(
        np.float32
    )
    return mock


def _mock_langchain_embedder(dim: int = 8):
    """Return a minimal LangChain Embeddings-compatible mock."""
    mock = MagicMock()
    mock.embed_documents.side_effect = lambda texts: np.random.rand(
        len(texts), dim
    ).tolist()
    mock.embed_query.side_effect = lambda text: np.random.rand(dim).tolist()
    return mock


class TestTextEmbedderImport:
    """Smoke-test: class is importable from both locations."""

    def test_import_from_module(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        assert TextEmbedder is not None, "TextEmbedder must be importable from text_embedder"

    def test_import_from_package(self):
        from RAG_supporters.embeddings import TextEmbedder

        assert TextEmbedder is not None, "TextEmbedder must be exported from embeddings package"


class TestTextEmbedderInit:
    """Test __init__ model detection and wrapping."""

    def test_init_with_sentence_transformer(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer()
        embedder = TextEmbedder(embedding_model=mock_model)

        assert embedder.model_type == "sentence-transformers", (
            "Model with .encode() must be detected as sentence-transformers"
        )

    def test_init_with_langchain_model(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_langchain_embedder()
        embedder = TextEmbedder(embedding_model=mock_model)

        assert embedder.model_type == "langchain", (
            "Model with .embed_documents/.embed_query must be detected as langchain"
        )

    def test_init_unknown_model_raises(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        with pytest.raises(ValueError, match="Unable to detect model type"):
            TextEmbedder(embedding_model=object())

    def test_init_explicit_model_name(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer()
        embedder = TextEmbedder(embedding_model=mock_model, model_name="my-model")

        assert embedder.model_name == "my-model", (
            "Explicit model_name must be respected over auto-detection"
        )


class TestCreateEmbeddings:
    """Tests for create_embeddings (existing method, full-dict return)."""

    def test_returns_dict_for_all_inputs(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=4)
        embedder = TextEmbedder(embedding_model=mock_model)

        texts = ["alpha", "beta", "gamma"]
        result = embedder.create_embeddings(texts)

        assert isinstance(result, dict), "create_embeddings must return a dict"
        assert len(result) == 3, "All unique texts must have embeddings"
        for text in ["alpha", "beta", "gamma"]:
            assert text in result, f"'{text}' must be present as a key"
            assert isinstance(result[text], np.ndarray), "Values must be np.ndarray"

    def test_deduplication(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=4)
        embedder = TextEmbedder(embedding_model=mock_model)

        result = embedder.create_embeddings(["word", "word", "word"])
        assert len(result) == 1, "Duplicate strings must be deduplicated"

    def test_empty_list_raises(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer()
        embedder = TextEmbedder(embedding_model=mock_model)

        with pytest.raises(ValueError, match="cannot be empty"):
            embedder.create_embeddings([])


class TestCreateEmbeddingsBatched:
    """Tests for the new create_embeddings_batched generator."""

    def test_yields_dicts(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=4)
        embedder = TextEmbedder(embedding_model=mock_model)

        texts = ["a", "b", "c", "d"]
        batches = list(embedder.create_embeddings_batched(texts, batch_size=2, show_progress=False))

        assert len(batches) == 2, "4 texts with batch_size=2 must yield exactly 2 batches"
        for batch in batches:
            assert isinstance(batch, dict), "Each yielded item must be a dict"
            for v in batch.values():
                assert isinstance(v, np.ndarray), "Each embedding must be np.ndarray"

    def test_all_texts_covered(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=4)
        embedder = TextEmbedder(embedding_model=mock_model)

        texts = ["x", "y", "z", "w", "v"]
        all_keys: set = set()
        for batch in embedder.create_embeddings_batched(texts, batch_size=2, show_progress=False):
            all_keys.update(batch.keys())

        assert all_keys == set(texts), "All input texts must appear across all yielded batches"

    def test_last_batch_partial(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=4)
        embedder = TextEmbedder(embedding_model=mock_model)

        texts = ["a", "b", "c"]  # 3 texts, batch_size=2 → batches of 2 and 1
        batches = list(embedder.create_embeddings_batched(texts, batch_size=2, show_progress=False))

        assert len(batches) == 2, "3 texts with batch_size=2 must yield 2 batches"
        sizes = sorted(len(b) for b in batches)
        assert sizes == [1, 2], "Batch sizes must be [2, 1] (last batch is partial)"

    def test_single_batch_when_batch_size_ge_len(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=4)
        embedder = TextEmbedder(embedding_model=mock_model)

        texts = ["one", "two"]
        batches = list(
            embedder.create_embeddings_batched(texts, batch_size=100, show_progress=False)
        )

        assert len(batches) == 1, "batch_size >= len(texts) must yield exactly one batch"
        assert len(batches[0]) == 2, "Single batch must contain all texts"

    def test_deduplication_applied_before_batching(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=4)
        embedder = TextEmbedder(embedding_model=mock_model)

        texts = ["dup", "dup", "dup", "unique"]
        all_keys: set = set()
        for batch in embedder.create_embeddings_batched(texts, batch_size=10, show_progress=False):
            all_keys.update(batch.keys())

        assert len(all_keys) == 2, (
            "Duplicates must be removed before batching; only unique texts should appear"
        )

    def test_empty_list_raises(self):
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=4)
        embedder = TextEmbedder(embedding_model=mock_model)

        with pytest.raises(ValueError, match="cannot be empty"):
            list(embedder.create_embeddings_batched([]))

    def test_is_generator(self):
        """create_embeddings_batched must return a generator, not a list."""
        import types
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=4)
        embedder = TextEmbedder(embedding_model=mock_model)

        result = embedder.create_embeddings_batched(["text"], show_progress=False)
        assert isinstance(result, types.GeneratorType), (
            "create_embeddings_batched must be a generator (uses yield)"
        )

    def test_batched_vs_full_same_keys(self):
        """Batched and non-batched must produce identical key sets."""
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=4)
        embedder = TextEmbedder(embedding_model=mock_model)

        texts = ["machine learning", "data science", "neural networks"]
        full = embedder.create_embeddings(texts)

        batched_keys: set = set()
        for batch in embedder.create_embeddings_batched(texts, batch_size=2, show_progress=False):
            batched_keys.update(batch.keys())

        assert full.keys() == batched_keys, (
            "create_embeddings and create_embeddings_batched must produce the same key set"
        )

    def test_normalize_embeddings_flag(self):
        """normalize_embeddings=True must produce unit-norm vectors."""
        from RAG_supporters.embeddings.text_embedder import TextEmbedder

        mock_model = _mock_sentence_transformer(dim=8)
        embedder = TextEmbedder(embedding_model=mock_model)

        texts = ["alpha", "beta"]
        for batch in embedder.create_embeddings_batched(
            texts, batch_size=2, show_progress=False, normalize_embeddings=True
        ):
            for text, emb in batch.items():
                norm = np.linalg.norm(emb)
                assert abs(norm - 1.0) < 1e-5, (
                    f"Embedding for '{text}' must be unit-norm when normalize_embeddings=True, "
                    f"got norm={norm:.6f}"
                )
