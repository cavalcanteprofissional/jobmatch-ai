from pathlib import Path

import numpy as np
import pytest

from src.models.vectorizer import (
    SentenceBertVectorizer,
    fit_vectorizer,
    load_sbert_vectorizer,
    load_vectorizer,
    transform,
)


class TestFitVectorizer:
    def test_fit_on_corpus(self, sample_tfidf_corpus):
        vec = fit_vectorizer(sample_tfidf_corpus, max_features=100)
        assert vec is not None
        assert hasattr(vec, "vocabulary_")
        assert len(vec.vocabulary_) <= 100

    def test_fit_and_save(self, sample_tfidf_corpus, tmp_path):
        save_path = tmp_path / "tfidf.pkl"
        vec = fit_vectorizer(sample_tfidf_corpus, max_features=50, save_path=save_path)
        assert save_path.exists()

    def test_min_df_filters_rare(self):
        corpus = [
            "unique_xyz_abc data science python",
            "data science python sql",
            "data science python sql",
            "data science python sql",
            "data science python sql",
        ]
        vec = fit_vectorizer(corpus, max_features=100, save_path=None)
        assert "unique_xyz_abc" not in vec.vocabulary_


class TestLoadVectorizer:
    def test_load_saved(self, sample_tfidf_corpus, tmp_path):
        save_path = tmp_path / "tfidf.pkl"
        fit_vectorizer(sample_tfidf_corpus, max_features=50, save_path=save_path)
        loaded = load_vectorizer(save_path)
        assert loaded is not None
        assert len(loaded.vocabulary_) > 0

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_vectorizer(Path("/nonexistent/path.pkl"))


class TestTransform:
    def test_transform_shape(self, sample_tfidf_corpus):
        vec = fit_vectorizer(sample_tfidf_corpus, max_features=100)
        matrix = transform(sample_tfidf_corpus, vec)
        assert matrix.shape[0] == len(sample_tfidf_corpus)
        assert matrix.shape[1] == len(vec.vocabulary_)

    def test_transform_new_text(self, sample_tfidf_corpus):
        vec = fit_vectorizer(sample_tfidf_corpus, max_features=100)
        new_texts = ["python data science"]
        matrix = transform(new_texts, vec)
        assert matrix.shape[0] == 1
        assert matrix.nnz > 0


# ── Fase 5 — SBERT Tests ──────────────────────────────────────────


class TestSentenceBertVectorizer:
    def test_sbert_init(self):
        sbert = SentenceBertVectorizer(model_name="fake-model")
        assert sbert.model_name == "fake-model"
        assert sbert._model is None

    def test_sbert_transform_mock(self, mock_sbert):
        sbert = SentenceBertVectorizer(model_name="mock-model")
        texts = ["data science python tensorflow"]
        emb = sbert.transform(texts)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (1, 384)
        assert np.isfinite(emb).all()

    def test_sbert_fit_save(self, mock_sbert, tmp_path):
        from src.models.vectorizer import SBERT_EMBEDDINGS_PATH
        original_emb_path = SBERT_EMBEDDINGS_PATH
        sbert = SentenceBertVectorizer(model_name="mock-model")
        texts = ["text one", "text two", "text three"]
        save_path = tmp_path / "sbert.pkl"
        result = sbert.fit(texts, save_path=save_path)
        assert result is sbert
        assert save_path.exists()
        loaded_emb = np.load(SBERT_EMBEDDINGS_PATH)
        assert loaded_emb.shape == (3, 384)

    def test_sbert_load_embeddings(self, mock_sbert, tmp_path):
        sbert = SentenceBertVectorizer(model_name="mock-model")
        texts = ["hello world"]
        sbert.fit(texts, save_path=tmp_path / "sbert.pkl")
        emb = sbert.load_embeddings()
        assert isinstance(emb, np.ndarray)
        assert emb.shape[1] == 384

    def test_load_sbert_vectorizer(self, mock_sbert, tmp_path):
        from joblib import dump

        sbert = SentenceBertVectorizer(model_name="mock-model")
        save_path = tmp_path / "sbert.pkl"
        dump(sbert, save_path)
        loaded = load_sbert_vectorizer(save_path)
        assert isinstance(loaded, SentenceBertVectorizer)
        assert loaded.model_name == "mock-model"

    @pytest.mark.slow
    def test_sbert_consistency_real(self):
        sbert = SentenceBertVectorizer()
        texts = ["data science with python and tensorflow"]
        emb1 = sbert.transform(texts)
        emb2 = sbert.transform(texts)
        diff = np.linalg.norm(emb1 - emb2)
        assert diff < 1e-5, f"Embeddings inconsistentes: norma L2 = {diff}"
