from pathlib import Path

import pytest

from src.models.vectorizer import fit_vectorizer, load_vectorizer, transform


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
