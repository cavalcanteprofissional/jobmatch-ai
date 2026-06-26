"""
Testes unitários para src.models.recommender.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.models.recommender import rank_jobs, rerank_with_cross_encoder


class TestRankJobs:
    def test_returns_top_k(self, sample_jobs_df, sample_resume_vec):
        n = len(sample_jobs_df)
        rng = np.random.default_rng(42)
        jobs_matrix = csr_matrix(rng.random((n, 20)))

        result = rank_jobs(sample_resume_vec, jobs_matrix, sample_jobs_df, top_k=3)
        assert len(result) == 3

    def test_sorted_by_score_desc(self, sample_jobs_df, sample_resume_vec):
        n = len(sample_jobs_df)
        rng = np.random.default_rng(42)
        jobs_matrix = csr_matrix(rng.random((n, 20)))

        result = rank_jobs(sample_resume_vec, jobs_matrix, sample_jobs_df, top_k=5)
        scores = result["adherence_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_contains_expected_columns(self, sample_jobs_df, sample_resume_vec):
        n = len(sample_jobs_df)
        rng = np.random.default_rng(42)
        jobs_matrix = csr_matrix(rng.random((n, 20)))

        result = rank_jobs(sample_resume_vec, jobs_matrix, sample_jobs_df, top_k=3)
        expected = {"title", "adherence_score", "fit_label"}
        assert expected.issubset(result.columns)

    def test_fit_threshold(self, sample_jobs_df, sample_resume_vec):
        n = len(sample_jobs_df)
        jobs_matrix = csr_matrix(np.zeros((n, 20)))  # scores zero
        result = rank_jobs(
            sample_resume_vec, jobs_matrix, sample_jobs_df,
            top_k=3, fit_threshold=50.0,
        )
        assert all(label == "❌ No Fit" for label in result["fit_label"])


# ── Fase 5 — Cross-encoder Tests ──────────────────────────────────


class TestCrossEncoder:
    def test_rerank_cross_encoder_mock(self, mock_cross_encoder, sample_jobs_df):
        result = rerank_with_cross_encoder(
            "data science python", sample_jobs_df, top_k=3,
        )
        assert len(result) == 3
        assert "adherence_score" in result.columns
        assert "fit_label" in result.columns
        assert all(result["adherence_score"] >= 0)

    def test_rank_jobs_cross_encoder_flag(self, sample_jobs_df, sample_resume_vec):
        n = len(sample_jobs_df)
        jobs_matrix = csr_matrix(np.random.default_rng(42).random((n, 20)))
        result = rank_jobs(
            sample_resume_vec, jobs_matrix, sample_jobs_df,
            top_k=3, resume_text="data science", use_cross_encoder=True,
        )
        assert len(result) == 3
        assert "adherence_score" in result.columns

    @pytest.mark.slow
    def test_cross_encoder_real_scores(self):
        pytest.importorskip("sentence_transformers")
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder("cross-encoder/stsb-MiniLM-L-6-v2")
            identical = float(model.predict([["python", "python"]])[0])
            opposite = float(model.predict([["python", "completely unrelated topic"]])[0])
            assert identical > 0.8, f"Par idêntico deveria ter score alto: {identical}"
            assert opposite < 0.6, f"Par diferente deveria ter score baixo: {opposite}"
        except Exception as e:
            pytest.skip(f"Cross-encoder não disponível: {e}")
