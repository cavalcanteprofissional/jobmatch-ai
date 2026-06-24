"""
Testes unitários para src.models.recommender.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.recommender import rank_jobs


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
