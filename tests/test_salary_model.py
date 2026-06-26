"""
Testes unitários para src.models.salary_model.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.salary_model import (
    DENSE_ONLY_MODELS,
    INDIVIDUAL_CANDIDATES,
    predict_salary_range,
    train_nested_cv_reg,
)


class TestPredictSalaryRange:
    def test_returns_dict_with_keys(self, mock_regressor):
        vec = csr_matrix(np.array([[100000]]))
        result = predict_salary_range(vec, mock_regressor)
        assert isinstance(result, dict)
        assert "estimated_annual_usd" in result
        assert "range_low" in result
        assert "range_high" in result

    def test_range_contains_estimate(self, mock_regressor):
        vec = csr_matrix(np.array([[100000]]))
        result = predict_salary_range(vec, mock_regressor)
        assert result["range_low"] <= result["estimated_annual_usd"]
        assert result["estimated_annual_usd"] <= result["range_high"]

    def test_range_is_15_percent(self):
        class ConstantRegressor:
            def predict(self, X):
                return np.array([100000.0])

        reg = ConstantRegressor()
        vec = csr_matrix(np.array([[0]]))
        result = predict_salary_range(vec, reg)
        assert result["estimated_annual_usd"] == 100000
        assert result["range_low"] == 85000
        assert result["range_high"] == 115000

    def test_rounds_to_integer(self, mock_regressor):
        vec = csr_matrix(np.array([[100000]]))
        result = predict_salary_range(vec, mock_regressor)
        assert isinstance(result["estimated_annual_usd"], int)
        assert isinstance(result["range_low"], int)
        assert isinstance(result["range_high"], int)


# ── Fase 5 — MLPRegressor + Nested CV ─────────────────────────────


class TestMLPRegressor:
    def test_mlp_reg_in_candidates(self):
        assert "mlp" in INDIVIDUAL_CANDIDATES

    def test_mlp_reg_dense_only(self):
        assert "mlp" in DENSE_ONLY_MODELS

    def test_nested_cv_reg_returns_tuple(self, dense_matrix, dense_reg_target):
        name, params, scores, model = train_nested_cv_reg(
            dense_matrix, dense_reg_target, outer_cv=2, inner_cv=2, n_iter=4,
        )
        assert isinstance(name, str)
        assert isinstance(params, dict)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert hasattr(model, "predict")
