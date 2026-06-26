import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from src.models.classifier import (
    HYPERPARAM_GRIDS,
    INDIVIDUAL_CANDIDATES,
    _is_sparse_compatible,
    predict,
    train_nested_cv_clf,
)


class TestPredict:
    def test_predict_returns_tuple(self):
        clf, _ = _make_trained_logistic()
        vec = csr_matrix(np.array([[1, 0]]))
        label, prob = predict(vec, clf)
        assert isinstance(label, str)
        assert label in ("Fit", "No Fit")
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_proba_logistic(self):
        clf, _ = _make_trained_logistic()
        vec = csr_matrix(np.array([[1, 1]]))
        label, prob = predict(vec, clf)
        assert prob > 0.5

    def test_predict_linear_svc(self):
        from sklearn.svm import LinearSVC
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y = np.array([1, 0, 1, 0])
        svc = LinearSVC()
        svc.fit(X, y)
        label, prob = predict(csr_matrix(np.array([[1, 1]])), svc)
        assert label in ("Fit", "No Fit")
        assert isinstance(prob, float)

    def test_predict_no_proba(self):
        class DummyModel:
            def predict(self, X):
                return np.array([1])
        label, prob = predict(csr_matrix(np.array([[0, 0]])), DummyModel())
        assert label == "Fit"
        assert prob == 1.0


class TestTrainBest:
    def test_train_best_returns_tuple(self):
        from src.models.classifier import train_best
        rng = np.random.default_rng(42)
        X = csr_matrix(rng.random((10, 5)))
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        name, model = train_best(X, y, cv=2)
        assert isinstance(name, str)
        assert hasattr(model, "predict")


# ── Fase 5 — MLP + GaussianNB + Dense Only + Nested CV ────────────


class TestMLPAndGaussianNB:
    def test_mlp_in_candidates(self):
        assert "mlp" in INDIVIDUAL_CANDIDATES

    def test_gaussian_nb_in_candidates(self):
        assert "gaussian_nb" in INDIVIDUAL_CANDIDATES

    def test_dense_only_skips_sparse(self):
        X_sparse = csr_matrix(np.zeros((5, 10)))
        assert issparse(X_sparse)
        assert not _is_sparse_compatible("mlp", X_sparse)
        assert not _is_sparse_compatible("gaussian_nb", X_sparse)

    def test_dense_only_allows_dense(self):
        X_dense = np.zeros((5, 10))
        assert not issparse(X_dense)
        assert _is_sparse_compatible("mlp", X_dense)
        assert _is_sparse_compatible("gaussian_nb", X_dense)

    def test_nested_cv_clf_returns_tuple(self, dense_matrix, dense_target):
        name, params, scores, model = train_nested_cv_clf(
            dense_matrix, dense_target, outer_cv=2, inner_cv=2, n_iter=4,
        )
        assert isinstance(name, str)
        assert isinstance(params, dict)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert hasattr(model, "predict")

    def test_nested_cv_clf_all_dense_models(self, dense_matrix, dense_target):
        name, params, scores, model = train_nested_cv_clf(
            dense_matrix, dense_target, outer_cv=2, inner_cv=2, n_iter=4,
        )
        assert name in INDIVIDUAL_CANDIDATES or name in (
            "stacking", "voting",
        )

    def test_mlp_grid_present(self):
        grid = HYPERPARAM_GRIDS["mlp"]
        assert "hidden_layer_sizes" in grid
        assert "alpha" in grid
        assert "learning_rate_init" in grid

    def test_gaussian_nb_grid_present(self):
        grid = HYPERPARAM_GRIDS["gaussian_nb"]
        assert "var_smoothing" in grid


def _make_trained_logistic():
    from sklearn.linear_model import LogisticRegression
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([1, 0, 1, 0])
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf, X
