import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.classifier import predict


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


def _make_trained_logistic():
    from sklearn.linear_model import LogisticRegression
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([1, 0, 1, 0])
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf, X
