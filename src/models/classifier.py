import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.special import expit
from scipy.sparse import issparse
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CLASSIFIER_PATH = Path("data/models/classifier.pkl")


def _to_dense(X):
    return X.toarray() if issparse(X) else X


INDIVIDUAL_CANDIDATES: dict[str, tuple[type, dict]] = {
    "logistic_regression": (
        LogisticRegression,
        {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
    ),
    "random_forest": (
        RandomForestClassifier,
        {"class_weight": "balanced", "n_jobs": -1, "random_state": 42},
    ),
    "svm": (
        LinearSVC,
        {"class_weight": "balanced", "max_iter": 2000, "random_state": 42},
    ),
    "extra_trees": (
        ExtraTreesClassifier,
        {"class_weight": "balanced", "n_jobs": -1, "random_state": 42},
    ),
    "mlp": (
        MLPClassifier,
        {"max_iter": 300, "random_state": 42, "early_stopping": True,
         "n_iter_no_change": 10},
    ),
    "gaussian_nb": (
        GaussianNB,
        {},
    ),
}

if _HAS_XGB:
    INDIVIDUAL_CANDIDATES["xgboost"] = (
        XGBClassifier,
        {"eval_metric": "logloss", "random_state": 42, "verbosity": 0,
         "tree_method": "hist", "n_jobs": -1},
    )

if _HAS_LGBM:
    INDIVIDUAL_CANDIDATES["lightgbm"] = (
        LGBMClassifier,
        {"random_state": 42, "verbosity": -1, "n_jobs": -1,
         "class_weight": "balanced"},
    )

if _HAS_CATBOOST:
    INDIVIDUAL_CANDIDATES["catboost"] = (
        CatBoostClassifier,
        {"random_seed": 42, "verbose": 0, "allow_writing_files": False},
    )

HYPERPARAM_GRIDS: dict[str, dict] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs"],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
    },
    "svm": {
        "C": [0.1, 0.5, 1.0, 5.0],
        "loss": ["squared_hinge"],
    },
    "extra_trees": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
    "mlp": {
        "hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.001, 0.01],
    },
    "gaussian_nb": {
        "var_smoothing": [1e-9, 1e-8, 1e-7],
    },
}

if _HAS_XGB:
    HYPERPARAM_GRIDS["xgboost"] = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05],
    }

if _HAS_LGBM:
    HYPERPARAM_GRIDS["lightgbm"] = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, -1],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    }

if _HAS_CATBOOST:
    HYPERPARAM_GRIDS["catboost"] = {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1],
        "iterations": [200, 500],
    }


DENSE_ONLY_MODELS = {"mlp", "gaussian_nb"}
N_JOBS = 2  # Evita thrashing em nested CV


def _is_sparse_compatible(name: str, X) -> bool:
    if not issparse(X):
        return True
    return name not in DENSE_ONLY_MODELS


def _make_candidate(name: str, n_jobs: int | None = None, **overrides) -> object:
    cls, kwargs = INDIVIDUAL_CANDIDATES[name]
    params = dict(kwargs)
    if n_jobs is not None and "n_jobs" in params:
        params["n_jobs"] = n_jobs
    params.update(overrides)
    return cls(**params)


def _make_stacking() -> StackingClassifier:
    estimators = [
        ("lr", _make_candidate("logistic_regression")),
        ("rf", _make_candidate("random_forest")),
        ("svm", _make_candidate("svm")),
        ("et", _make_candidate("extra_trees")),
    ]
    if _HAS_XGB:
        estimators.append(("xgb", _make_candidate("xgboost")))
    if _HAS_LGBM:
        estimators.append(("lgbm", _make_candidate("lightgbm")))
    if _HAS_CATBOOST:
        estimators.append(("cb", _make_candidate("catboost")))
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=2,
        n_jobs=-1,
    )


def _make_voting() -> VotingClassifier:
    estimators = [
        ("lr", _make_candidate("logistic_regression")),
        ("rf", _make_candidate("random_forest")),
        ("et", _make_candidate("extra_trees")),
    ]
    if _HAS_XGB:
        estimators.append(("xgb", _make_candidate("xgboost")))
    if _HAS_LGBM:
        estimators.append(("lgbm", _make_candidate("lightgbm")))
    if _HAS_CATBOOST:
        estimators.append(("cb", _make_candidate("catboost")))
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


ENSEMBLE_CANDIDATES: dict[str, object] = {
    "stacking": _make_stacking(),
    "voting": _make_voting(),
}

ALL_CANDIDATES = list(INDIVIDUAL_CANDIDATES.keys()) + list(ENSEMBLE_CANDIDATES.keys())

NESTED_CV_CANDIDATES = [
    k for k in INDIVIDUAL_CANDIDATES
    if k not in DENSE_ONLY_MODELS
]


def _get_model(name: str) -> object:
    if name in ENSEMBLE_CANDIDATES:
        return ENSEMBLE_CANDIDATES[name]
    return _make_candidate(name)


def train_best(
    X_train,
    y_train: np.ndarray,
    cv: int = 5,
    save_path: Optional[Path] = None,
) -> tuple[str, object]:
    best_name, best_score, best_model = None, 0.0, None

    logger.info("Avaliando classificadores candidatos...")
    for name in ALL_CANDIDATES:
        if not _is_sparse_compatible(name, X_train):
            logger.debug("  %-25s incompatível com matriz sparse, pulando", name)
            continue
        model = _get_model(name)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
            mean_f1 = scores.mean()
            logger.info("  %-25s F1=%.4f (±%.4f)", name, mean_f1, scores.std())
        except Exception as e:
            logger.warning("  %-25s falhou: %s", name, e)
            continue
        if mean_f1 > best_score:
            best_score, best_name, best_model = mean_f1, name, model

    logger.info("Melhor modelo: %s (F1=%.4f)", best_name, best_score)
    logger.info("Treinando com todos os dados...")
    best_model.fit(X_train, y_train)

    if save_path is None:
        save_path = CLASSIFIER_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, save_path)
    logger.info("Classificador salvo em: %s", save_path)

    return best_name, best_model


def train_nested_cv_clf(
    X,
    y: np.ndarray,
    outer_cv: int = 5,
    inner_cv: int = 3,
    n_iter: int = 20,
    random_state: int = 42,
    save_path: Optional[Path] = None,
) -> tuple[str, dict, list[float], object]:
    outer_kfold = StratifiedKFold(
        n_splits=outer_cv, shuffle=True, random_state=random_state
    )
    outer_scores = []
    best_overall_score = -1.0
    best_overall_model = None
    best_overall_name = ""
    best_overall_params = {}

    logger.info("=" * 50)
    logger.info("Nested CV — Classificação (%d outer x %d inner)", outer_cv, inner_cv)
    logger.info("=" * 50)

    candidates = NESTED_CV_CANDIDATES if issparse(X) else list(INDIVIDUAL_CANDIDATES.keys())

    for fold, (train_idx, test_idx) in enumerate(outer_kfold.split(X, y)):
        X_train_fold = X[train_idx]
        X_test_fold = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        logger.info("--- Fold %d/%d ---", fold + 1, outer_cv)
        fold_best_score = -1.0
        fold_best_model = None
        fold_best_name = ""
        fold_best_params = {}

        for name in candidates:
            if not _is_sparse_compatible(name, X_train_fold):
                logger.debug("  %-25s incompatível com matriz sparse, pulando", name)
                continue
            base = _get_model(name)
            grid = HYPERPARAM_GRIDS.get(name, {})

            try:
                if grid:
                    grid_size = max(1, math.prod(len(v) for v in grid.values()))
                    actual_n_iter = min(n_iter, grid_size)
                    use_grid = grid_size <= actual_n_iter
                    Searcher = GridSearchCV if use_grid else RandomizedSearchCV
                    common = dict(cv=inner_cv, scoring="f1", n_jobs=N_JOBS, verbose=0)
                    if not use_grid:
                        common["random_state"] = random_state
                    if use_grid:
                        search = Searcher(base, param_grid=grid, **common)
                    else:
                        search = Searcher(
                            base, param_distributions=grid, n_iter=actual_n_iter, **common
                        )
                    search.fit(X_train_fold, y_train_fold)
                    inner_score = search.best_score_
                    candidate_model = search.best_estimator_
                    candidate_params = search.best_params_
                else:
                    scores = cross_val_score(
                        base, X_train_fold, y_train_fold, cv=inner_cv, scoring="f1",
                        n_jobs=N_JOBS,
                    )
                    inner_score = scores.mean()
                    candidate_model = base
                    candidate_params = {}
            except Exception as e:
                logger.warning("  %-25s falhou no fold: %s", name, e)
                continue

            if inner_score > fold_best_score:
                fold_best_score = inner_score
                fold_best_model = candidate_model
                fold_best_name = name
                fold_best_params = candidate_params

        fold_best_model.fit(X_train_fold, y_train_fold)
        y_pred = fold_best_model.predict(X_test_fold)
        outer_score = f1_score(y_test_fold, y_pred)
        outer_scores.append(outer_score)

        logger.info(
            "  >> Vencedor: %s | outer F1=%.4f | params=%s",
            fold_best_name,
            outer_score,
            fold_best_params,
        )

        if outer_score > best_overall_score:
            best_overall_score = outer_score
            best_overall_model = fold_best_model
            best_overall_name = fold_best_name
            best_overall_params = fold_best_params

    outer_mean = float(np.mean(outer_scores))
    outer_std = float(np.std(outer_scores))

    logger.info("=" * 50)
    logger.info("Nested CV — F1 médio: %.4f (±%.4f)", outer_mean, outer_std)
    logger.info(
        "Melhor fold: %s com F1=%.4f (params=%s)",
        best_overall_name,
        best_overall_score,
        best_overall_params,
    )

    logger.info("Retreinando com todos os dados...")
    final_model = _get_model(best_overall_name)
    if best_overall_params:
        try:
            final_model.set_params(**best_overall_params)
        except (ValueError, AttributeError):
            logger.warning("Não foi possível aplicar params a %s", best_overall_name)
    final_model.fit(X, y)

    if save_path is None:
        save_path = CLASSIFIER_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, save_path)
    logger.info("Classificador salvo em: %s", save_path)

    return best_overall_name, best_overall_params, outer_scores, final_model


def predict(
    resume_vec,
    classifier: object,
) -> tuple[str, float]:
    label = int(classifier.predict(resume_vec)[0])

    if hasattr(classifier, "predict_proba"):
        prob = float(classifier.predict_proba(resume_vec).max())
    elif hasattr(classifier, "decision_function"):
        raw_score = classifier.decision_function(resume_vec)[0]
        prob = float(expit(raw_score))
    else:
        prob = 1.0 if label == 1 else 0.0

    return ("Fit" if label == 1 else "No Fit"), round(prob, 4)
