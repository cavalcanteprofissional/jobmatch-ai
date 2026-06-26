import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.sparse import issparse
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

SALARY_MODEL_PATH = Path("data/models/salary_regressor.pkl")

DENSE_ONLY_MODELS = {"mlp"}


def _is_sparse_compatible(name: str, X) -> bool:
    if not issparse(X):
        return True
    return name not in DENSE_ONLY_MODELS


INDIVIDUAL_CANDIDATES: dict[str, tuple[type, dict]] = {
    "gradient_boosting": (
        GradientBoostingRegressor,
        {"random_state": 42},
    ),
    "random_forest": (
        RandomForestRegressor,
        {"n_jobs": -1, "random_state": 42},
    ),
    "xgboost": (
        XGBRegressor,
        {"random_state": 42, "verbosity": 0, "tree_method": "hist", "n_jobs": -1},
    ),
    "extra_trees": (
        ExtraTreesRegressor,
        {"n_jobs": -1, "random_state": 42},
    ),
    "knn": (
        KNeighborsRegressor,
        {"n_neighbors": 3, "n_jobs": -1},
    ),
    "sgd": (
        SGDRegressor,
        {"max_iter": 2000, "random_state": 42, "tol": 1e-3},
    ),
    "decision_tree": (
        DecisionTreeRegressor,
        {"random_state": 42},
    ),
    "mlp": (
        MLPRegressor,
        {"max_iter": 300, "random_state": 42, "early_stopping": True,
         "n_iter_no_change": 10},
    ),
}

if _HAS_LGBM:
    INDIVIDUAL_CANDIDATES["lightgbm"] = (
        LGBMRegressor,
        {"random_state": 42, "verbosity": -1, "n_jobs": -1},
    )

if _HAS_CATBOOST:
    INDIVIDUAL_CANDIDATES["catboost"] = (
        CatBoostRegressor,
        {"random_seed": 42, "verbose": 0, "allow_writing_files": False},
    )

HYPERPARAM_GRIDS: dict[str, dict] = {
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05],
    },
    "extra_trees": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
    "knn": {
        "n_neighbors": [3, 5, 11],
        "weights": ["uniform", "distance"],
    },
    "sgd": {
        "alpha": [1e-5, 1e-4, 1e-3],
        "penalty": ["l2", "elasticnet"],
    },
    "decision_tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    },
    "mlp": {
        "hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.001, 0.01],
    },
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


def _make_candidate(name: str) -> object:
    cls, kwargs = INDIVIDUAL_CANDIDATES[name]
    return cls(**kwargs)


def _make_stacking() -> StackingRegressor:
    estimators = [
        ("gb", _make_candidate("gradient_boosting")),
        ("rf", _make_candidate("random_forest")),
        ("xgb", _make_candidate("xgboost")),
        ("et", _make_candidate("extra_trees")),
    ]
    if _HAS_LGBM:
        estimators.append(("lgbm", _make_candidate("lightgbm")))
    if _HAS_CATBOOST:
        estimators.append(("cb", _make_candidate("catboost")))
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(random_state=42),
        cv=2,
        n_jobs=-1,
    )


def _make_voting() -> VotingRegressor:
    estimators = [
        ("gb", _make_candidate("gradient_boosting")),
        ("rf", _make_candidate("random_forest")),
        ("xgb", _make_candidate("xgboost")),
        ("et", _make_candidate("extra_trees")),
    ]
    if _HAS_LGBM:
        estimators.append(("lgbm", _make_candidate("lightgbm")))
    if _HAS_CATBOOST:
        estimators.append(("cb", _make_candidate("catboost")))
    return VotingRegressor(estimators=estimators, n_jobs=-1)


ENSEMBLE_CANDIDATES: dict[str, object] = {
    "stacking": _make_stacking(),
    "voting": _make_voting(),
}

ALL_CANDIDATES = list(INDIVIDUAL_CANDIDATES.keys()) + list(ENSEMBLE_CANDIDATES.keys())


def _get_model(name: str) -> object:
    if name in ENSEMBLE_CANDIDATES:
        return ENSEMBLE_CANDIDATES[name]
    return _make_candidate(name)


def train_salary_model(
    X_train,
    y_train: np.ndarray,
    cv: int = 5,
    save_path: Optional[Path] = None,
) -> object:
    best_name, best_score, best_model = None, float("inf"), None

    logger.info("Avaliando regressores candidatos...")
    for name in ALL_CANDIDATES:
        if not _is_sparse_compatible(name, X_train):
            logger.debug("  %-25s incompatível com matriz sparse, pulando", name)
            continue
        model = _get_model(name)
        try:
            rmse_scores = -cross_val_score(
                model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
            )
            mean_rmse = rmse_scores.mean()
            logger.info("  %-25s RMSE=$%.0f (±$%.0f)", name, mean_rmse, rmse_scores.std())
        except Exception as e:
            logger.warning("  %-25s falhou: %s", name, e)
            continue
        if mean_rmse < best_score:
            best_score, best_name, best_model = mean_rmse, name, model

    logger.info("Melhor regressor: %s (RMSE=$%.0f)", best_name, best_score)
    logger.info("Treinando com todos os dados...")
    best_model.fit(X_train, y_train)

    if save_path is None:
        save_path = SALARY_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, save_path)
    logger.info("Regressor salvo em: %s", save_path)

    return best_model


def train_nested_cv_reg(
    X,
    y: np.ndarray,
    outer_cv: int = 3,
    inner_cv: int = 3,
    n_iter: int = 10,
    random_state: int = 42,
    save_path: Optional[Path] = None,
) -> tuple[str, dict, list[float], object]:
    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    outer_scores = []
    best_overall_score = float("inf")
    best_overall_model = None
    best_overall_name = ""
    best_overall_params = {}

    logger.info("=" * 50)
    logger.info("Nested CV — Regressão (%d outer x %d inner)", outer_cv, inner_cv)
    logger.info("=" * 50)

    for fold, (train_idx, test_idx) in enumerate(outer_kfold.split(X, y)):
        X_train_fold = X[train_idx]
        X_test_fold = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        logger.info("--- Fold %d/%d ---", fold + 1, outer_cv)
        fold_best_score = float("inf")
        fold_best_model = None
        fold_best_name = ""
        fold_best_params = {}

        for name in ALL_CANDIDATES:
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
                    common = dict(cv=inner_cv, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=0)
                    if not use_grid:
                        common["random_state"] = random_state
                    if use_grid:
                        search = Searcher(base, param_grid=grid, **common)
                    else:
                        search = Searcher(
                            base, param_distributions=grid, n_iter=actual_n_iter, **common
                        )
                    search.fit(X_train_fold, y_train_fold)
                    inner_score = -search.best_score_
                    candidate_model = search.best_estimator_
                    candidate_params = search.best_params_
                else:
                    rmse_scores = -cross_val_score(
                        base,
                        X_train_fold,
                        y_train_fold,
                        cv=inner_cv,
                        scoring="neg_root_mean_squared_error",
                    )
                    inner_score = rmse_scores.mean()
                    candidate_model = base
                    candidate_params = {}
            except Exception as e:
                logger.warning("  %-25s falhou no fold: %s", name, e)
                continue

            if inner_score < fold_best_score:
                fold_best_score = inner_score
                fold_best_model = candidate_model
                fold_best_name = name
                fold_best_params = candidate_params

        fold_best_model.fit(X_train_fold, y_train_fold)
        y_pred = fold_best_model.predict(X_test_fold)
        outer_score = float(np.sqrt(mean_squared_error(y_test_fold, y_pred)))
        outer_scores.append(outer_score)

        logger.info(
            "  >> Vencedor: %s | outer RMSE=$%.0f | params=%s",
            fold_best_name,
            outer_score,
            fold_best_params,
        )

        if outer_score < best_overall_score:
            best_overall_score = outer_score
            best_overall_model = fold_best_model
            best_overall_name = fold_best_name
            best_overall_params = fold_best_params

    outer_mean = float(np.mean(outer_scores))
    outer_std = float(np.std(outer_scores))

    logger.info("=" * 50)
    logger.info("Nested CV — RMSE médio: $%.0f (±$%.0f)", outer_mean, outer_std)
    logger.info(
        "Melhor fold: %s com RMSE=$%.0f (params=%s)",
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
        save_path = SALARY_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, save_path)
    logger.info("Regressor salvo em: %s", save_path)

    return best_overall_name, best_overall_params, outer_scores, final_model


def predict_salary_range(job_vec, model: object) -> dict:
    pred = float(model.predict(job_vec)[0])
    return {
        "estimated_annual_usd": round(pred),
        "range_low": round(pred * 0.85),
        "range_high": round(pred * 1.15),
    }
