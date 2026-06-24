from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

SALARY_MODEL_PATH = Path("data/models/salary_regressor.pkl")

CANDIDATES: dict[str, object] = {
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=42
    ),
}


def train_salary_model(
    X_train,
    y_train: np.ndarray,
    cv: int = 5,
    save_path: Optional[Path] = None,
) -> object:
    best_name, best_score, best_model = None, float("inf"), None

    logger.info("Avaliando regressores candidatos...")
    for name, model in CANDIDATES.items():
        rmse_scores = -cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
        )
        mean_rmse = rmse_scores.mean()
        logger.info("  %-25s RMSE=$%.0f (±$%.0f)", name, mean_rmse, rmse_scores.std())
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


def predict_salary_range(job_vec, model: object) -> dict:
    pred = float(model.predict(job_vec)[0])
    return {
        "estimated_annual_usd": round(pred),
        "range_low": round(pred * 0.85),
        "range_high": round(pred * 1.15),
    }
