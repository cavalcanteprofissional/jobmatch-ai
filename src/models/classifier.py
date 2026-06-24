from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CLASSIFIER_PATH = Path("data/models/classifier.pkl")

CANDIDATES: dict[str, object] = {
    "logistic_regression": LogisticRegression(C=1.0, max_iter=500, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
    ),
    "svm": LinearSVC(C=0.5, class_weight="balanced", max_iter=2000, random_state=42),
}


def train_best(
    X_train,
    y_train: np.ndarray,
    cv: int = 5,
    save_path: Optional[Path] = None,
) -> tuple[str, object]:
    best_name, best_score, best_model = None, 0.0, None

    logger.info("Avaliando classificadores candidatos...")
    for name, model in CANDIDATES.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        mean_f1 = scores.mean()
        logger.info("  %-25s F1=%.4f (±%.4f)", name, mean_f1, scores.std())
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
