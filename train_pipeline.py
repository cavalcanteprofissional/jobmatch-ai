"""
Executa toda a pipeline de treino do JobMatch AI.

Uso:
    python train_pipeline.py

Fluxo:
    1. Carregar datasets (load_data)
    2. Compor e limpar (compose_datasets)
    3. Vetorizar com TF-IDF e salvar jobs_matrix.npz
    4. Treinar classificador Fit/No Fit (classifier)
    5. Treinar regressor de salário (salary_model)
    6. Salvar métricas dos modelos (metrics.json)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.models.classifier import train_best
from src.models.salary_model import train_salary_model
from src.models.vectorizer import fit_vectorizer, transform
from src.pipeline.compose_datasets import compose, quality_check
from src.pipeline.load_data import load_all
from src.pipeline.preprocess import clean_text
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def save_metrics(y_true, y_pred, y_sal_true, y_sal_pred, model_info: dict) -> None:
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics = {
        "classification": {
            "model_type": "",
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "test_samples": int(len(y_true)),
            "confusion_matrix": cm,
        },
        "regression": {
            "model_type": "",
            "rmse": round(float(np.sqrt(mean_squared_error(y_sal_true, y_sal_pred))), 2),
            "mae": round(float(mean_absolute_error(y_sal_true, y_sal_pred)), 2),
            "r2": round(float(r2_score(y_sal_true, y_sal_pred)), 4),
            "test_samples": int(len(y_sal_true)),
        },
        "model_info": model_info,
    }
    path = Path("data/models/metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métricas salvas em: %s", path)


def main() -> None:
    logger.info("=" * 50)
    logger.info("JobMatch AI — Pipeline de Treino")
    logger.info("=" * 50)

    logger.info("1/6 — Carregando datasets...")
    raw = load_all()

    logger.info("2/6 — Compondo e limpando datasets...")
    jobs_df, pairs_df = compose(raw)
    quality_check(jobs_df, pairs_df)

    logger.info("3/6 — Vetorizando com TF-IDF...")
    corpus = jobs_df["full_text"].dropna().tolist()
    vec = fit_vectorizer(corpus)

    logger.info("Pré-vetorizando matrix de jobs...")
    jobs_matrix = transform(corpus, vec)
    save_npz("data/models/jobs_matrix.npz", jobs_matrix)
    logger.info("Jobs matrix salva: %s", jobs_matrix.shape)

    pairs_df["resume_clean"] = pairs_df["resume"].fillna("").apply(clean_text)
    pairs_df["job_clean"] = pairs_df["job_description"].fillna("").apply(clean_text)
    combined = (pairs_df["resume_clean"] + " " + pairs_df["job_clean"]).tolist()
    X = transform(combined, vec)

    label_map = {"Good Fit": 1, "Potential Fit": 1, "No Fit": 0}
    y = pairs_df["label"].map(label_map).values

    if y.sum() == 0:
        logger.warning("Nenhum Fit encontrado! Usando label == 'Good Fit' como fallback.")
        y = (pairs_df["label"] == "Good Fit").astype(int).values

    logger.info("4/6 — Treinando classificador Fit/No Fit...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    name, clf = train_best(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_score = accuracy_score(y_test, y_pred)
    logger.info("Acurácia no teste: %.4f (F1: %.4f)", test_score, f1_score(y_test, y_pred))

    logger.info("5/6 — Treinando regressor de salário...")
    jobs_with_salary = jobs_df.dropna(subset=["salary_annual_avg"])
    logger.info("Vagas com salário: %s", len(jobs_with_salary))

    X_sal = transform(jobs_with_salary["full_text"].tolist(), vec)
    y_sal = jobs_with_salary["salary_annual_avg"].values
    sal = train_salary_model(X_sal, y_sal)

    X_sal_train, X_sal_test, y_sal_train, y_sal_test = train_test_split(
        X_sal, y_sal, test_size=0.2, random_state=42,
    )
    sal.fit(X_sal_train, y_sal_train)
    y_sal_pred = sal.predict(X_sal_test)

    logger.info("6/6 — Salvando métricas dos modelos...")
    save_metrics(
        y_test, y_pred,
        y_sal_test, y_sal_pred,
        {
            "vectorizer_features": len(vec.vocabulary_),
            "total_jobs": len(jobs_df),
            "jobs_with_salary": len(jobs_with_salary),
            "training_pairs": len(pairs_df),
        },
    )

    logger.info("=" * 50)
    logger.info("Pipeline concluída! Modelos salvos em data/models/")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
