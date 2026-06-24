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
"""

import pandas as pd
from scipy.sparse import save_npz
from sklearn.model_selection import train_test_split

from src.models.classifier import train_best
from src.models.salary_model import train_salary_model
from src.models.vectorizer import fit_vectorizer, transform
from src.pipeline.compose_datasets import compose, quality_check
from src.pipeline.load_data import load_all
from src.pipeline.preprocess import clean_text
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    logger.info("=" * 50)
    logger.info("JobMatch AI — Pipeline de Treino")
    logger.info("=" * 50)

    logger.info("1/5 — Carregando datasets...")
    raw = load_all()

    logger.info("2/5 — Compondo e limpando datasets...")
    jobs_df, pairs_df = compose(raw)
    quality_check(jobs_df, pairs_df)

    logger.info("3/5 — Vetorizando com TF-IDF...")
    corpus = jobs_df["full_text"].dropna().tolist()
    vec = fit_vectorizer(corpus)

    # Pré-vetorizar matrix de jobs e salvar para uso rápido no Predictor
    logger.info("Pré-vetorizando matrix de jobs...")
    jobs_matrix = transform(corpus, vec)
    save_npz("data/models/jobs_matrix.npz", jobs_matrix)
    logger.info("Jobs matrix salva: %s", jobs_matrix.shape)

    pairs_df["resume_clean"] = pairs_df["resume"].fillna("").apply(clean_text)
    pairs_df["job_clean"] = pairs_df["job_description"].fillna("").apply(clean_text)
    combined = (pairs_df["resume_clean"] + " " + pairs_df["job_clean"]).tolist()
    X = transform(combined, vec)
    y = (pairs_df["label"] == "Fit").astype(int).values

    logger.info("4/5 — Treinando classificador Fit/No Fit...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    _, clf = train_best(X_train, y_train)

    test_score = clf.score(X_test, y_test)
    logger.info("Acurácia no teste: %.4f", test_score)

    logger.info("5/5 — Treinando regressor de salário...")
    jobs_with_salary = jobs_df.dropna(subset=["salary_annual_avg"])
    logger.info("Vagas com salário: %s", len(jobs_with_salary))

    X_sal = transform(jobs_with_salary["full_text"].tolist(), vec)
    y_sal = jobs_with_salary["salary_annual_avg"].values
    train_salary_model(X_sal, y_sal)

    logger.info("=" * 50)
    logger.info("Pipeline concluída! Modelos salvos em data/models/")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
