"""
Predictor — classe central que encapsula todo o ML do JobMatch AI.

Pré-carrega modelos uma única vez e expõe predict() que retorna
dict pronto para serialização JSON.

Uso:
    predictor = JobMatchPredictor()
    result = predictor.predict("Meu currículo...", top_k=5, threshold=40.0)
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.models.classifier import predict as classify
from src.models.recommender import rank_jobs
from src.models.salary_model import predict_salary_range
from src.models.vectorizer import load_vectorizer, transform
from src.pipeline.preprocess import clean_text
from src.skills.skills_analyzer import analyze_gap
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class JobMatchPredictor:
    """
    Predictor singleton que pré-carrega todos os modelos e dados.

    Atributos:
        vec (TfidfVectorizer): Vetorizador TF-IDF treinado.
        clf (object): Classificador Fit/No Fit treinado.
        sal (object): Regressor de salário treinado.
        jobs (pd.DataFrame): DataFrame de vagas limpas.
        jobs_matrix (sparse.csr_matrix): Matrix TF-IDF pré-computada das vagas.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        if models_dir is None:
            models_dir = Path("data/models")
        if processed_dir is None:
            processed_dir = Path("data/processed")

        logger.info("Inicializando JobMatchPredictor...")

        self.vec = load_vectorizer(models_dir / "tfidf_vectorizer.pkl")
        logger.info("Vetorizador carregado")

        self.clf = joblib.load(models_dir / "classifier.pkl")
        logger.info("Classificador carregado")

        self.sal = joblib.load(models_dir / "salary_regressor.pkl")
        logger.info("Regressor carregado")

        self.jobs = pd.read_parquet(processed_dir / "jobs_clean.parquet")
        logger.info("Jobs carregados: %s", len(self.jobs))

        matrix_path = models_dir / "jobs_matrix.npz"
        if matrix_path.exists():
            self.jobs_matrix = load_npz(str(matrix_path))
            logger.info("Jobs matrix carregada: %s", self.jobs_matrix.shape)
        else:
            logger.warning(
                "jobs_matrix.npz não encontrado. Computando on-the-fly..."
            )
            self.jobs_matrix = transform(
                self.jobs["full_text"].tolist(), self.vec,
            )

        logger.info("JobMatchPredictor pronto!")

    def predict(
        self,
        resume_text: str,
        top_k: int = 5,
        fit_threshold: float = 40.0,
    ) -> dict:
        """
        Executa pipeline completa de predição e retorna dict JSON-serializável.

        Args:
            resume_text: Texto bruto do currículo/perfil.
            top_k: Número de vagas no ranking.
            fit_threshold: Score mínimo (%) para Fit.

        Returns:
            Dict com chaves:
                - score_pct: probabilidade de Fit (%)
                - fit_label: 'Fit' | 'No Fit'
                - avg_adherence: score médio do top-k
                - fit_count: quantas vagas são Fit no top-k
                - top_k: número solicitado
                - salary_est: dict {estimated_annual_usd, range_low, range_high}
                - gap: dict {compatible, missing, development_plan}
                - top_jobs: list[dict] com dados das vagas
        """
        logger.info("Predict chamado (top_k=%s, threshold=%.0f)", top_k, fit_threshold)

        resume_clean = clean_text(resume_text)
        resume_vec = transform([resume_clean], self.vec)

        fit_label, fit_prob = classify(resume_vec, self.clf)
        score_pct = round(fit_prob * 100, 1)

        top_jobs_df = rank_jobs(
            resume_vec, self.jobs_matrix, self.jobs,
            top_k=top_k, fit_threshold=fit_threshold,
        )

        best_job_idx = top_jobs_df.index[0]
        best_job_vec = transform(
            [self.jobs.loc[best_job_idx, "full_text"]], self.vec,
        )
        salary_est = predict_salary_range(best_job_vec, self.sal)

        gap = analyze_gap(resume_text, top_jobs_df.iloc[0]["title"])

        result = {
            "score_pct": score_pct,
            "fit_label": fit_label,
            "avg_adherence": round(top_jobs_df["adherence_score"].mean(), 1),
            "fit_count": int((top_jobs_df["adherence_score"] >= fit_threshold).sum()),
            "top_k": top_k,
            "salary_est": salary_est,
            "gap": gap,
            "top_jobs": top_jobs_df.reset_index().to_dict(orient="records"),
        }

        logger.info(
            "Predict concluído: score=%.1f%%, fit=%s, top_jobs=%s",
            score_pct, fit_label, len(result["top_jobs"]),
        )
        return result


# Singleton para reuso entre requisições FastAPI
_predictor_instance: Optional[JobMatchPredictor] = None


def get_predictor() -> JobMatchPredictor:
    """Retorna instância singleton do JobMatchPredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = JobMatchPredictor()
    return _predictor_instance
