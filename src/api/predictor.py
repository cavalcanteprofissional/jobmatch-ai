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

import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.models.classifier import predict as classify
from src.models.recommender import rank_jobs
from src.models.salary_model import predict_salary_range
from src.models.vectorizer import load_vectorizer, transform, SentenceBertVectorizer
from src.pipeline.preprocess import clean_text
from src.skills.skills_analyzer import analyze_gap
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class JobMatchPredictor:
    """
    Predictor singleton que pré-carrega todos os modelos e dados.

    Atributos:
        vec (TfidfVectorizer): Vetorizador TF-IDF treinado.
        clf (object): Classificador Fit/No Fit treinado (TF-IDF).
        clf_sbert (object): Classificador treinado em embeddings SBERT.
        sbert (SentenceBertVectorizer): Vetorizador Sentence-BERT.
        sal (object): Regressor de salário treinado.
        jobs (pd.DataFrame): DataFrame de vagas limpas.
        jobs_matrix (sparse.csr_matrix): Matrix TF-IDF pré-computada das vagas.
        jobs_sbert (np.ndarray): Embeddings SBERT pré-computados das vagas.
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

        self.clf_sbert = None
        self.sbert = None
        self.jobs_sbert = None
        sbert_clf_path = models_dir / "classifier_sbert.pkl"
        sbert_path = models_dir / "sentence_bert.pkl"
        sbert_emb_path = models_dir / "jobs_sbert_embeddings.npy"
        if sbert_clf_path.exists() and sbert_path.exists() and sbert_emb_path.exists():
            try:
                self.clf_sbert = joblib.load(sbert_clf_path)
                self.sbert = joblib.load(sbert_path)
                self.jobs_sbert = np.load(sbert_emb_path)
                logger.info("SBERT carregado: %s", self.jobs_sbert.shape)
            except Exception as e:
                logger.warning("SBERT não pôde ser carregado: %s", e)

        logger.info("JobMatchPredictor pronto!")

    @staticmethod
    def _clean_nan(obj):
        if isinstance(obj, dict):
            return {k: JobMatchPredictor._clean_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [JobMatchPredictor._clean_nan(v) for v in obj]
        if isinstance(obj, np.generic):
            obj = obj.item()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj

    def predict(
        self,
        resume_text: str,
        top_k: int = 5,
        fit_threshold: float = 40.0,
        use_sbert: bool = False,
        use_cross_encoder: bool = False,
    ) -> dict:
        """
        Executa pipeline completa de predição e retorna dict JSON-serializável.

        Args:
            resume_text: Texto bruto do currículo/perfil.
            top_k: Número de vagas no ranking.
            fit_threshold: Score mínimo (%) para Fit.
            use_sbert: Se True, usa Sentence-BERT em vez de TF-IDF.

        Returns:
            Dict com chaves:
                - score_pct: probabilidade de Fit (%)
                - fit_label: 'Fit' | 'No Fit'
                - avg_adherence: score médio do top-k
                - fit_count: quantas vagas são Fit no top-k
                - top_k: número solicitado
                - employability_score: % de empregabilidade
                - salary_est: dict {estimated_annual_usd, range_low, range_high}
                - gap: dict {compatible, missing, development_plan}
                - top_jobs: list[dict] com dados das vagas
        """
        logger.info("Predict chamado (top_k=%s, threshold=%.0f, sbert=%s)", top_k, fit_threshold, use_sbert)

        resume_clean = clean_text(resume_text)

        if use_sbert and self.clf_sbert is not None and self.sbert is not None and self.jobs_sbert is not None:
            resume_vec = self.sbert.transform([resume_clean])
            jobs_matrix = self.jobs_sbert
            clf = self.clf_sbert
            logger.info("Usando SBERT para inferência")
        else:
            resume_vec = transform([resume_clean], self.vec)
            jobs_matrix = self.jobs_matrix
            clf = self.clf
            if use_sbert:
                logger.warning("SBERT solicitado mas não disponível. Usando TF-IDF.")

        fit_label, fit_prob = classify(resume_vec, clf)
        score_pct = round(float(fit_prob) * 100, 1)

        top_jobs_df = rank_jobs(
            resume_vec, jobs_matrix, self.jobs,
            top_k=top_k, fit_threshold=fit_threshold,
            resume_text=resume_text if use_cross_encoder else "",
            use_cross_encoder=use_cross_encoder,
        )

        best_job_idx = top_jobs_df.index[0]
        best_job_vec = transform(
            [self.jobs.loc[best_job_idx, "full_text"]], self.vec,
        )
        salary_est = predict_salary_range(best_job_vec, self.sal)

        gap = analyze_gap(resume_text, top_jobs_df.iloc[0]["title"])

        job_scores = []
        all_unique_required = set()
        all_unique_compat = set()
        for _, row in top_jobs_df.iterrows():
            g = analyze_gap(resume_text, row.get("title", ""))
            compat = set(g["compatible"])
            missing = set(g["missing"])
            total = len(compat) + len(missing)
            all_unique_required.update(compat, missing)
            all_unique_compat.update(compat)
            if total > 0:
                job_scores.append(len(compat) / total * 100)
        if job_scores:
            employability_score = round(sum(job_scores) / len(job_scores), 1)
        elif all_unique_required:
            employability_score = round(
                len(all_unique_compat) / len(all_unique_required) * 100, 1
            )
        else:
            employability_score = 0.0

        result = {
            "score_pct": score_pct,
            "fit_label": fit_label,
            "avg_adherence": round(float(top_jobs_df["adherence_score"].mean()), 1),
            "fit_count": int((top_jobs_df["adherence_score"] >= fit_threshold).sum()),
            "top_k": top_k,
            "employability_score": employability_score,
            "salary_est": salary_est,
            "gap": gap,
            "top_jobs": json.loads(
                top_jobs_df.reset_index().to_json(orient="records", date_format="iso")
            ),
        }

        result = self._clean_nan(result)

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
