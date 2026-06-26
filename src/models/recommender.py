import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_HAS_CROSSENCODER = False
_cross_encoder_model = None


def _load_cross_encoder():
    global _cross_encoder_model, _HAS_CROSSENCODER
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder_model = CrossEncoder("cross-encoder/stsb-MiniLM-L-6-v2")
            _HAS_CROSSENCODER = True
            logger.info("Cross-encoder carregado: stsb-MiniLM-L-6-v2")
        except Exception as e:
            logger.warning("Cross-encoder não disponível: %s", e)
            _HAS_CROSSENCODER = False
    return _HAS_CROSSENCODER


def rerank_with_cross_encoder(
    resume_text: str,
    jobs_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    if not _load_cross_encoder():
        return jobs_df.head(top_k)

    pairs = [[resume_text, row.get("full_text", row.get("title", ""))]
             for _, row in jobs_df.iterrows()]
    raw_scores = _cross_encoder_model.predict(pairs, show_progress_bar=False)
    scores = np.clip(raw_scores, 0.0, 1.0)
    jobs_df = jobs_df.copy()
    jobs_df["adherence_score"] = np.round(scores * 100, 1)
    jobs_df["fit_label"] = jobs_df["adherence_score"].apply(
        lambda s: "✅ Fit" if s >= 40 else "❌ No Fit"
    )
    jobs_df = jobs_df.sort_values("adherence_score", ascending=False).head(top_k)
    logger.debug("Cross-encoder re-ranking: top-%s de %s vagas", top_k, len(jobs_df))
    return jobs_df


def rank_jobs(
    resume_vec,
    jobs_matrix,
    jobs_df: pd.DataFrame,
    top_k: int = 5,
    fit_threshold: float = 40.0,
    resume_text: str = "",
    use_cross_encoder: bool = False,
) -> pd.DataFrame:
    scores = cosine_similarity(resume_vec, jobs_matrix).flatten()
    candidate_k = min(top_k * 4, len(jobs_df))
    top_indices = np.argsort(scores)[::-1][:candidate_k]

    result = jobs_df.iloc[top_indices].copy()
    result["adherence_score"] = np.round(scores[top_indices] * 100, 1)
    result["fit_label"] = result["adherence_score"].apply(
        lambda s: "✅ Fit" if s >= fit_threshold else "❌ No Fit"
    )

    if use_cross_encoder and resume_text:
        logger.info("Aplicando cross-encoder re-ranking nas top-%s vagas...", candidate_k)
        result = rerank_with_cross_encoder(resume_text, result, top_k=top_k)
    else:
        result = result.head(top_k)

    cols = [
        "title",
        "company_name",
        "location",
        "adherence_score",
        "fit_label",
        "min_salary_annual",
        "max_salary_annual",
        "skills_desc",
    ]
    available_cols = [c for c in cols if c in result.columns]
    logger.debug("Ranking: top-%s de %s vagas", top_k, len(jobs_df))
    return result[available_cols]
