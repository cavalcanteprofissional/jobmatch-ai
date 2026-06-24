import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def rank_jobs(
    resume_vec,
    jobs_matrix,
    jobs_df: pd.DataFrame,
    top_k: int = 5,
    fit_threshold: float = 40.0,
) -> pd.DataFrame:
    scores = cosine_similarity(resume_vec, jobs_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    result = jobs_df.iloc[top_indices].copy()
    result["adherence_score"] = np.round(scores[top_indices] * 100, 1)
    result["fit_label"] = result["adherence_score"].apply(
        lambda s: "✅ Fit" if s >= fit_threshold else "❌ No Fit"
    )

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
