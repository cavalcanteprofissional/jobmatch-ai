import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from tqdm import tqdm

from src.pipeline.preprocess import clean_text
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


PERIOD_TO_ANNUAL = {
    "HOURLY": 40 * 52,
    "DAILY": 5 * 52,
    "WEEKLY": 52,
    "MONTHLY": 12,
    "YEARLY": 1,
}


def normalize_salary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "min_salary" not in df.columns or "max_salary" not in df.columns:
        logger.warning("Colunas de salário não encontradas. Pulando normalização.")
        df["min_salary_annual"] = np.nan
        df["max_salary_annual"] = np.nan
        df["salary_annual_avg"] = np.nan
        return df

    if "pay_period" in df.columns:
        df["pay_period"] = df["pay_period"].fillna("YEARLY").astype(str).str.upper()
        multiplier = df["pay_period"].map(PERIOD_TO_ANNUAL).fillna(1)
    else:
        multiplier = 1

    df["min_salary"] = pd.to_numeric(df["min_salary"], errors="coerce")
    df["max_salary"] = pd.to_numeric(df["max_salary"], errors="coerce")

    df["min_salary_annual"] = df["min_salary"] * multiplier
    df["max_salary_annual"] = df["max_salary"] * multiplier
    df["salary_annual_avg"] = (
        df["min_salary_annual"] + df["max_salary_annual"]
    ) / 2

    valid = df["salary_annual_avg"].dropna()
    valid = valid[valid > 0]

    if len(valid) > 10:
        q_low = valid.quantile(0.01)
        q_high = valid.quantile(0.99)
        mask = (
            df["salary_annual_avg"].isna()
            | ((df["salary_annual_avg"] >= q_low) & (df["salary_annual_avg"] <= q_high))
        )
        removed = (~mask).sum()
        df = df[mask]
        if removed:
            logger.info("Removidos %s outliers de salário", removed)
    else:
        logger.warning("Poucos salários válidos para remoção de outliers")

    logger.info("Salário anual médio: $%.0f", df['salary_annual_avg'].median())
    return df


def build_full_text(row: pd.Series) -> str:
    parts = [
        str(row.get("title", "")),
        str(row.get("description", "")),
        str(row.get("skills_desc", "")),
    ]
    combined = " ".join(p for p in parts if p and p != "nan")
    return clean_text(combined)


def add_full_text(jobs_df: pd.DataFrame) -> pd.DataFrame:
    jobs_df = jobs_df.copy()
    tqdm.pandas(desc="Criando full_text")
    jobs_df["full_text"] = jobs_df.progress_apply(build_full_text, axis=1)

    n_before = len(jobs_df)
    jobs_df = jobs_df[jobs_df["full_text"].str.split().str.len() >= 30]
    n_after = len(jobs_df)
    logger.info("Removidas %s vagas com texto curto (< 30 tokens)", n_before - n_after)
    logger.info("%s vagas com full_text válido", n_after)

    return jobs_df


def build_skills_map(
    jobs_df: pd.DataFrame,
    skills_df: pd.DataFrame,
    score_threshold: int = 80,
    save_path: Optional[Path] = None,
    fuzzy_limit: int = 2000,
) -> dict:
    # Garantir que skills_df tenha job_title e skills
    raw_skills_df = skills_df.copy()

    # Se não tiver job_title, tentar merge com linkedin_job_postings
    if "job_title" not in raw_skills_df.columns:
        linkedin_path = RAW_DIR / "linkedin_job_postings.csv"
        if linkedin_path.exists():
            logger.info("Mergeando skills com linkedin_job_postings.csv...")
            titles = pd.read_csv(
                linkedin_path, usecols=["job_link", "job_title"], low_memory=False
            )
            raw_skills_df = raw_skills_df.merge(titles, on="job_link", how="left")

    # Renomear coluna de skills
    if "skills" not in raw_skills_df.columns and "job_skills" in raw_skills_df.columns:
        raw_skills_df = raw_skills_df.rename(columns={"job_skills": "skills"})

    skill_titles_set = set(raw_skills_df["job_title"].dropna().str.lower().unique())
    unique_titles = jobs_df["title"].dropna().unique().tolist()

    if not skill_titles_set:
        logger.warning("Nenhum título encontrado no Skills Dataset")
        return {}

    skills_map = {}

    # Passo 1: matching exato (rápido)
    logger.info("Match exato de %s títulos...", len(unique_titles))
    matched_titles = []
    remaining = []
    title_lower_map = {t.lower(): t for t in unique_titles if isinstance(t, str)}

    for title in unique_titles:
        if title.lower() in skill_titles_set:
            matched_titles.append(title)
        else:
            remaining.append(title)

    logger.info("  Exatos: %s | Restantes: %s", len(matched_titles), len(remaining))

    # Passo 2: fuzzy match (sem limite artificial agressivo)
    actual_fuzzy_limit = min(fuzzy_limit, len(remaining))
    if actual_fuzzy_limit > 0:
        fuzzy_sample = remaining[:actual_fuzzy_limit]
        skill_titles_list = sorted(skill_titles_set)

        logger.info(
            "Fuzzy match de %s títulos (amostra de %s restantes)...",
            actual_fuzzy_limit, len(remaining),
        )
        for title in tqdm(fuzzy_sample, desc="Fuzzy títulos"):
            match, score, _ = process.extractOne(
                title, skill_titles_list, scorer=fuzz.token_sort_ratio
            )
            if score >= score_threshold:
                matched_titles.append(title)

    # Passo 3: montar mapa a partir dos matches
    skill_lookup = raw_skills_df.dropna(subset=["job_title"]).copy()
    skill_lookup["job_title_lower"] = skill_lookup["job_title"].str.lower()
    skill_lookup = skill_lookup.drop_duplicates(subset="job_title_lower")

    for title in matched_titles:
        matched = skill_lookup[skill_lookup["job_title_lower"] == title.lower()]
        if matched.empty:
            continue
        raw_skills = matched.iloc[0].get("skills", "")

        if isinstance(raw_skills, str):
            skills_list = [
                s.strip().lower()
                for s in raw_skills.split(",")
                if s.strip()
            ]
        elif isinstance(raw_skills, list):
            skills_list = [s.strip().lower() for s in raw_skills if s]
        else:
            skills_list = []

        skills_map[title.lower()] = skills_list

    if save_path is None:
        save_path = PROCESSED_DIR / "skills_map.json"
    _ensure_dir(save_path.parent)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(skills_map, f, ensure_ascii=False, indent=2)

    logger.info(
        "Skills map criado: %s títulos mapeados (de %s únicos)",
        len(skills_map), len(unique_titles),
    )
    return skills_map


def compose(
    raw: dict,
    output_dir: Optional[Path] = None,
    skills_threshold: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if output_dir is None:
        output_dir = PROCESSED_DIR
    _ensure_dir(output_dir)

    logger.info("=" * 50)
    logger.info("JobMatch AI — Composição de Datasets")
    logger.info("=" * 50)

    logger.info("1/4 — Preparando DataFrame de vagas (LinkedIn)...")
    jobs_df = raw["linkedin"].copy()
    logger.info("Vagas iniciais: %s", len(jobs_df))

    logger.info("2/4 — Normalizando salários...")
    jobs_df = normalize_salary(jobs_df)

    logger.info("3/4 — Criando campo full_text...")
    jobs_df = add_full_text(jobs_df)

    logger.info("4/4 — Mapeando skills via fuzzy match...")
    skills_map = build_skills_map(jobs_df, raw["job_skills"], skills_threshold)

    jobs_df["required_skills"] = jobs_df["title"].str.lower().map(
        lambda t: skills_map.get(t, [])
    )
    n_with_skills = (jobs_df["required_skills"].str.len() > 0).sum()
    logger.info("Vagas com skills mapeadas: %s", n_with_skills)

    jobs_path = output_dir / "jobs_clean.parquet"
    jobs_df.to_parquet(jobs_path, index=False)
    logger.info("Jobs salvo em: %s", jobs_path)

    pairs_df = raw["resume_jd"].copy()
    pairs_path = output_dir / "training_pairs.parquet"
    pairs_df.to_parquet(pairs_path, index=False)
    logger.info("Pairs salvo em: %s", pairs_path)

    logger.info(
        "Composição concluída: %s vagas, %s pares, %s títulos c/ skills",
        len(jobs_df), len(pairs_df), len(skills_map),
    )

    return jobs_df, pairs_df


def quality_check(
    jobs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
) -> None:
    logger.info("=" * 50)
    logger.info("QUALITY CHECK")
    logger.info("=" * 50)

    logger.info("Jobs DataFrame (%s linhas, %s colunas)", len(jobs_df), len(jobs_df.columns))
    logger.info("Nulos em full_text: %s", jobs_df['full_text'].isna().sum())
    logger.info("Com salário: %s", jobs_df['salary_annual_avg'].notna().sum())

    if "required_skills" in jobs_df.columns:
        n_with_skills = (jobs_df["required_skills"].str.len() > 0).sum()
        logger.info("Com skills mapeadas: %s", n_with_skills)

    logger.info("Salário médio (mediana): $%.0f/ano", jobs_df['salary_annual_avg'].median())
    logger.info("Títulos únicos: %s", jobs_df['title'].nunique())

    logger.info("Pairs DataFrame (%s linhas, %s colunas)", len(pairs_df), len(pairs_df.columns))
    logger.info(
        "Nulos em resume: %s | job_description: %s",
        pairs_df.get('resume', pd.Series()).isna().sum(),
        pairs_df.get('job_description', pd.Series()).isna().sum(),
    )

    if "label" in pairs_df.columns:
        label_counts = pairs_df["label"].value_counts()
        for label, count in label_counts.items():
            pct = count / len(pairs_df) * 100
            logger.info("Label '%s': %s (%.1f%%)", label, count, pct)

        balance = label_counts.min() / len(pairs_df)
        if balance < 0.30:
            logger.warning(
                "Dataset desbalanceado (minoria=%.1f%%). "
                "Considere class_weight='balanced' nos modelos.",
                balance * 100,
            )
        else:
            logger.info("Balanceamento adequado (minoria=%.1f%%)", balance * 100)
    else:
        logger.warning("Coluna 'label' não encontrada")

    logger.info("Quality Check concluído!")


def main(
    linkedin_dest: Optional[Path] = None,
    resume_dest: Optional[Path] = None,
    skills_slug: str = "asaniczka/1-3m-linkedin-jobs-and-skills-2024",
    skills_dest: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    skills_threshold: int = 80,
) -> None:
    from src.pipeline.load_data import load_all

    raw = load_all(linkedin_dest, resume_dest, skills_slug, skills_dest)
    jobs_df, pairs_df = compose(raw, output_dir, skills_threshold)
    quality_check(jobs_df, pairs_df)


if __name__ == "__main__":
    main()
