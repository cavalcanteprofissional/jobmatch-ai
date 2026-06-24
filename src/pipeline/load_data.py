import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset

from src.utils.config import get_kaggle_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

RAW_DIR = Path("data/raw")


def _configure_kaggle() -> None:
    config = get_kaggle_config()
    if not config["username"] or not config["key"]:
        logger.warning(
            "KAGGLE_USERNAME / KAGGLE_KEY não definidos. "
            "Downloads do Kaggle falharão se a CLI não estiver autenticada."
        )
        return

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        return

    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json.write_text(
        json.dumps({"username": config["username"], "key": config["key"]})
    )
    kaggle_json.chmod(0o600)
    logger.info("Credenciais Kaggle configuradas em %s", kaggle_json)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_csv(directory: Path, hint: str = "") -> Optional[Path]:
    csvs = list(directory.glob("*.csv"))
    if not csvs:
        return None
    if hint:
        for c in csvs:
            if hint in c.name.lower():
                return c
    return csvs[0]


def download_linkedin_jobs(dest: Optional[Path] = None) -> pd.DataFrame:
    if dest is None:
        dest = RAW_DIR
    _ensure_dir(dest)

    _configure_kaggle()

    logger.info("Baixando LinkedIn Job Postings do Kaggle...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "arshkon/linkedin-job-postings",
            "-p", str(dest), "--unzip",
        ],
        check=True,
    )

    csv_path = _find_csv(dest, hint="linkedin")
    if csv_path is None:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {dest} após download do LinkedIn."
        )

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"LinkedIn jobs carregado: %s linhas, %s colunas", len(df), len(df.columns))
    return df


def download_resume_jd(dest: Optional[Path] = None) -> pd.DataFrame:
    if dest is None:
        dest = RAW_DIR
    _ensure_dir(dest)

    logger.info("Baixando Resume-JD-Match do HuggingFace...")
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")
    df = pd.DataFrame(ds["train"])

    col_map = {
        "resume": ["resume", "resume_text", "resume_str"],
        "job_description": ["job_description", "jd", "description", "job_desc"],
        "label": ["label", "fit_label", "target"],
    }

    for standard_name, alternatives in col_map.items():
        if standard_name not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    df.rename(columns={alt: standard_name}, inplace=True)
                    break
            else:
                raise ValueError(
                    f"Coluna '{standard_name}' não encontrada no dataset "
                    f"Resume-JD-Match. Colunas disponíveis: {list(df.columns)}"
                )

    df["label"] = df["label"].astype(str)

    parquet_path = dest / "resume_jd_train.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info("Resume-JD-Match carregado: %s pares", len(df))
    return df


def download_job_skills(
    slug: str,
    dest: Optional[Path] = None,
) -> pd.DataFrame:
    if dest is None:
        dest = RAW_DIR
    _ensure_dir(dest)

    _configure_kaggle()

    logger.info("Baixando Job Skill Set do Kaggle (%s)...", slug)
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", slug,
            "-p", str(dest), "--unzip",
        ],
        check=True,
    )

    csv_path = _find_csv(dest, hint="skill")
    if csv_path is None:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {dest} para o dataset {slug}."
        )

    df = pd.read_csv(csv_path, low_memory=False)

    col_map = {
        "job_title": ["job_title", "title", "job_title_name", "position"],
        "skills": ["skills", "skill_list", "required_skills", "skills_required"],
    }
    for standard_name, alternatives in col_map.items():
        if standard_name not in df.columns:
            for alt in alternatives:
                if alt in df.columns:
                    df.rename(columns={alt: standard_name}, inplace=True)
                    break

    if "job_title" not in df.columns or "skills" not in df.columns:
        raise ValueError(
            f"Colunas esperadas 'job_title' e 'skills' não encontradas. "
            f"Disponíveis: {list(df.columns)}"
        )

    logger.info("Job Skills carregado: %s linhas", len(df))
    return df


def load_all(
    linkedin_dest: Optional[Path] = None,
    resume_dest: Optional[Path] = None,
    skills_slug: str = "asaniczka/1-3m-linkedin-jobs-and-skills-2024",
    skills_dest: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    logger.info("=" * 50)
    logger.info("JobMatch AI — Pipeline de Dados")
    logger.info("=" * 50)

    linkedin = download_linkedin_jobs(linkedin_dest)
    resume_jd = download_resume_jd(resume_dest)
    job_skills = download_job_skills(skills_slug, skills_dest)

    logger.info("Todos os datasets carregados com sucesso!")
    return {
        "linkedin": linkedin,
        "resume_jd": resume_jd,
        "job_skills": job_skills,
    }


if __name__ == "__main__":
    raw = load_all()
    logger.info("LinkedIn: %s", raw['linkedin'].shape)
    logger.info("Resume-JD: %s", raw['resume_jd'].shape)
    logger.info("Job Skills: %s", raw['job_skills'].shape)
