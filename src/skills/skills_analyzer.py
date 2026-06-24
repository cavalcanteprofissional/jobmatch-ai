import json
import re
from pathlib import Path
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

SKILLS_MAP_PATH = Path("data/processed/skills_map.json")

_RESOURCES = {
    "python": {"curso": "Python for Everybody (Coursera)", "tempo": "~6 semanas"},
    "sql": {"curso": "SQL for Data Science (Coursera)", "tempo": "~4 semanas"},
    "machine learning": {"curso": "ML Specialization - Andrew Ng (Coursera)", "tempo": "~3 meses"},
    "deep learning": {"curso": "Deep Learning Specialization (Coursera)", "tempo": "~3 meses"},
    "power bi": {"curso": "Microsoft Power BI Desktop (Udemy)", "tempo": "~3 semanas"},
    "tableau": {"curso": "Tableau 2023 A-Z (Udemy)", "tempo": "~3 semanas"},
    "docker": {"curso": "Docker & Kubernetes: The Practical Guide (Udemy)", "tempo": "~4 semanas"},
    "kubernetes": {"curso": "Docker & Kubernetes: The Practical Guide (Udemy)", "tempo": "~4 semanas"},
    "aws": {"curso": "AWS Solutions Architect (Coursera)", "tempo": "~8 semanas"},
    "azure": {"curso": "Microsoft Azure Fundamentals (Microsoft Learn)", "tempo": "~4 semanas"},
    "spark": {"curso": "Apache Spark with Python (Udemy)", "tempo": "~5 semanas"},
    "airflow": {"curso": "Apache Airflow: The Hands-On Guide (Udemy)", "tempo": "~3 semanas"},
    "git": {"curso": "Git & GitHub Crash Course (freeCodeCamp)", "tempo": "~2 semanas"},
    "tensorflow": {"curso": "TensorFlow Developer Certificate (Coursera)", "tempo": "~8 semanas"},
    "pytorch": {"curso": "PyTorch for Deep Learning (Udemy)", "tempo": "~6 semanas"},
    "scikit-learn": {"curso": "Machine Learning with scikit-learn (DataCamp)", "tempo": "~4 semanas"},
    "pandas": {"curso": "Pandas for Data Analysis (DataCamp)", "tempo": "~3 semanas"},
    "numpy": {"curso": "NumPy for Data Science (DataCamp)", "tempo": "~2 semanas"},
    "excel": {"curso": "Excel Skills for Business (Coursera)", "tempo": "~4 semanas"},
}


def extract_skills_from_text(text: str, skills_vocabulary: set) -> set:
    text_lower = text.lower()
    found = set()
    for skill in skills_vocabulary:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    logger.debug("Extraídas %s skills de %s no vocabulário", len(found), len(skills_vocabulary))
    return found


def analyze_gap(
    resume_text: str,
    job_title: str,
    skills_map_path: Optional[Path] = None,
) -> dict:
    if skills_map_path is None:
        skills_map_path = SKILLS_MAP_PATH

    if not skills_map_path.exists():
        logger.warning("Skills map não encontrado em %s", skills_map_path)
        return {"compatible": [], "missing": [], "development_plan": []}

    with open(skills_map_path, encoding="utf-8") as f:
        skills_map = json.load(f)

    norm_title = job_title.lower().strip()
    required = set(skills_map.get(norm_title, []))

    if not required:
        logger.info("Nenhuma skill mapeada para o título '%s'", job_title)
        return {"compatible": [], "missing": [], "development_plan": []}

    candidate_skills = extract_skills_from_text(resume_text, required)
    missing = required - candidate_skills

    logger.info(
        "Gap analysis para '%s': %s compatíveis, %s faltantes",
        job_title, len(candidate_skills), len(missing),
    )

    return {
        "compatible": sorted(candidate_skills),
        "missing": sorted(missing),
        "development_plan": _generate_plan(missing),
    }


def _generate_plan(missing_skills: set) -> list[dict]:
    plan = []
    for skill in sorted(missing_skills):
        resource = _RESOURCES.get(
            skill.lower(),
            {"curso": f"Buscar curso de {skill} no Coursera/Udemy", "tempo": "~2-6 semanas"},
        )
        plan.append({"skill": skill, **resource})
    return plan
