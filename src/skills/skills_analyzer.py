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
    "java": {"curso": "Java Programming Masterclass (Udemy)", "tempo": "~8 semanas"},
    "javascript": {"curso": "The Complete JavaScript Course (Udemy)", "tempo": "~6 semanas"},
    "typescript": {"curso": "Understanding TypeScript (Udemy)", "tempo": "~4 semanas"},
    "react": {"curso": "React - The Complete Guide (Udemy)", "tempo": "~6 semanas"},
    "angular": {"curso": "Angular - The Complete Guide (Udemy)", "tempo": "~6 semanas"},
    "node.js": {"curso": "Node.js Complete Guide (Udemy)", "tempo": "~5 semanas"},
    "html": {"curso": "HTML & CSS Crash Course (freeCodeCamp)", "tempo": "~2 semanas"},
    "css": {"curso": "HTML & CSS Crash Course (freeCodeCamp)", "tempo": "~2 semanas"},
    "c++": {"curso": "C++ Programming Fundamentals (Coursera)", "tempo": "~6 semanas"},
    "c#": {"curso": "C# Basics for Beginners (Udemy)", "tempo": "~5 semanas"},
    "go": {"curso": "Go - The Complete Guide (Udemy)", "tempo": "~4 semanas"},
    "rust": {"curso": "Rust Programming Language (Udemy)", "tempo": "~5 semanas"},
    "r": {"curso": "R Programming for Data Science (Coursera)", "tempo": "~4 semanas"},
    "matlab": {"curso": "MATLAB Programming (Coursera)", "tempo": "~4 semanas"},
    "linux": {"curso": "Linux Command Line Basics (Udemy)", "tempo": "~3 semanas"},
    "bash": {"curso": "Shell Scripting for Beginners (Udemy)", "tempo": "~2 semanas"},
    "postgresql": {"curso": "PostgreSQL Bootcamp (Udemy)", "tempo": "~5 semanas"},
    "mongodb": {"curso": "MongoDB Complete Guide (Udemy)", "tempo": "~4 semanas"},
    "redis": {"curso": "Redis Bootcamp (Udemy)", "tempo": "~2 semanas"},
    "kafka": {"curso": "Apache Kafka Series (Conduktor)", "tempo": "~3 semanas"},
    "terraform": {"curso": "Terraform for Beginners (Udemy)", "tempo": "~4 semanas"},
    "ansible": {"curso": "Ansible for DevOps (Udemy)", "tempo": "~3 semanas"},
    "jenkins": {"curso": "Jenkins CI/CD Pipeline (Udemy)", "tempo": "~3 semanas"},
    "github actions": {"curso": "GitHub Actions CI/CD (freeCodeCamp)", "tempo": "~2 semanas"},
    "ci/cd": {"curso": "CI/CD Pipeline Guide (Udemy)", "tempo": "~3 semanas"},
    "rest api": {"curso": "REST API Design (Coursera)", "tempo": "~3 semanas"},
    "graphql": {"curso": "GraphQL Full Course (freeCodeCamp)", "tempo": "~2 semanas"},
    "agile": {"curso": "Agile & Scrum Fundamentals (Coursera)", "tempo": "~3 semanas"},
    "scrum": {"curso": "Scrum Master Certification (Udemy)", "tempo": "~4 semanas"},
    "jira": {"curso": "Jira for Project Management (Udemy)", "tempo": "~2 semanas"},
    "sap": {"curso": "SAP ERP Fundamentals (Coursera)", "tempo": "~6 semanas"},
    "oracle": {"curso": "Oracle SQL & Database (Udemy)", "tempo": "~5 semanas"},
    "salesforce": {"curso": "Salesforce Admin Certification (Udemy)", "tempo": "~6 semanas"},
    "hadoop": {"curso": "Hadoop & Big Data (Udemy)", "tempo": "~6 semanas"},
    "nlp": {"curso": "NLP with Python (Coursera)", "tempo": "~5 semanas"},
    "computer vision": {"curso": "Computer Vision (OpenCV) (Udemy)", "tempo": "~5 semanas"},
    "llm": {"curso": "LLM Specialization (Coursera)", "tempo": "~6 semanas"},
    "langchain": {"curso": "LangChain for LLM Apps (Udemy)", "tempo": "~3 semanas"},
    "rag": {"curso": "RAG Systems (freeCodeCamp)", "tempo": "~3 semanas"},
    "data engineering": {"curso": "Data Engineering (DataCamp)", "tempo": "~8 semanas"},
    "data pipeline": {"curso": "Data Pipeline with Airflow (Udemy)", "tempo": "~4 semanas"},
    "etl": {"curso": "ETL & Data Warehousing (Coursera)", "tempo": "~5 semanas"},
    "data warehouse": {"curso": "Data Warehousing (Coursera)", "tempo": "~4 semanas"},
    "dbt": {"curso": "dBT Fundamentals (Udemy)", "tempo": "~2 semanas"},
    "snowflake": {"curso": "Snowflake Cloud DW (Udemy)", "tempo": "~3 semanas"},
    "bigquery": {"curso": "Google BigQuery (Coursera)", "tempo": "~2 semanas"},
    "databricks": {"curso": "Databricks Lakehouse (Udemy)", "tempo": "~4 semanas"},
    "communication skills": {"curso": "Communication Skills (Coursera)", "tempo": "~3 semanas"},
    "project management": {"curso": "Google PM Certificate (Coursera)", "tempo": "~6 meses"},
    "leadership": {"curso": "Leadership Specialization (Coursera)", "tempo": "~4 semanas"},
    "problem solving": {"curso": "Problem Solving Skills (LinkedIn)", "tempo": "~2 semanas"},
    "time management": {"curso": "Time Management (Coursera)", "tempo": "~2 semanas"},
    "customer service": {"curso": "Customer Service (HubSpot)", "tempo": "~2 semanas"},
}

_SYNONYMS = {
    "tensorflow": {"tf", "tensor flow"},
    "pytorch": {"torch", "py torch"},
    "scikit-learn": {"sklearn", "scikit learn"},
    "natural language processing": {"nlp"},
    "computer vision": {"cv", "machine vision"},
    "deep learning": {"dl"},
    "machine learning": {"ml"},
    "artificial intelligence": {"ai"},
    "aws": {"amazon web services", "amazon aws"},
    "gcp": {"google cloud platform", "google cloud"},
    "azure": {"microsoft azure", "ms azure"},
    "kubernetes": {"k8s"},
    "javascript": {"js"},
    "typescript": {"ts"},
    "node.js": {"nodejs", "node"},
    "react": {"reactjs", "react.js"},
    "c++": {"cpp", "cplusplus"},
    "c#": {"csharp", "c sharp"},
    "power bi": {"powerbi", "microsoft power bi"},
    "tableau": {"tableau desktop"},
    "postgresql": {"postgres"},
    "github actions": {"gh actions"},
    "ci/cd": {"cicd", "continuous integration", "continuous delivery"},
    "rest api": {"restful", "rest"},
    "agile": {"agile methodology"},
    "scrum": {"scrum master"},
    "data engineering": {"data engineer"},
    "data science": {"data scientist"},
}


def extract_skills_from_text(text: str, skills_vocabulary: set) -> set:
    text_lower = text.lower()
    found = set()

    # Busca direta por palavra-chave
    for skill in skills_vocabulary:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)

    # Busca por sinônimos
    for canonical, aliases in _SYNONYMS.items():
        if canonical in skills_vocabulary and canonical not in found:
            for alias in aliases:
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, text_lower):
                    found.add(canonical)
                    break

    logger.debug("Extraídas %s skills de %s no vocabulário", len(found), len(skills_vocabulary))
    return found


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
