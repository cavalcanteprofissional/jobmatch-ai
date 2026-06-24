"""
Fixtures compartilhadas para os testes do JobMatch AI.

Fornece dados sintéticos e helpers para todos os módulos de teste.
"""

from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.pipeline.preprocess import clean_text


@pytest.fixture
def sample_resume_text() -> str:
    """Texto de currículo sintético para testes."""
    return (
        "Data Scientist with 5 years of experience in Python, SQL, and Machine Learning. "
        "Skilled in TensorFlow, scikit-learn, and Docker. "
        "Worked on NLP projects and deployed models to AWS."
    )


@pytest.fixture
def sample_jobs_df() -> pd.DataFrame:
    """DataFrame sintético de 5 vagas para testes de ranking e composição."""
    return pd.DataFrame({
        "title": [
            "Data Scientist",
            "Data Analyst",
            "Software Engineer",
            "Machine Learning Engineer",
            "DevOps Engineer",
        ],
        "description": [
            "We need a data scientist with Python and ML experience.",
            "Looking for a data analyst with SQL and Power BI skills.",
            "Java developer for backend microservices.",
            "ML engineer with TensorFlow and Docker experience.",
            "DevOps with AWS, Docker and Kubernetes.",
        ],
        "skills_desc": [
            "Python, SQL, Machine Learning, Statistics",
            "SQL, Excel, Power BI, Tableau",
            "Java, Spring Boot, Microservices",
            "Python, TensorFlow, Docker, MLOps",
            "AWS, Docker, Kubernetes, CI/CD",
        ],
        "company_name": [
            "TechCorp", "DataInc", "SoftSys", "AIMinds", "CloudOps",
        ],
        "location": [
            "São Paulo, BR", "Rio de Janeiro, BR", "NY, USA", "SF, USA", "Berlin, DE",
        ],
        "max_salary": [150000, 80000, 120000, 160000, 130000],
        "min_salary": [100000, 50000, 80000, 110000, 90000],
        "pay_period": ["YEARLY", "YEARLY", "YEARLY", "YEARLY", "YEARLY"],
        "work_type": ["Remote", "Hybrid", "On-site", "Remote", "Remote"],
    })


@pytest.fixture
def sample_pairs_df() -> pd.DataFrame:
    """DataFrame sintético de pares currículo-vaga para classificador."""
    return pd.DataFrame({
        "resume": [
            "Data scientist expert in Python and ML",
            "Software engineer Java Spring Boot",
            "Data analyst with SQL and Excel",
            "DevOps engineer AWS Docker",
            "ML engineer TensorFlow Python",
        ],
        "job_description": [
            "Looking for data scientist with ML experience",
            "Java developer for microservices",
            "Data analyst SQL Power BI position",
            "DevOps with Kubernetes experience",
            "Machine learning engineer for NLP tasks",
        ],
        "label": ["Fit", "Fit", "Fit", "No Fit", "Fit"],
    })


@pytest.fixture
def sample_skills_df() -> pd.DataFrame:
    """DataFrame sintético de skills por cargo."""
    return pd.DataFrame({
        "job_title": [
            "Data Scientist",
            "Data Analyst",
            "Machine Learning Engineer",
            "Software Engineer",
            "DevOps Engineer",
        ],
        "skills": [
            "python, sql, machine learning, statistics, deep learning",
            "sql, excel, tableau, power bi",
            "python, tensorflow, pytorch, docker, mlops",
            "java, python, javascript, docker, kubernetes",
            "aws, docker, kubernetes, ci/cd, terraform",
        ],
    })


@pytest.fixture
def sample_skills_map(tmp_path: Path) -> Path:
    """Salva um skills_map.json temporário e retorna o caminho."""
    import json
    data = {
        "data scientist": ["python", "sql", "machine learning", "statistics"],
        "data analyst": ["sql", "excel", "power bi"],
        "software engineer": ["java", "python", "docker"],
        "machine learning engineer": ["python", "tensorflow", "docker"],
        "devops engineer": ["aws", "docker", "kubernetes"],
    }
    path = tmp_path / "skills_map.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def sample_tfidf_corpus() -> list[str]:
    """Corpus pequeno para testar TF-IDF."""
    return [
        "data scientist python machine learning",
        "software engineer java python docker",
        "data analyst sql power bi tableau",
        "devops engineer aws docker kubernetes",
        "machine learning engineer tensorflow python",
    ]


@pytest.fixture
def sample_tfidf_matrix(sample_tfidf_corpus):
    """Matriz TF-IDF esparsa sintética (5 docs, 20 features)."""
    rng = np.random.default_rng(42)
    return csr_matrix(rng.random((5, 20)))


@pytest.fixture
def sample_resume_vec() -> csr_matrix:
    """Vetor TF-IDF de currículo sintético (1, 20)."""
    rng = np.random.default_rng(42)
    return csr_matrix(rng.random((1, 20)))


@pytest.fixture
def mock_classifier():
    """Mock de classificador para testes de predict()."""
    from sklearn.linear_model import LogisticRegression
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([1, 0, 1, 0])
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf


@pytest.fixture
def mock_regressor():
    """Mock de regressor para testes de predict_salary_range()."""
    from sklearn.dummy import DummyRegressor
    reg = DummyRegressor(strategy="constant", constant=100000)
    reg.fit([[0]], [100000])
    return reg
