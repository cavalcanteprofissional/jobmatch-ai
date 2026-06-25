"""
Testes de integração com dados sintéticos realistas.

Marcados como @pytest.mark.slow — pule com -m "not slow" em CI rápido.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from src.models.classifier import predict as classify, train_best
from src.models.recommender import rank_jobs
from src.models.salary_model import predict_salary_range, train_salary_model
from src.models.vectorizer import fit_vectorizer, transform
from src.pipeline.compose_datasets import build_full_text, normalize_salary
from src.pipeline.preprocess import clean_text
from src.skills.skills_analyzer import analyze_gap


pytestmark = pytest.mark.slow


@pytest.fixture
def real_like_jobs() -> pd.DataFrame:
    """DataFrame simulando a estrutura real do LinkedIn Job Postings."""
    np.random.seed(42)
    n = 50
    titles = [
        "Data Scientist", "Data Analyst", "Software Engineer",
        "Machine Learning Engineer", "DevOps Engineer", "Product Manager",
        "Data Engineer", "Business Analyst", "Backend Developer", "Frontend Developer",
    ]
    descriptions = [
        "We are looking for a skilled professional with experience in Python, SQL, "
        "and machine learning. The ideal candidate will have strong analytical skills "
        "and the ability to work with cross-functional teams.",
        "Seeking a data analyst with expertise in SQL, Excel, and data visualization. "
        "Experience with Power BI or Tableau is a plus.",
        "Java developer needed for building scalable microservices. Experience with "
        "Spring Boot, Docker, and Kubernetes is required.",
        "Machine learning engineer with TensorFlow, PyTorch, and MLOps experience. "
        "Must have strong Python skills and understanding of CI/CD pipelines.",
        "DevOps engineer with AWS, Docker, Kubernetes, and Terraform experience. "
        "Responsible for maintaining cloud infrastructure and CI/CD pipelines.",
    ]
    skills_opts = [
        "Python, SQL, Machine Learning, Statistics, Deep Learning",
        "SQL, Excel, Tableau, Power BI, Statistics",
        "Java, Spring Boot, Microservices, Docker, Kubernetes",
        "Python, TensorFlow, PyTorch, Docker, MLOps, CI/CD",
        "AWS, Docker, Kubernetes, Terraform, CI/CD, Linux",
    ]
    companies = ["TechCorp", "DataInc", "SoftSys", "AIMinds", "CloudOps"]

    rows = []
    for i in range(n):
        idx = i % len(titles)
        desc_idx = i % len(descriptions)
        rows.append({
            "title": titles[idx],
            "description": descriptions[desc_idx],
            "skills_desc": skills_opts[desc_idx],
            "company_name": companies[desc_idx],
            "location": "Remote",
            "min_salary": int(np.random.uniform(50000, 150000)),
            "max_salary": int(np.random.uniform(70000, 200000)),
            "pay_period": "YEARLY",
            "work_type": np.random.choice(["Remote", "Hybrid", "On-site"]),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def real_like_pairs() -> pd.DataFrame:
    """Pares currículo-vaga simulando Resume-JD-Match."""
    return pd.DataFrame({
        "resume": [
            "Experienced data scientist with Python, SQL, and machine learning. "
            "Worked on NLP and computer vision projects. Proficient in TensorFlow and PyTorch.",
            "Software engineer with 5 years of Java development. "
            "Built microservices with Spring Boot and deployed on Kubernetes.",
            "Data analyst skilled in SQL, Excel, and data visualization. "
            "Experience creating dashboards in Power BI and Tableau.",
            "DevOps engineer with AWS certifications. Experienced with Docker, "
            "Kubernetes, CI/CD pipelines, and infrastructure as code.",
            "Recent graduate with basic Python. No professional experience.",
            "Marketing professional with 10 years of brand management experience. "
            "Expert in social media strategy and content marketing.",
            "Data scientist with Python and SQL. No cloud experience.",
            "Java developer with Spring Boot experience. No Docker or Kubernetes.",
            "HR generalist with recruiting and payroll experience.",
            "Frontend developer with React and TypeScript skills.",
        ],
        "job_description": [
            "Senior data scientist needed for ML model development. Must know "
            "Python, TensorFlow, and have experience deploying models to production.",
            "Backend Java developer for microservices architecture. Spring Boot, "
            "Kafka, and Docker experience required.",
            "Data analyst position requiring SQL, Power BI, and Excel skills. "
            "Tableau experience is a bonus.",
            "Cloud DevOps engineer. Must have AWS, Docker, Kubernetes, and "
            "Terraform experience. CI/CD pipeline management.",
            "Junior Python developer. Training provided. Basic Python knowledge required.",
            "Marketing manager with digital marketing expertise. Social media, "
            "content strategy, and analytics experience required.",
            "Senior data scientist position. Must have TensorFlow, PyTorch, and "
            "experience with MLOps and cloud deployment.",
            "Senior backend engineer. Must have Java, Spring Boot, Docker, "
            "Kubernetes, and microservices architecture experience.",
            "HR coordinator position. Experience with HRIS, payroll processing, "
            "and employee relations required.",
            "Senior frontend engineer. React, TypeScript, and Next.js experience "
            "required. Must have experience with state management.",
        ],
        "label": ["Fit", "Fit", "Fit", "Fit", "No Fit",
                  "No Fit", "No Fit", "No Fit", "No Fit", "Fit"],
    })


class TestIntegrationWithRealData:
    def test_pipeline_end_to_end(self, real_like_jobs, real_like_pairs):
        # 1. Normalize salaries
        jobs = normalize_salary(real_like_jobs)
        assert "salary_annual_avg" in jobs.columns
        assert jobs["salary_annual_avg"].notna().sum() > 0

        # 2. Build full text
        jobs["full_text"] = jobs.apply(build_full_text, axis=1)
        assert jobs["full_text"].str.len().min() > 0

        # 3. Vectorize
        corpus = jobs["full_text"].tolist()
        vec = fit_vectorizer(corpus, max_features=500)
        assert len(vec.vocabulary_) > 0

        jobs_matrix = transform(corpus, vec)
        assert jobs_matrix.shape[0] == len(jobs)

        # 4. Train classifier
        pairs_clean = real_like_pairs.copy()
        pairs_clean["combined"] = (
            pairs_clean["resume"].fillna("").apply(clean_text) + " " +
            pairs_clean["job_description"].fillna("").apply(clean_text)
        )
        X = transform(pairs_clean["combined"].tolist(), vec)
        y = (pairs_clean["label"] == "Fit").astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42,
        )
        name, clf = train_best(X_train, y_train, cv=2)
        assert isinstance(name, str)

        # 5. Predict
        resume_clean = clean_text("Data scientist with Python, SQL, and ML")
        resume_vec = transform([resume_clean], vec)
        label, prob = classify(resume_vec, clf)
        assert label in ("Fit", "No Fit")
        assert 0.0 <= prob <= 1.0

        # 6. Rank jobs
        result = rank_jobs(resume_vec, jobs_matrix, jobs, top_k=3)
        assert len(result) == 3
        assert "adherence_score" in result.columns

        # 7. Train salary model
        with_salary = jobs.dropna(subset=["salary_annual_avg"])
        if len(with_salary) > 5:
            X_sal = transform(with_salary["full_text"].tolist(), vec)
            y_sal = with_salary["salary_annual_avg"].values
            reg = train_salary_model(X_sal, y_sal, cv=2)
            job_vec = transform([jobs["full_text"].iloc[0]], vec)
            sal = predict_salary_range(job_vec, reg)
            assert "estimated_annual_usd" in sal
            assert sal["range_low"] <= sal["estimated_annual_usd"] <= sal["range_high"]

        # 8. Skills gap analysis
        gap = analyze_gap(resume_clean, "Data Scientist")
        assert "compatible" in gap
        assert "missing" in gap
        assert "development_plan" in gap
