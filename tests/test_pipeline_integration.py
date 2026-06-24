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


class TestPipelineSmoke:
    @pytest.fixture(autouse=True)
    def setup(self, sample_skills_map):
        self.jobs = pd.DataFrame({
            "title": [
                "Data Scientist", "Data Analyst", "Engineer", "ML Engineer", "DevOps",
                "Data Scientist", "Data Analyst", "Engineer", "ML Engineer", "DevOps",
            ],
            "description": [
                "Python SQL ML", "SQL Excel BI", "Java Spring", "Python TF Docker", "AWS Docker",
                "Python ML NLP", "SQL PowerBI", "Java Kafka", "Python PyTorch", "AWS K8s",
            ],
            "skills_desc": [
                "python, sql", "sql, excel", "java", "python, tf", "aws, docker",
                "python, nlp", "sql, bi", "java, kafka", "python, torch", "aws, k8s",
            ],
            "company_name": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "location": ["BR"] * 10,
            "max_salary": [150000, 80000, 120000, 160000, 130000] * 2,
            "min_salary": [100000, 50000, 80000, 110000, 90000] * 2,
            "pay_period": ["YEARLY"] * 10,
        })
        self.pairs = pd.DataFrame({
            "resume": [
                "Python SQL ML", "Python SQL ML", "Java AWS", "Java AWS",
                "Python SQL ML", "Python SQL ML", "Java AWS", "Java AWS",
            ],
            "job_description": [
                "Python SQL ML", "Python SQL ML", "Java AWS", "Java AWS",
                "Java AWS", "Java AWS", "Python SQL ML", "Python SQL ML",
            ],
            "label": [
                "Fit", "Fit", "Fit", "Fit",
                "No Fit", "No Fit", "No Fit", "No Fit",
            ],
        })
        self.skills_map = sample_skills_map

    def test_full_pipeline(self):
        resume_clean = clean_text("Data Scientist with Python, SQL, and ML")
        assert len(resume_clean) > 0

        jobs_norm = normalize_salary(self.jobs)
        assert "salary_annual_avg" in jobs_norm.columns

        jobs_norm["full_text"] = jobs_norm.apply(build_full_text, axis=1)
        assert jobs_norm["full_text"].str.len().min() > 0

        corpus = jobs_norm["full_text"].tolist()
        vec = fit_vectorizer(corpus, max_features=100)
        assert len(vec.vocabulary_) > 0

        jobs_matrix = transform(corpus, vec)
        assert jobs_matrix.shape[0] == len(self.jobs)

        pairs_clean = self.pairs.copy()
        pairs_clean["combined"] = (
            pairs_clean["resume"].fillna("").apply(clean_text) + " " +
            pairs_clean["job_description"].fillna("").apply(clean_text)
        )
        X = transform(pairs_clean["combined"].tolist(), vec)
        y = (pairs_clean["label"] == "Fit").astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42,
        )
        name, clf = train_best(X_train, y_train, cv=2)
        assert isinstance(name, str)

        resume_vec = transform([resume_clean], vec)
        label, prob = classify(resume_vec, clf)
        assert label in ("Fit", "No Fit")
        assert 0.0 <= prob <= 1.0

        result = rank_jobs(resume_vec, jobs_matrix, jobs_norm, top_k=3)
        assert len(result) == 3
        assert "adherence_score" in result.columns

        job_vec = transform([jobs_norm["full_text"].iloc[0]], vec)
        y_sal = jobs_norm["salary_annual_avg"].dropna().values
        if len(y_sal) > 1:
            X_sal = transform(jobs_norm.loc[jobs_norm["salary_annual_avg"].notna(), "full_text"].tolist(), vec)
            reg = train_salary_model(X_sal, y_sal, cv=2)
            sal = predict_salary_range(job_vec, reg)
            assert "estimated_annual_usd" in sal
            assert sal["range_low"] <= sal["estimated_annual_usd"]

        gap = analyze_gap(resume_clean, "Data Scientist", self.skills_map)
        assert "compatible" in gap
        assert "missing" in gap
