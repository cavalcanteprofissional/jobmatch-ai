import numpy as np
import pandas as pd
import pytest

from src.pipeline.compose_datasets import (
    PERIOD_TO_ANNUAL,
    build_full_text,
    build_skills_map,
    normalize_salary,
    quality_check,
)


class TestNormalizeSalary:
    def test_basic_conversion(self):
        df = pd.DataFrame({
            "min_salary": [50],
            "max_salary": [70],
            "pay_period": ["HOURLY"],
        })
        result = normalize_salary(df)
        expected = 50 * PERIOD_TO_ANNUAL["HOURLY"]
        assert result["min_salary_annual"].iloc[0] == expected

    def test_yearly_passthrough(self):
        df = pd.DataFrame({
            "min_salary": [100000],
            "max_salary": [150000],
            "pay_period": ["YEARLY"],
        })
        result = normalize_salary(df)
        assert result["min_salary_annual"].iloc[0] == 100000
        assert result["max_salary_annual"].iloc[0] == 150000

    def test_missing_salary_columns(self):
        df = pd.DataFrame({"title": ["Data Scientist"]})
        result = normalize_salary(df)
        assert "salary_annual_avg" in result.columns
        assert pd.isna(result["salary_annual_avg"].iloc[0])

    def test_outlier_removal(self):
        np.random.seed(42)
        salaries = np.random.normal(100000, 20000, 1000).tolist()
        salaries.extend([1_000_000_000, 1])
        df = pd.DataFrame({
            "min_salary": salaries,
            "max_salary": salaries,
            "pay_period": "YEARLY",
        })
        result = normalize_salary(df)
        assert len(result) < len(df)


class TestBuildFullText:
    def test_basic(self):
        row = pd.Series({
            "title": "Data Scientist",
            "description": "Python and SQL experience",
            "skills_desc": "Python, SQL, ML",
        })
        text = build_full_text(row)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_missing_columns(self):
        row = pd.Series({"title": "Engineer"})
        text = build_full_text(row)
        assert isinstance(text, str)

    def test_nan_values(self):
        row = pd.Series({
            "title": "Analyst",
            "description": None,
            "skills_desc": "nan",
        })
        text = build_full_text(row)
        assert isinstance(text, str)
        assert len(text) > 0


class TestBuildSkillsMap:
    def test_matching_titles(self, sample_jobs_df, sample_skills_df):
        skills_map = build_skills_map(sample_jobs_df, sample_skills_df, score_threshold=50)
        assert len(skills_map) > 0
        assert "data scientist" in skills_map
        assert "python" in skills_map["data scientist"]

    def test_no_titles_in_skills(self, sample_jobs_df):
        empty_skills = pd.DataFrame({"job_title": pd.Series([], dtype=str), "skills": pd.Series([], dtype=str)})
        skills_map = build_skills_map(sample_jobs_df, empty_skills)
        assert skills_map == {}

    def test_high_threshold_no_match(self):
        jobs = pd.DataFrame({"title": ["Totally Unique Title XYZ 123"]})
        skills = pd.DataFrame({"job_title": ["Data Scientist"], "skills": ["python"]})
        skills_map = build_skills_map(jobs, skills, score_threshold=100)
        assert len(skills_map) == 0


class TestQualityCheck:
    def test_runs_without_error(self, sample_jobs_df, sample_pairs_df):
        sample_jobs_df["full_text"] = "data scientist python"
        sample_jobs_df["salary_annual_avg"] = 100000
        sample_jobs_df["required_skills"] = [[] for _ in range(len(sample_jobs_df))]
        quality_check(sample_jobs_df, sample_pairs_df)

    def test_balanced_dataset(self, sample_jobs_df):
        pairs = pd.DataFrame({"label": ["Fit", "Fit", "No Fit", "No Fit"]})
        sample_jobs_df["full_text"] = "test"
        sample_jobs_df["salary_annual_avg"] = 100000
        sample_jobs_df["required_skills"] = [[] for _ in range(len(sample_jobs_df))]
        quality_check(sample_jobs_df, pairs)
