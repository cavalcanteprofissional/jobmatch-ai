from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipeline.load_data import (
    _find_csv,
    download_linkedin_jobs,
    download_resume_jd,
    download_job_skills,
)


class TestFindCsv:
    def test_no_csv_dir(self, tmp_path):
        assert _find_csv(tmp_path) is None

    def test_find_with_hint(self, tmp_path):
        (tmp_path / "linkedin_jobs_2023.csv").touch()
        (tmp_path / "other.csv").touch()
        result = _find_csv(tmp_path, hint="linkedin")
        assert result is not None
        assert "linkedin" in result.name

    def test_first_csv_no_hint(self, tmp_path):
        (tmp_path / "aaa.csv").touch()
        (tmp_path / "bbb.csv").touch()
        result = _find_csv(tmp_path)
        assert result is not None


class TestDownloadResumeJD:
    def test_column_normalization(self, monkeypatch, tmp_path):
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = pd.DataFrame({
            "resume_text": ["abc"],
            "jd": ["def"],
            "fit_label": ["Fit"],
        })
        monkeypatch.setattr("src.pipeline.load_data.load_dataset", lambda x: mock_ds)
        df = download_resume_jd(tmp_path)
        assert "resume" in df.columns
        assert "job_description" in df.columns
        assert "label" in df.columns
        assert df.iloc[0]["resume"] == "abc"
        assert df.iloc[0]["label"] == "Fit"

    def test_missing_column_raises(self, monkeypatch, tmp_path):
        mock_ds = MagicMock()
        mock_ds.__getitem__.return_value = pd.DataFrame({"unknown": ["x"]})
        monkeypatch.setattr("src.pipeline.load_data.load_dataset", lambda x: mock_ds)
        with pytest.raises(ValueError, match="resume"):
            download_resume_jd(tmp_path)


class TestDownloadJobSkills:
    def test_column_normalization(self, monkeypatch, tmp_path):
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
        mock_df = pd.DataFrame({
            "title": ["Data Scientist"],
            "skill_list": ["python, sql"],
        })
        monkeypatch.setattr("pandas.read_csv", lambda *a, **kw: mock_df)
        monkeypatch.setattr(
            "src.pipeline.load_data._find_csv",
            lambda *a, **kw: Path("dummy.csv"),
        )
        df = download_job_skills("any/slug", tmp_path)
        assert "job_title" in df.columns
        assert "skills" in df.columns

    def test_missing_columns_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
        monkeypatch.setattr("pandas.read_csv", lambda *a, **kw: pd.DataFrame({"a": [1]}))
        monkeypatch.setattr(
            "src.pipeline.load_data._find_csv",
            lambda *a, **kw: Path("dummy.csv"),
        )
        with pytest.raises(ValueError, match="job_title"):
            download_job_skills("any/slug", tmp_path)
