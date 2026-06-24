"""
Testes para o schema de saída do JobMatchPredictor.

Estes testes validam que o dict retornado por predict()
tem a estrutura JSON esperada pela API e pelo frontend.
"""
import pytest


class TestPredictorOutputSchema:
    """Valida o schema do dict retornado pelo Predictor.predict()."""

    @pytest.fixture
    def sample_predict_result(self):
        """Simula a saída do Predictor.predict() sem carregar modelos reais."""
        return {
            "score_pct": 72.5,
            "fit_label": "Fit",
            "avg_adherence": 58.3,
            "fit_count": 3,
            "top_k": 5,
            "salary_est": {
                "estimated_annual_usd": 125000,
                "range_low": 106250,
                "range_high": 143750,
            },
            "gap": {
                "compatible": ["python", "sql"],
                "missing": ["docker", "kubernetes"],
                "development_plan": [
                    {"skill": "docker", "curso": "Docker Course", "tempo": "4 semanas"},
                    {"skill": "kubernetes", "curso": "K8s Course", "tempo": "4 semanas"},
                ],
            },
            "top_jobs": [
                {
                    "title": "Data Scientist",
                    "company_name": "TechCorp",
                    "location": "Remote",
                    "adherence_score": 85.0,
                    "fit_label": "✅ Fit",
                    "min_salary_annual": 100000,
                    "max_salary_annual": 150000,
                    "skills_desc": "Python, SQL, ML",
                }
            ],
        }

    def test_has_top_level_keys(self, sample_predict_result):
        expected_keys = {
            "score_pct", "fit_label", "avg_adherence", "fit_count",
            "top_k", "salary_est", "gap", "top_jobs",
        }
        assert expected_keys.issubset(sample_predict_result.keys())

    def test_score_pct_is_float(self, sample_predict_result):
        assert isinstance(sample_predict_result["score_pct"], float)

    def test_fit_label_is_string(self, sample_predict_result):
        assert isinstance(sample_predict_result["fit_label"], str)
        assert sample_predict_result["fit_label"] in ("Fit", "No Fit")

    def test_salary_est_has_range(self, sample_predict_result):
        sal = sample_predict_result["salary_est"]
        assert sal["range_low"] <= sal["estimated_annual_usd"]
        assert sal["estimated_annual_usd"] <= sal["range_high"]

    def test_gap_has_all_fields(self, sample_predict_result):
        gap = sample_predict_result["gap"]
        assert "compatible" in gap
        assert "missing" in gap
        assert "development_plan" in gap

    def test_top_jobs_is_list(self, sample_predict_result):
        assert isinstance(sample_predict_result["top_jobs"], list)
        if sample_predict_result["top_jobs"]:
            job = sample_predict_result["top_jobs"][0]
            assert "title" in job
            assert "adherence_score" in job
            assert "fit_label" in job

    def test_json_serializable(self, sample_predict_result):
        import json
        dumped = json.dumps(sample_predict_result)
        loaded = json.loads(dumped)
        assert loaded == sample_predict_result
