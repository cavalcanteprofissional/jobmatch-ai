"""Testes de inicialização e schema da API REST."""

from fastapi.testclient import TestClient


def test_app_imports_successfully():
    """Verifica que o módulo server.py inicializa sem erros (ex: openapi_tags=None)."""
    from src.api.server import app
    assert app.title == "JobMatch AI API"
    assert app.openapi_tags is not None
    assert len(app.openapi_tags) == 4


def test_openapi_schema_is_valid():
    """Gera o schema OpenAPI para garantir que todas as rotas estão bem definidas."""
    from src.api.server import app
    schema = app.openapi()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/models/info" in schema["paths"]
    assert "/models/metrics" in schema["paths"]
    assert "/metrics" in schema["paths"]
    assert "/eval/classification" in schema["paths"]
    assert "/eval/regression" in schema["paths"]


def test_health_endpoint():
    """GET /health retorna 200 com status.
    O predictor não está carregado em testes, então espera 200 com models_loaded=False.
    """
    from src.api.server import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "models_loaded" in data


def test_models_metrics_endpoint_shape():
    """GET /models/metrics retorna métricas (200) ou 404 se arquivo não existe."""
    from src.api.server import app
    client = TestClient(app)
    resp = client.get("/models/metrics")
    if resp.status_code == 200:
        data = resp.json()
        assert "classification" in data
        assert "regression" in data
        assert "model_info" in data
    else:
        assert resp.status_code == 404
        assert "Execute" in resp.json()["detail"]


def test_eval_classification_shape():
    """GET /eval/classification retorna dados (200) ou 404."""
    from src.api.server import app
    client = TestClient(app)
    resp = client.get("/eval/classification")
    if resp.status_code == 200:
        data = resp.json()
        assert "data" in data
        assert "total" in data
    else:
        assert resp.status_code == 404
        assert "reload_eval" in resp.json()["detail"]


def test_eval_regression_shape():
    """GET /eval/regression retorna dados (200) ou 404."""
    from src.api.server import app
    client = TestClient(app)
    resp = client.get("/eval/regression")
    if resp.status_code == 200:
        data = resp.json()
        assert "data" in data
        assert "total" in data
    else:
        assert resp.status_code == 404
        assert "reload_eval" in resp.json()["detail"]


def test_predict_endpoint_schema():
    """POST /predict retorna erro estruturado (503 sem modelos, 500 erro interno)."""
    from src.api.server import app
    client = TestClient(app)
    resp = client.post("/predict", json={
        "resume_text": "Data Scientist with Python and Machine Learning",
        "top_k": 3,
        "fit_threshold": 40.0,
        "use_sbert": False,
        "use_cross_encoder": False,
    })
    # Sem modelos treinados → 503. Erro interno de modelo → 500.
    assert resp.status_code in (500, 503)
    assert "detail" in resp.json()
