"""
JobMatch AI — FastAPI REST Server

Endpoints públicos para matching inteligente entre currículos e vagas.

## Match
- POST /predict           — Pipeline completa de predição (match, salário, skills, vagas)

## Modelos
- GET  /models/info       — Metadados dos modelos carregados (tipo, features, etc.)
- GET  /models/metrics    — Métricas de avaliação dos modelos (accuracy, F1, RMSE, nested CV)

## Dados de Avaliação
- GET  /eval/classification — Dados de avaliação do classificador (y_true, y_pred, y_prob)
- GET  /eval/regression     — Dados de avaliação da regressão salarial (y_true, y_pred)

## Servidor
- GET  /health              — Health check do servidor e modelos
- GET  /metrics             — Métricas de uso da API (latência, requisições, erros)

Uso:
    uvicorn src.api.server:app --reload --port 8000
    # Docs interativas: http://localhost:8000/docs
    # Schema OpenAPI:  http://localhost:8000/openapi.json
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.predictor import get_predictor
from src.monitoring.metrics import metrics_collector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MODELS_DIR = Path("data/models")

# ====================================================================
# App
# ====================================================================

app = FastAPI(
    title="JobMatch AI API",
    description=(
        "API de matching inteligente entre currículos e vagas de emprego. "
        "Fornece predição de fit, estimativa salarial, análise de skills "
        "e ranking de vagas com suporte a TF-IDF e Sentence-BERT.\n\n"
        "📘 **Documentação completa**: https://github.com/cavalcanteprofissional/jobmatch-ai\n"
        "🎯 **Frontend**: http://localhost (produção) | http://localhost:5173 (dev)"
    ),
    version="0.9.0",
    contact={
        "name": "JobMatch AI",
        "url": "https://github.com/cavalcanteprofissional/jobmatch-ai",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

# ====================================================================
# Metrics middleware
# ====================================================================


@app.middleware("http")
async def metrics_middleware(request, call_next):
    import time

    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    endpoint = f"{request.method} {request.url.path}"
    metrics_collector.record_request(endpoint, elapsed_ms, response.status_code)
    return response


# ====================================================================
# Schemas
# ====================================================================


# ── Match ──


class PredictRequest(BaseModel):
    """Payload para predição de compatibilidade."""

    resume_text: str = Field(
        ...,
        min_length=20,
        max_length=100_000,
        description="Texto do currículo ou perfil profissional (mín. 20 caracteres)",
        example="Data Scientist with 5 years of experience in Python, SQL, Machine Learning, and TensorFlow. "
        "Worked on NLP and computer vision projects. Proficient in AWS, Docker, and Git.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de vagas no ranking de recomendação",
        example=5,
    )
    fit_threshold: float = Field(
        default=40.0,
        ge=10.0,
        le=90.0,
        description="Score mínimo (%) para classificar uma vaga como 'Fit'",
        example=40.0,
    )
    use_sbert: bool = Field(
        default=False,
        description="Usar Sentence-BERT (embedding semântico) em vez de TF-IDF. Requer modelos SBERT pré-treinados.",
        example=False,
    )
    use_cross_encoder: bool = Field(
        default=False,
        description="Re-ranking das vagas com Cross-Encoder para maior precisão semântica (~2s extra por requisição)",
        example=False,
    )


class PredictResponse(BaseModel):
    """Resposta completa da predição de compatibilidade."""

    score_pct: float = Field(..., description="Probabilidade de Fit (%)", example=72.3)
    fit_label: str = Field(..., description="Classificação: 'Fit' ou 'No Fit'", example="Fit")
    avg_adherence: float = Field(..., description="Score médio de aderência do top-k (%)", example=65.2)
    fit_count: int = Field(..., description="Quantidade de vagas classificadas como Fit no top-k", example=3)
    top_k: int = Field(..., description="Número de vagas solicitado", example=5)
    employability_score: float = Field(..., description="Score de empregabilidade (%) baseado nas skills compatíveis", example=78.5)
    salary_est: dict[str, Any] = Field(
        ...,
        description="Estimativa salarial para a vaga #1",
        example={"estimated_annual_usd": 120000, "range_low": 102000, "range_high": 138000},
    )
    gap: dict[str, Any] = Field(
        ...,
        description="Análise de gap de skills (compatible, missing, development_plan)",
    )
    top_jobs: list[dict[str, Any]] = Field(
        ...,
        description="Ranking das vagas mais compatíveis",
        example=[{"title": "Data Scientist", "company_name": "Tech Corp", "adherence_score": 85.0}],
    )


# ── Server ──


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status do servidor: 'ok' ou mensagem de erro", example="ok")
    models_loaded: bool = Field(..., description="Indica se os modelos foram carregados com sucesso", example=True)
    jobs_count: int = Field(default=0, description="Número de vagas carregadas no banco", example=2978)


class ModelsInfoResponse(BaseModel):
    vectorizer_features: int = Field(default=0, description="Número de features do TF-IDF", example=15000)
    classifier_type: str = Field(default="", description="Tipo do classificador em uso", example="ExtraTreesClassifier")
    regressor_type: str = Field(default="", description="Tipo do regressor salarial em uso", example="GradientBoostingRegressor")
    jobs_count: int = Field(default=0, description="Total de vagas na base", example=2978)
    skills_map_titles: int = Field(default=0, description="Total de títulos mapeados no skills_map.json", example=922)


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Mensagem detalhada do erro")


# ====================================================================
# Tags para organização da documentação OpenAPI
# ====================================================================

TAG_MATCH = "1. Match"
TAG_MODELS = "2. Modelos"
TAG_EVAL = "3. Dados de Avaliação"
TAG_SERVER = "4. Servidor"
TAGS_META = [
    {"name": TAG_MATCH, "description": "Pipeline completa de matching entre currículo e vagas"},
    {"name": TAG_MODELS, "description": "Informações e métricas dos modelos de ML treinados"},
    {"name": TAG_EVAL, "description": "Dados de avaliação (holdout) para análise do desempenho dos modelos"},
    {"name": TAG_SERVER, "description": "Health check e métricas de uso do servidor"},
]

app.openapi_tags = TAGS_META


# ====================================================================
# Endpoints — Match
# ====================================================================


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={503: {"model": ErrorResponse, "description": "Modelos não treinados"}},
    tags=[TAG_MATCH],
    summary="Predição completa de compatibilidade",
)
async def predict(req: PredictRequest):
    """
    Executa a pipeline completa de matching entre currículo e vagas.

    **Fluxo:**
    1. Limpeza do texto do currículo (NLTK)
    2. Vetorização (TF-IDF ou Sentence-BERT)
    3. Classificação Fit/No Fit (modelo treinado via nested CV)
    4. Ranking das top-k vagas mais compatíveis
    5. Re-ranking opcional com Cross-Encoder
    6. Estimativa salarial para a melhor vaga
    7. Análise de skills (gap analysis + empregabilidade)
    8. Plano de desenvolvimento sugerido

    **Exemplo:**
    ```json
    {
      "resume_text": "Data Scientist with Python, SQL, and ML",
      "top_k": 5,
      "fit_threshold": 40.0,
      "use_sbert": false,
      "use_cross_encoder": false
    }
    ```
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(
            resume_text=req.resume_text,
            top_k=req.top_k,
            fit_threshold=req.fit_threshold,
            use_sbert=req.use_sbert,
            use_cross_encoder=req.use_cross_encoder,
        )
        logger.info(
            "POST /predict: top_k=%s, score=%.1f%%, fit=%s",
            req.top_k, result["score_pct"], result["fit_label"],
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Modelos não encontrados. Execute train_pipeline.py primeiro. {e}",
        )
    except Exception as e:
        logger.error("Erro no predict: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# Endpoints — Modelos
# ====================================================================


@app.get(
    "/models/info",
    response_model=ModelsInfoResponse,
    tags=[TAG_MODELS],
    summary="Metadados dos modelos carregados",
)
async def models_info():
    """
    Retorna informações sobre os modelos atualmente carregados em memória:
    - Tipo do classificador e regressor
    - Número de features do TF-IDF
    - Total de vagas e títulos de skills mapeados
    """
    try:
        predictor = get_predictor()
        skills_map_path = Path("data/processed/skills_map.json")
        skills_count = 0
        if skills_map_path.exists():
            with open(skills_map_path) as f:
                skills_count = len(json.load(f))

        return ModelsInfoResponse(
            vectorizer_features=len(predictor.vec.vocabulary_),
            classifier_type=type(predictor.clf).__name__,
            regressor_type=type(predictor.sal).__name__,
            jobs_count=len(predictor.jobs),
            skills_map_titles=skills_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/models/metrics",
    tags=[TAG_MODELS],
    summary="Métricas de avaliação dos modelos",
    responses={
        200: {"description": "Métricas de classificação, regressão e nested CV"},
        404: {"description": "metrics.json não encontrado (execute reload_eval.py primeiro)"},
    },
)
async def models_metrics():
    """
    Retorna as métricas de avaliação dos modelos salvas em `data/models/metrics.json`.

    **Estrutura do response:**
    ```json
    {
      "classification": {
        "model_type": "ExtraTreesClassifier",
        "accuracy": 0.7182,
        "f1_score": 0.7233,
        "precision": 0.7055,
        "recall": 0.7419,
        "test_samples": 1249,
        "confusion_matrix": [[424, 205], [159, 461]],
        "nested_cv": { "scores": [0.6866, 0.6756], "mean": 0.6811, "std": 0.0055 },
        "best_params": { "n_estimators": 200, "max_depth": 10 },
        "best_candidate": "extra_trees",
        "sbert": { ... }
      },
      "regression": { ... },
      "model_info": {
        "vectorizer": "tfidf",
        "vectorizer_features": 15000,
        "total_jobs": 2978,
        "jobs_with_salary": 891,
        "training_pairs": 6241
      }
    }
    ```
    """
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.json não encontrado. Execute `python scripts/reload_eval.py` para gerar as métricas.",
        )
    with open(path) as f:
        return json.load(f)


# ====================================================================
# Endpoints — Dados de Avaliação
# ====================================================================


@app.get(
    "/eval/classification",
    tags=[TAG_EVAL],
    summary="Dados de avaliação do classificador (holdout)",
    responses={
        200: {"description": "Lista de predições com y_true, y_pred, y_prob e labels"},
        404: {"description": "eval_clf.parquet não encontrado"},
    },
)
async def eval_classification():
    """
    Retorna os dados de avaliação do classificador sobre o conjunto de holdout.

    **Campos:**
    - `y_true`: valor real (1 = Fit, 0 = No Fit)
    - `y_pred`: valor predito pelo modelo
    - `y_prob`: probabilidade estimada (0-1)
    - `y_true_label`: label real ("Fit" / "No Fit")
    - `y_pred_label`: label predito ("Fit" / "No Fit")

    **Uso:** Útil para gerar curvas ROC, matrices de confusão customizadas
    ou análises de erro no frontend.
    """
    path = MODELS_DIR / "eval_clf.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="eval_clf.parquet não encontrado. Execute reload_eval.py primeiro.")
    df = pd.read_parquet(path)
    return {"data": json.loads(df.to_json(orient="records")), "total": len(df)}


@app.get(
    "/eval/regression",
    tags=[TAG_EVAL],
    summary="Dados de avaliação do regressor salarial (holdout)",
    responses={
        200: {"description": "Lista de predições com y_true (real) e y_pred (previsto)"},
        404: {"description": "eval_reg.parquet não encontrado"},
    },
)
async def eval_regression():
    """
    Retorna os dados de avaliação do regressor salarial.

    **Campos:**
    - `y_true`: salário anual real (USD)
    - `y_pred`: salário anual predito pelo modelo

    **Uso:** Para gerar scatter plots, histogramas de resíduos
    e métricas customizadas de regressão.
    """
    path = MODELS_DIR / "eval_reg.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="eval_reg.parquet não encontrado. Execute reload_eval.py primeiro.")
    df = pd.read_parquet(path)
    return {"data": json.loads(df.to_json(orient="records")), "total": len(df)}


# ====================================================================
# Endpoints — Servidor
# ====================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=[TAG_SERVER],
    summary="Health check do servidor",
)
async def health():
    """Verifica se o servidor e os modelos estão carregados e operacionais."""
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="ok",
            models_loaded=True,
            jobs_count=len(predictor.jobs),
        )
    except Exception as e:
        logger.error("Health check falhou: %s", e)
        return HealthResponse(status=f"error: {e}", models_loaded=False)


@app.get(
    "/metrics",
    tags=[TAG_SERVER],
    summary="Métricas de uso da API",
    responses={
        200: {
            "description": "Métricas de requisições, latência e erros por endpoint",
        },
    },
)
async def metrics():
    """
    Retorna métricas de uso do servidor desde o startup:
    - Requisições totais e por endpoint
    - Latência média, mínima, máxima e P99 por endpoint
    - Taxa de erro geral
    - Uptime do servidor
    """
    return metrics_collector.get_metrics()
