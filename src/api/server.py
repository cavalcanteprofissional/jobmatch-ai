"""
JobMatch AI — FastAPI REST Server

Endpoints:
    POST /predict      — Executa pipeline completa e retorna JSON
    GET  /health       — Health check do servidor
    GET  /models/info  — Informações dos modelos carregados

Uso:
    uvicorn src.api.server:app --reload --port 8000
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.predictor import get_predictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="JobMatch AI API",
    description="API de matching inteligente entre currículos e vagas",
    version="0.3.0",
)


# ── Schemas ──────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    resume_text: str = Field(..., min_length=20, description="Texto do currículo/perfil")
    top_k: int = Field(default=5, ge=1, le=20, description="Número de vagas no ranking")
    fit_threshold: float = Field(
        default=40.0, ge=10.0, le=90.0,
        description="Score mínimo (%) para classificar como Fit",
    )


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    jobs_count: int = 0


class ModelsInfoResponse(BaseModel):
    vectorizer_features: int = 0
    classifier_type: str = ""
    regressor_type: str = ""
    jobs_count: int = 0
    skills_map_titles: int = 0


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Verifica se o servidor e os modelos estão carregados."""
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


@app.get("/models/info", response_model=ModelsInfoResponse)
async def models_info():
    """Retorna metadados dos modelos carregados."""
    try:
        predictor = get_predictor()
        skills_map_path = Path("data/processed/skills_map.json")
        skills_count = 0
        if skills_map_path.exists():
            import json
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


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Executa pipeline completa de matching.

    Retorna JSON com score, fit_label, top_jobs, salary_est e gap analysis.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(
            resume_text=req.resume_text,
            top_k=req.top_k,
            fit_threshold=req.fit_threshold,
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
