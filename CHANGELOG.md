# Changelog

## [0.5.0] — 2026-06-25

### Adicionado
- **Métricas dos modelos ML** — Script computa accuracy, F1, precision, recall, matriz de confusão (classificação) e RMSE, MAE, R² (regressão) e salva em `data/models/metrics.json`
- **Dashboard de métricas ML** — `monitor_dashboard.py` reescrito para exibir métricas de classificação e regressão (lê `metrics.json`, não depende da API)
- **Integração no `train_pipeline.py`** — Salvamento automático de `metrics.json` ao final do treino
- **Endpoint `/metrics`** — Middleware FastAPI coleta latência, erros e contagem por endpoint
- **Treino direto com dados reais** — Script de treino rápido usando `jobs_clean.parquet` (2978 jobs) + `resume_jd_train.parquet` (6241 pares)
  - Classificador SVM: F1=0.715, acurácia 0.714
  - Regressor GradientBoosting: RMSE=$39.132
  - Jobs matrix: (2978, 15000) pré-computada

### Corrigido
- **Coluna `job_description_text`** — Mapeamento no dataset HuggingFace (campo é `job_description_text`, não `job_description`)
- **Merge skills dataset** — Dataset `asaniczka/1-3m-linkedin-jobs-and-skills-2024` tem `job_link`+`job_skills`, não `job_title`+`skills`; merge via `linkedin_job_postings.csv`
- **Lookup `postings.csv`** — Dataset `arshkon/linkedin-job-postings` gera `postings.csv`, não `linkedin_job_postings.csv`
- **`build_skills_map` otimizado** — Match exato primeiro, fuzzy limitado a 500 amostras (evita 5h+ de processamento)
- **NaN na serialização JSON** — `np.float64` não era capturado por `isinstance(obj, float)`, causando `ValueError: Out of range float values are not JSON compliant`; ordem de verificação corrigida no `_clean_nan`
- **Pipeline lento** — Criado atalho para treino direto com dados já processados, sem reprocessar 123k vagas com NLTK
- **Label mapping no `train_pipeline.py`** — Agora mapeia "Good Fit"/"Potential Fit" → 1, "No Fit" → 0 (antes comparava com `== "Fit"` que nunca batia)

### Alterado
- `monitor_dashboard.py` — Agora exibe métricas de ML (classificação + regressão) em vez de métricas HTTP da API
- `train_pipeline.py` — Label mapping corrigido + salvamento automático de `metrics.json`
- `CHANGELOG.md`, `README.md` — Atualizados com estado atual
- `pyproject.toml` — Versão 0.5.0
- `todo.md` — Validação local concluída (API, metrics, dashboard)

## [0.4.0] — 2026-06-24

### Adicionado
- `Dockerfile` — Multi-stage build com Poetry para deploy da API
- `docker-compose.yml` — Serviços `api` (FastAPI) + `streamlit` (frontend) com rede compartilhada
- `.dockerignore` — Exclusão de dados, cache, logs e notebooks da imagem
- `.github/workflows/ci.yml` — CI com matrix Python 3.11/3.12, Ruff lint, pytest-cov, upload coverage
- `.github/workflows/deploy.yml` — Build e push Docker image para GitHub Container Registry
- `src/monitoring/metrics.py` — `MetricsCollector` singleton para tracking de requisições, latência e erros
- `src/monitoring/__init__.py` — Export do `metrics_collector`
- `src/api/server.py` — Middleware `@app.middleware("http")` para métricas + endpoint `GET /metrics`
- `src/app/monitor_dashboard.py` — Página Streamlit com dashboard de métricas em tempo real
- `tests/test_monitoring.py` — 6 testes unitários para MetricsCollector
- `tests/test_integration_real_data.py` — Teste de integração com dados sintéticos realistas (`@pytest.mark.slow`)

### Alterado
- `pyproject.toml` — Versão 0.4.0, marcador `slow` registrado, `monitor_dashboard.py` omitido do coverage
- `todo.md` — Status atualizado com os 4 novos módulos
- Testes existentes mantidos e todos passando (78 testes, 67% cobertura)

## [0.3.0] — 2026-06-23
...

## [0.2.0] — 2026-06-23
...

## [0.1.0] — 2026-06-23
...
