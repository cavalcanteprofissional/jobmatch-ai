# JobMatch AI — Plano de Implementação

## Status Geral

| Módulo | Status |
|--------|--------|
| Estrutura de diretórios | ✅ Concluído |
| Pipeline de dados (preprocess, load, compose) | ✅ Concluído |
| Modelos (vectorizer, classifier, recommender, salary) | ✅ Concluído |
| Skills analyzer | ✅ Concluído |
| Frontend Streamlit (híbrido API/direto) | ✅ Concluído |
| API REST FastAPI (predict, health, models/info) | ✅ Concluído |
| Logging estruturado (RotatingFileHandler) | ✅ Concluído |
| Testes (78 testes, 67% cobertura) | ✅ Concluído |
| Config Kaggle (.env.local, .gitignore, config.py) | ✅ Concluído |
| Notebooks (EDA, pipeline, classificador, salário) | ✅ Concluído |
| `train_pipeline.py` | ✅ Concluído |

## Pendentes (próximas versões)

### Etapa 1 — Deploy da API (Docker)
- [x] Dockerfile (multi-stage)
- [x] docker-compose.yml (api + streamlit)
- [x] .dockerignore

### Etapa 2 — CI/CD (GitHub Actions)
- [x] CI workflow (testes + coverage)
- [x] Deploy workflow (Docker image)

### Etapa 3 — Dashboard de Monitoramento
- [x] Middleware de métricas (FastAPI)
- [x] Endpoint GET /metrics
- [x] Página Streamlit de monitoramento

### Etapa 4 — Testes de Integração com Dados Reais
- [x] Testes @pytest.mark.slow com dados reais
- [x] Integração na pipeline principal

### Etapa 5 — Validação / Teste Local do Dashboard
- [x] Corrigir script de treino (caminho resume_jd_train.parquet + labels 3-classes)
- [x] Treinar classificador Fit/No Fit + regressor de salário
- [x] Iniciar API FastAPI (uvicorn) — http://localhost:8001
- [x] Verificar endpoint /predict com curl — JSON válido, NaN tratados como null
- [x] Verificar endpoint /metrics — uptime, total_requests, latência por endpoint
- [x] Iniciar dashboard Streamlit de monitoramento — http://localhost:8501

### Etapa 6 — Dashboard Unificado (Modelo ML + API)
- [x] `monitor_dashboard.py` reescrito com `st.tabs()`: aba "Modelo ML" (métricas de classificação e regressão) + aba "API" (requisições, latência, erros via GET /metrics)
- [x] `train_pipeline.py` corrigido (label mapping "Good Fit"/"Potential Fit" → 1) e passou a salvar `metrics.json` automaticamente ao final do treino

## Decisões Técnicas

| Decisão | Escolha | Motivo |
|---------|---------|--------|
| Vetorização | TF-IDF (sklearn) | Fase inicial, sem LLMs |
| Serialização | joblib | Padrão sklearn |
| Formato dados | Parquet (pyarrow) | Eficiente para colunas textuais |
| Fuzzy match | rapidfuzz | Mais rápido que fuzzywuzzy |
| Frontend | Streamlit + FastAPI | Híbrido: REST com fallback direto |
| Python | >=3.11,<3.13 | Compatibilidade com dependências |
| Gerenciamento | Poetry | Reprodutibilidade de ambiente |
| Testes | pytest + pytest-cov | 78 testes, fail_under=55% |
| Credenciais | python-dotenv + .env.local | Segurança, sem versionar segredos |
