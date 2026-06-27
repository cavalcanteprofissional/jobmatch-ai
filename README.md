# JobMatch AI

Sistema de Machine Learning para matching inteligente entre currículos e vagas de emprego.

## Funcionalidades

- **Score de Aderência**: Similaridade cosseno entre currículo e vagas via TF-IDF
- **Classificação Fit/No Fit**: Classificador binário (ExtraTrees / XGBoost / LightGBM / RF / SVM / LR / SGD / KNN / DT) com seleção automática via nested CV + tuning
- **Top-5 Vagas**: Ranking das vagas mais compatíveis com o perfil (com suporte a re-ranking via cross-encoder)
- **Skills Gap Analysis**: Identifica skills faltantes (922 títulos mapeados, 20.6k skills) e sugere plano de desenvolvimento com sinônimos
- **Estimativa Salarial**: Regressão (GradientBoosting / XGBoost / LightGBM / RF / ET / SGD / DT / KNN) para prever faixa salarial por cargo
- **Embeddings Semânticos (SBERT)**: Opcional — Sentence-BERT `all-MiniLM-L6-v2` para classificação com MLP/GaussianNB
- **API REST**: Endpoints FastAPI para predição, health check, info dos modelos e métricas
- **Frontend React + Vite**: SPA moderna com TypeScript, Tailwind CSS e Recharts
- **Página Match**: Upload de PDF/DOCX, formulário de análise, resultados com cards, skills e plano de desenvolvimento
- **Página Monitor**: Dashboard com métricas do modelo ML (matriz de confusão, nested CV, scatter) e métricas da API (latência, requisições, erros)
- **Docker**: Deploy multi-stage com Docker Compose (FastAPI + Nginx + React)
- **CI/CD**: GitHub Actions (lint + testes + coverage) com deploy automático para GHCR

## Arquitetura

```
jobmatch/
├── data/
│   ├── raw/              # Datasets originais (não versionado)
│   ├── processed/        # Dados limpos em Parquet (não versionado)
│   └── models/           # Modelos serializados joblib (não versionado)
├── frontend/             # React + Vite + TypeScript + Tailwind
│   ├── src/
│   │   ├── pages/        # JobMatch.tsx, Monitor.tsx
│   │   ├── components/   # Charts (Recharts), MetricCard, etc.
│   │   └── services/     # api.ts, models.ts
│   └── package.json
├── src/
│   ├── pipeline/         # Download (Kaggle + HF), limpeza e composição
│   ├── models/           # TF-IDF, classificador, recommender, salary model
│   ├── skills/           # Gap analysis de skills
│   ├── api/              # FastAPI server + JobMatchPredictor singleton
│   ├── monitoring/       # MetricsCollector singleton
│   └── utils/            # Logger (RotatingFileHandler) + Config (dotenv)
├── tests/                # ~95 testes (pytest + pytest-cov, fail_under=60%)
├── logs/                 # Logs rotativos (não versionado)
├── deploy/               # Deploy: Dockerfile.frontend + nginx.conf
├── render.yaml           # Render Blueprint (API Web Service)
├── Dockerfile            # Build da API FastAPI
├── docker-compose.yml    # api + frontend
├── scripts/
│   ├── startup.sh        # Inicialização do container (só uvicorn)
│   ├── reload_eval.py    # Re-treino com nested CV (~10 min)
│   └── build_skills_map.py
└── train_pipeline.py     # Script unificado de treino
```

## Datasets

| Dataset | Fonte | Uso |
|---------|-------|-----|
| LinkedIn Job Postings 2023–2024 | Kaggle (`arshkon/linkedin-job-postings`) | Base de vagas para ranking e regressão |
| Resume-JD-Match | HuggingFace (`cnamuangtoun/resume-job-description-fit`) | Treino do classificador binário |
| Job Skill Set | Kaggle (`asaniczka/1-3m-linkedin-jobs-and-skills-2024`) | Mapeamento cargo → skills |

## Setup

```bash
# 1. Instalar Poetry via pipx
pipx install poetry

# 2. Instalar dependências do projeto (Python)
poetry install

# 3. Ativar ambiente virtual
poetry shell

# 4. Configurar credenciais Kaggle
#    Copie .env.example para .env.local e preencha suas credenciais

# 5. Executar pipeline de dados (download + composição)
python -c "from src.pipeline.compose_datasets import main; main()"

# 6. Treinar modelos
python train_pipeline.py

# 7. Instalar dependências do frontend
cd frontend && npm install

# 8. Rodar frontend em modo dev (porta 5173, proxy /api → localhost:8000)
npm run dev

# 9. Rodar API REST
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# 10. Re-treino rápido + dados de avaliação com nested CV (opcional, ~10 min)
python scripts/reload_eval.py

# 11. Docker (deploy completo)
docker compose up --build
```

Acesse:
- **Match**: http://localhost:5173 (dev) ou http://localhost (produção)
- **API**: http://localhost:8000/docs

### Testar frontend local com API na nuvem

Para rodar o frontend React apontando para a API já deployada no Render:

```bash
cd frontend
cp .env.example .env.local
# Edite .env.local com a URL da sua API:
# VITE_API_URL=https://jobmatch-api-hpan.onrender.com
npm run dev
```

O frontend tentará a cloud primeiro; se falhar (container idle), cairá para `http://localhost:8000`.
Sem `VITE_API_URL`, usa o proxy Vite (`/api` → `localhost:8000`).

---

## Deploy Free Tier — Render.com

> **Problema:** O free tier do Render tem 512 MB de RAM, insuficiente para
> executar a pipeline de dados (123k vagas + NLTK + pandas) e o nested CV.
>
> **Solução:** Modelos treinados são embutidos na imagem Docker (~14 MB).
> A imagem é construída com `data/models/` e `data/processed/` inclusos,
> eliminando a necessidade de treinar no container.

### Pré-requisitos

```bash
# 1. Treinar modelos LOCALMENTE (já feito se você rodou o setup)
poetry run python train_pipeline.py
poetry run python scripts/reload_eval.py

# 2. Adicionar data/ ao git e fazer push
git add data/
git commit -m "feat: trained models for Docker deploy"
git push origin feat/frontend-react
```

### Deploy da API (Web Service)

1. Dashboard Render → **New +** → **Web Service**
2. GitHub → `cavalcanteprofissional/jobmatch-ai` → branch `feat/frontend-react`
3. **Name**: `jobmatch-api`
4. **Runtime**: `Docker`
5. **Health Check Path**: `/health`
6. **Instance Type**: `Free`
7. **Environment Variables**: *(nenhuma — configuradas no Dockerfile)*
8. **Create Web Service** — aguardar build ~5 min

### Deploy do Frontend (Static Site)

1. Dashboard Render → **New +** → **Static Site**
2. GitHub → `cavalcanteprofissional/jobmatch-ai` → branch `feat/frontend-react`
3. **Name**: `jobmatch-frontend`
4. **Root Directory**: `frontend`
5. **Build Command**: `npm ci && npm run build`
6. **Publish Directory**: `dist`
7. **Environment Variable**: `VITE_API_URL = https://jobmatch-api.onrender.com`
8. **Create Static Site** — aguardar ~2 min

### Resultado

```bash
curl https://jobmatch-api.onrender.com/health
# {"status":"ok","models_loaded":true,"jobs_count":2978}
```

Frontend em `https://jobmatch-frontend.onrender.com`, API em `https://jobmatch-api.onrender.com/docs`.

### Fallback automático

O frontend tenta a cloud primeiro (`VITE_API_URL`); se falhar
(ex.: container em idle/sleep), cai para `http://localhost:8000`
(API rodando local). Em desenvolvimento (sem `VITE_API_URL`),
usa `/api` com proxy do Vite.

---

## API REST

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/predict` | Predição completa (score, fit, salário, gap, top vagas) |
| `GET`  | `/health` | Health check do servidor |
| `GET`  | `/models/info` | Metadados dos modelos carregados |
| `GET`  | `/models/metrics` | Métricas de avaliação dos modelos (accuracy, F1, RMSE, nested CV) |
| `GET`  | `/eval/classification` | Dados de avaliação do classificador (y_true, y_pred, y_prob) |
| `GET`  | `/eval/regression` | Dados de avaliação da regressão salarial (y_true, y_pred) |
| `GET`  | `/metrics` | Métricas de uso da API (latência, erros, requisições por endpoint) |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Data Scientist with Python, SQL, and ML", "top_k": 5}'
```

## Testes

```bash
poetry run pytest tests/ -v --cov=src --ignore=tests/test_load_data.py
# 85 passed (rápidos) · 12 deselected (slow / SBERT)
```

| Arquivo | Testes |
|---------|--------|
| `test_preprocess.py` | 7 |
| `test_load_data.py` | 4 (requer datasets HuggingFace) |
| `test_compose_datasets.py` | 8 |
| `test_vectorizer.py` | 6 unit + 1 slow |
| `test_classifier.py` | 13 (inclui MLP, GaussianNB, nested CV) |
| `test_recommender.py` | 4 unit + 1 slow + cross-encoder |
| `test_salary_model.py` | 7 (inclui MLPRegressor, nested CV) |
| `test_skills_analyzer.py` | 7 |
| `test_predictor.py` | 11 (inclui employability, SBERT flag) |
| `test_monitoring.py` | 6 |
| `test_pipeline_integration.py` | 1 (smoke: 11 etapas) |
| `test_integration_real_data.py` | 1 (slow: dados reais) |

## Métricas

| Tarefa | Métrica | Valor | Melhor Modelo |
|--------|---------|-------|---------------|
| Classificação | F1-Score | **72.33%** | XGBoost |
| Classificação | Acurácia | **71.82%** | XGBoost |
| Classificação (Nested CV 2×2) | F1 médio | **68.11%** (±0.55pp) | ExtraTrees |
| Regressão Salarial | RMSE | **$33.452** | VotingRegressor |
| Regressão Salarial | R² | **40.09%** | VotingRegressor |
| Regressão (Nested CV 3×2) | RMSE médio | **$39.651** (±$2.047) | GradientBoosting |

## Stack

### Backend (Python)
- Python 3.11–3.12
- scikit-learn (TF-IDF, RF, LR, SVM, ExtraTrees, KNN, GB, stacking, voting)
- XGBoost (classificação — melhor F1 72.33%)
- LightGBM (classificação + regressão)
- FastAPI + Uvicorn (REST API)
- Parquet + pyarrow (armazenamento)
- rapidfuzz (fuzzy match)
- NLTK (lemmatização e stopwords)
- joblib (serialização)
- Sentence-Transformers (SBERT + cross-encoder, opcional — grupo `nlp`)
- Poetry (dependências)
- pytest + pytest-cov (testes, fail_under=60%)

### Frontend (TypeScript)
- Vite (build)
- React 18 + TypeScript
- Tailwind CSS (estilo)
- Recharts (gráficos)
- react-dropzone (upload)
- React Router v6

### Infra
- Docker + Docker Compose (api + frontend com Nginx)
- GitHub Actions (CI/CD)

## Licença

Projeto acadêmico — Residência Prática TIC 44 (UFC) / AV03.
