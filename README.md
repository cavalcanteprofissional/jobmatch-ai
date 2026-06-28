# JobMatch AI

> Sistema de Machine Learning para matching inteligente entre currículos e vagas de emprego.
>
> **v0.10.0** · [Changelog](CHANGELOG.md) · [API](https://jobmatch-api-hpan.onrender.com/docs) · [Frontend](https://jobmatch-frontend-u6vt.onrender.com)

---

## Sumário

- [Funcionalidades](#funcionalidades)
- [Stack](#stack)
- [Resultados dos Modelos](#resultados-dos-modelos)
- [Arquitetura](#arquitetura)
- [Setup Local](#setup-local)
- [Deploy no Render](#deploy-no-render)
- [API REST](#api-rest)
- [Testes](#testes)
- [Datasets](#datasets)

---

## Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| **Score de Aderência** | Similaridade cosseno entre currículo e vagas via TF-IDF (ou SBERT opcional) |
| **Classificação Fit/No Fit** | 9 classificadores (XGBoost, LightGBM, RF, SVM, LR, SGD, KNN, DT, ExtraTrees) com seleção automática via nested CV |
| **Top-5 Vagas** | Ranking com suporte a re-ranking via cross-encoder |
| **Skills Gap Analysis** | 922 títulos mapeados, 20.6k skills, sinônimos e plano de desenvolvimento |
| **Estimativa Salarial** | Regressão (GradientBoosting, XGBoost, LightGBM, RF, ET, SGD, DT, KNN) |
| **Upload de Currículo** | PDF/DOCX com extração automática de texto |
| **Dark Mode** | Toggle 🌙/☀️ com persistência em localStorage |
| **Dashboard Monitor** | Métricas ML (matriz de confusão, nested CV, scatter, score distribution) + métricas da API |

## Stack

### Backend (Python)

| Camada | Tecnologia |
|--------|-----------|
| Runtime | Python 3.11–3.12 · Poetry |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost (opcional), Sentence-BERT (opcional) |
| API | FastAPI + Uvicorn |
| Dados | Parquet + pyarrow, rapidfuzz, NLTK |
| Testes | pytest + pytest-cov (fail_under=60%) |

### Frontend (TypeScript)

| Camada | Tecnologia |
|--------|-----------|
| Build | Vite |
| UI | React 18 + TypeScript |
| Estilo | Tailwind CSS 3 (dark mode via `class`) |
| Gráficos | Recharts (com CSS variables para tema) |
| Upload | react-dropzone |
| Router | React Router v6 |

### Infra

| Serviço | Plataforma | Tipo |
|---------|-----------|------|
| API FastAPI | Render | Web Service (Docker, free tier) |
| Frontend React | Render | Static Site (free tier) |
| CI/CD | GitHub Actions | Lint + testes + coverage |

## Resultados dos Modelos

| Tarefa | Métrica | Valor | Melhor Modelo |
|--------|---------|-------|---------------|
| Classificação | F1-Score | **72.33%** | XGBoost |
| Classificação | Acurácia | **71.82%** | XGBoost |
| Classificação (Nested CV) | F1 médio | **68.11%** (±0.55pp) | ExtraTrees |
| Regressão Salarial | RMSE | **$33.452** | VotingRegressor |
| Regressão Salarial | R² | **40.09%** | VotingRegressor |
| Regressão (Nested CV) | RMSE médio | **$39.651** (±$2.047) | GradientBoosting |

## Arquitetura

```
jobmatch/
├── data/
│   ├── raw/                  # Datasets originais (não versionado)
│   ├── processed/            # Dados limpos (Parquet + skills_map)
│   └── models/               # Modelos treinados (joblib)
├── frontend/                 # React + Vite + TypeScript
│   └── src/
│       ├── pages/            # JobMatch.tsx, Monitor.tsx
│       ├── components/       # Charts (Recharts)
│       ├── context/          # ThemeContext (dark mode)
│       └── services/         # api.ts (fetch com timeout + retry), models.ts
├── src/
│   ├── pipeline/             # Download, limpeza e composição de datasets
│   ├── models/               # TF-IDF, classificador, recommender, salary model
│   ├── skills/               # Skills gap analysis + sinônimos
│   ├── api/                  # FastAPI (predict, health, metrics, eval)
│   ├── monitoring/           # MetricsCollector
│   └── utils/                # Logger + Config
├── tests/                    # 95+ testes (pytest)
├── scripts/                  # startup.sh, reload_eval.py, build_skills_map.py
├── Dockerfile                # Build da API (multi-stage com Poetry)
├── render.yaml               # Blueprint da API no Render
└── deploy/                   # Dockerfile.frontend + nginx.conf
```

## Setup Local

### Pré-requisitos

- Python 3.11–3.12 com Poetry
- Node.js 18+
- Credenciais Kaggle (para pipeline de dados)

### Passos

```bash
# 1. Instalar dependências Python
poetry install && poetry shell

# 2. Pipeline de dados (download + composição)
python -c "from src.pipeline.compose_datasets import main; main()"

# 3. Treinar modelos
python train_pipeline.py

# 4. Re-treino com nested CV (opcional, ~10 min)
python scripts/reload_eval.py

# 5. Instalar dependências do frontend
cd frontend && npm install

# 6. Rodar frontend (http://localhost:5173)
npm run dev

# 7. Rodar API (http://localhost:8000/docs)
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### Testar frontend local com API na nuvem

```bash
cd frontend
cp .env.example .env.local
# Edite .env.local com a URL da API:
# VITE_API_URL=https://jobmatch-api-hpan.onrender.com
npm run dev
```

O frontend tenta a cloud primeiro com timeout de 15s e até 3 retentativas (backoff 2-4-8s). Se todas falharem, usa `/api` (proxy Vite para `localhost:8000`).

## Deploy no Render

### API (Web Service)

1. Dashboard Render → **New +** → **Web Service**
2. GitHub → `cavalcanteprofissional/jobmatch-ai` → branch `main`
3. **Name**: `jobmatch-api` · **Runtime**: `Docker` · **Health Check**: `/health` · **Free**
4. Create → aguardar build ~5 min

### Frontend (Static Site)

1. Dashboard Render → **New +** → **Static Site**
2. GitHub → `cavalcanteprofissional/jobmatch-ai` → branch `main`
3. **Name**: `jobmatch-frontend`
4. **Root Directory**: `frontend`
5. **Build Command**: `npm ci && npm run build`
6. **Publish Directory**: `dist`
7. **Environment Variable**: `VITE_API_URL = https://jobmatch-api-hpan.onrender.com`
8. **Create Static Site** → aguardar ~2 min

> ⚠️ **Importante:** Configure também uma **Rewrite Rule** no Dashboard do Static Site:
> Settings → Redirects/Rewrites → Add Rule — `/*` → `/index.html` → **Rewrite** → Status `200`
> (Isso garante SPA routing para refresh/F5 em qualquer rota como `/monitor`)

### Verificar

```bash
curl https://jobmatch-api-hpan.onrender.com/health
# {"status":"ok","models_loaded":true,"jobs_count":2978}
```

- **Frontend**: `https://jobmatch-frontend-u6vt.onrender.com`
- **API Docs**: `https://jobmatch-api-hpan.onrender.com/docs`

## API REST

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/predict` | Predição completa (score, fit, salário, gap, top vagas) |
| `GET`  | `/health` | Health check do servidor |
| `GET`  | `/models/info` | Metadados dos modelos carregados |
| `GET`  | `/models/metrics` | Métricas de avaliação dos modelos |
| `GET`  | `/eval/classification` | Dados de avaliação do classificador |
| `GET`  | `/eval/regression` | Dados de avaliação da regressão salarial |
| `GET`  | `/metrics` | Métricas de uso da API |

```bash
curl -X POST https://jobmatch-api-hpan.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Data Scientist with Python, SQL, and ML", "top_k": 5}'
```

## Testes

```bash
poetry run pytest tests/ -v --cov=src --ignore=tests/test_load_data.py
```

| Arquivo | Testes |
|---------|--------|
| `test_classifier.py` | 13 (inclui MLP, GaussianNB, nested CV) |
| `test_predictor.py` | 11 (inclui employability, SBERT flag) |
| `test_recommender.py` | 4 unit + 1 slow + cross-encoder |
| `test_salary_model.py` | 7 (inclui MLPRegressor, nested CV) |
| `test_vectorizer.py` | 6 unit + 1 slow |
| `test_skills_analyzer.py` | 7 |
| `test_preprocess.py` | 7 |
| `test_monitoring.py` | 6 |
| `test_compose_datasets.py` | 8 |
| `test_server.py` | 7 |
| `test_load_data.py` | 4 (requer datasets HuggingFace) |
| `test_integration_real_data.py` | 1 (slow) |
| `test_pipeline_integration.py` | 1 (smoke) |

## Datasets

| Dataset | Fonte | Uso |
|---------|-------|-----|
| LinkedIn Job Postings 2023–2024 | Kaggle (`arshkon/linkedin-job-postings`) | Base de vagas para ranking e regressão |
| Resume-JD-Match | HuggingFace (`cnamuangtoun/resume-job-description-fit`) | Treino do classificador binário |
| Job Skill Set | Kaggle (`asaniczka/1-3m-linkedin-jobs-and-skills-2024`) | Mapeamento cargo → skills |

---

**Projeto acadêmico** — Residência Prática TIC 44 (UFC) / AV03.
