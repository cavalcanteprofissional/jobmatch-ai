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
- **Frontend Híbrido**: Streamlit que consome a API REST com fallback direto para os modelos
- **Dashboard de Monitoramento**: Métricas em tempo real (latência, erros, requisições) + gráficos Altair do modelo ML (heatmap, histograma de scores, scatter salários)
- **Docker**: Deploy multi-stage com Docker Compose (FastAPI + Streamlit)
- **CI/CD**: GitHub Actions (lint + testes + coverage) com deploy automático para GHCR

## Arquitetura

```
jobmatch/
├── data/
│   ├── raw/              # Datasets originais (não versionado)
│   ├── processed/        # Dados limpos em Parquet (não versionado)
│   └── models/           # Modelos serializados joblib (não versionado)
├── src/
│   ├── pipeline/         # Download (Kaggle + HF), limpeza e composição
│   ├── models/           # TF-IDF, classificador, recommender, salary model
│   ├── skills/           # Gap analysis de skills
│   ├── api/              # FastAPI server + JobMatchPredictor singleton
│   ├── app/              # Frontend Streamlit híbrido
│   ├── monitoring/       # MetricsCollector singleton
│   └── utils/            # Logger (RotatingFileHandler) + Config (dotenv)
├── notebooks/            # Jupyter notebooks de EDA e experimentos
├── tests/                # ~95 testes (pytest + pytest-cov, fail_under=60%)
├── logs/                 # Logs rotativos (não versionado)
├── .env.example          # Template de variáveis de ambiente
├── .env.local            # Config local com credenciais (não versionado)
├── .gitignore
├── pyproject.toml        # Poetry: dependências e build
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

# 2. Instalar dependências do projeto
poetry install

# 3. Ativar ambiente virtual
poetry shell

# 4. Configurar credenciais Kaggle
#    Copie .env.example para .env.local e preencha suas credenciais:
#    KAGGLE_USERNAME=seu_usuario
#    KAGGLE_KEY=sua_chave_api
#    Ou edite diretamente .env.local (já incluso no .gitignore)

# 5. Executar pipeline de dados (download + composição)
python -c "from src.pipeline.compose_datasets import main; main()"

# 6. Treinar modelos
python train_pipeline.py

# 7. Rodar API REST (opcional — necessário para modo híbrido)
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# 8. Rodar frontend Streamlit (modo híbrido)
streamlit run src/app/streamlit_app.py

# 9. Dashboard de monitoramento (requer API rodando ou modelos pré-treinados)
streamlit run src/app/monitor_dashboard.py --server.port 8501

# 10. Re-treino rápido + dados de avaliação com nested CV (opcional, ~10 min)
python scripts/reload_eval.py

# 11. Docker (deploy completo)
docker compose up --build
```

## API REST

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/predict` | Predição completa (score, fit, salário, gap, top vagas) |
| `GET`  | `/health` | Health check do servidor |
| `GET`  | `/models/info` | Metadados dos modelos carregados |
| `GET`  | `/metrics` | Métricas de uso (latência, erros, requisições por endpoint) |

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

- Python 3.11–3.12
- scikit-learn (TF-IDF, RF, LR, SVM, ExtraTrees, KNN, GB, stacking, voting)
- XGBoost (classificação — melhor F1 72.33%)
- LightGBM (classificação + regressão)
- FastAPI + Uvicorn (REST API)
- Streamlit + Altair (frontend híbrido + dashboard de monitoramento com gráficos)
- Docker + Docker Compose (deploy)
- GitHub Actions (CI/CD)
- Parquet + pyarrow (armazenamento)
- rapidfuzz (fuzzy match)
- NLTK (lemmatização e stopwords)
- joblib (serialização)
- Sentence-Transformers (SBERT + cross-encoder, opcional — grupo `nlp`)
- Poetry (dependências)
- pytest + pytest-cov (testes, fail_under=60%)
- python-dotenv (config segura)

## Licença

Projeto acadêmico — Residência Prática TIC 44 (UFC) / AV03.
