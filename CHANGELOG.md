# Changelog

## [0.4.0] — 2026-06-23

### Adicionado
- `.gitignore` — Segredos, dados, cache, logs, notebooks, IDE
- `.env.local` — Credenciais Kaggle (não versionado)
- `src/utils/config.py` — Carregamento automático de `.env.local` com python-dotenv

### Alterado
- `.env.example` — Adicionadas variáveis `KAGGLE_USERNAME` / `KAGGLE_KEY` com placeholders
- `src/pipeline/load_data.py` — Função `_configure_kaggle()` que cria `~/.kaggle/kaggle.json` a partir das env vars
- `tests/test_vectorizer.py` — Corrigido `test_min_df_filters_rare` com corpus maior (5 docs)
- `tests/test_pipeline_integration.py` — Corrigido smoke test: 10 jobs + 8 pairs para suportar `cv=2` com stratified folds
- `CHANGELOG.md`, `todo.md`, `README.md` — Sincronizados com estado atual

## [0.3.0] — 2026-06-23

### Adicionado
- `src/utils/logger.py` — Sistema centralizado de logging com RotatingFileHandler
- `src/api/predictor.py` — Classe `JobMatchPredictor` singleton com pre-load de modelos
- `src/api/server.py` — API REST FastAPI com endpoints `/predict`, `/health`, `/models/info`
- `tests/conftest.py` — Fixtures sintéticas e helpers para todos os módulos
- `tests/test_preprocess.py` — 7 testes para clean_text e tokenize
- `tests/test_load_data.py` — 4 testes para normalização de colunas com mocks
- `tests/test_compose_datasets.py` — 8 testes para salário, full_text, fuzzy match, quality check
- `tests/test_vectorizer.py` — 6 testes para fit, transform, load/save
- `tests/test_classifier.py` — 5 testes para predict e train_best
- `tests/test_recommender.py` — 4 testes para rank_jobs (top_k, ordenação, colunas, threshold)
- `tests/test_salary_model.py` — 4 testes para predict_salary_range (schema, range, ±15%)
- `tests/test_skills_analyzer.py` — 7 testes para extract_skills, analyze_gap, generate_plan
- `tests/test_predictor.py` — 6 testes de schema do JSON de saída
- `tests/test_pipeline_integration.py` — Smoke test completo (11 etapas)
- `.env.example` — Configuração de ambiente (API_URL, modo Streamlit, LOG_LEVEL)

### Alterado
- **Todos os 10 módulos**: `print()` substituído por `logging` estruturado
- `pyproject.toml` — Adicionadas dependências: fastapi, uvicorn, httpx, pytest, pytest-cov
- `pyproject.toml` — Configuração pytest e coverage (fail_under=55%)
- `src/app/streamlit_app.py` — Modo híbrido: tenta API REST, fallback direto se offline
- `train_pipeline.py` — Salva `jobs_matrix.npz` ao final para inicialização rápida do Predictor
- `README.md` — Seções "API REST" e "Testes"
- `requirements.txt` — Atualizado

## [0.2.0] — 2026-06-23

### Adicionado
- `pyproject.toml` com Poetry para gerenciamento de dependências
- `src/models/vectorizer.py` — TF-IDF com ngram_range=(1,2), min_df=3, max_df=0.85
- `src/models/classifier.py` — LogisticRegression, RandomForest, LinearSVC + cross-val F1
- `src/models/recommender.py` — Cosine similarity, Top-K ranking com threshold configurável
- `src/models/salary_model.py` — GradientBoosting + RandomForest para regressão salarial
- `src/skills/skills_analyzer.py` — Gap analysis com extração de skills e plano de desenvolvimento
- `src/app/streamlit_app.py` — Frontend completo com métricas, expanders e análise de skills
- `notebooks/01_eda.ipynb`, `03_modelo_classificador.ipynb`, `04_modelo_salario.ipynb`
- `train_pipeline.py` — Script unificado de treino completo

### Alterado
- `README.md` — Seção de setup com Poetry + pipx
- `requirements.txt` — Atualizado

## [0.1.0] — 2026-06-23

### Adicionado
- Estrutura de diretórios completa do projeto JobMatch AI
- `src/pipeline/preprocess.py`, `load_data.py`, `compose_datasets.py`
- `notebooks/02_pipeline_composicao.ipynb`
- `todo.md`, `README.md`, `CHANGELOG.md`
