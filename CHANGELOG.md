# Changelog

## [0.9.1] — 2026-06-27

### Corrigido
- **`TypeError: 'NoneType' object is not iterable` na inicialização da API** — `app.openapi_tags` é `None` por padrão no FastAPI. O loop de deduplicação em `server.py:192` tentava iterar sobre `None` ao usar `getattr(app, "openapi_tags", [])` (o atributo existe, mas é `None`). Substituído por `app.openapi_tags = TAGS_META` direto.
- **`render.yaml` com `type: static` inválido** — Render Blueprint não suporta `type: static`. Seção do frontend removida do `render.yaml`. Frontend será criado como Static Site manualmente no Dashboard.
- **Env vars movidas para o Dockerfile** — `PYTHONUNBUFFERED` e `PYTHONDONTWRITEBYTECODE` agora são `ENV` no `Dockerfile`, eliminando necessidade de configurá-las no Render Dashboard ou no `docker-compose.yml`.

### Alterado (estratégia de deploy para free tier)
- **`scripts/startup.sh` simplificado** — Remove pipeline de treino (`compose_datasets` + `train_pipeline` + `reload_eval`). Agora só verifica se modelos existem e inicia uvicorn. Modelos são embutidos na imagem Docker (~14 MB).
- **`.dockerignore` reescrito** — Agora **inclui** `data/models/` e `data/processed/` na imagem Docker (antes excluía tudo de `data/`).
- **`.gitignore` simplificado** — Remove exclusão de `*.pkl`, `*.parquet`, `*.npz` — permite versionar modelos treinados.
- **`api.ts` com fallback cloud → local** — Em produção (Static Site com `VITE_API_URL`), tenta cloud primeiro; se falhar, cai para `http://localhost:8000`. Em dev, usa `/api` (Vite proxy).

### Motivação
- Free tier do Render tem 512 MB de RAM — insuficiente para rodar pipeline de dados (123k vagas + NLTK + pandas + lematização).
- Solução: treinar local, commitar `data/` no git, embutir na imagem Docker.

## [0.9.0] — 2026-06-26

### Adicionado
- **Frontend React + Vite + TypeScript** — Substitui os dois Streamlits por um SPA moderno.
  - `frontend/` — Scaffold completo com Vite, React 18, TypeScript, Tailwind CSS
  - Página **JobMatch**: upload PDF/DOCX com react-dropzone, formulário de análise, cards de métricas, vagas expansíveis, skills (compatíveis/faltantes), plano de desenvolvimento
  - Página **Monitor**: abas Modelo ML (matriz de confusão, métricas, nested CV, scatter, comparação TF-IDF vs SBERT) e API (requisições, erros, latência), gráficos com Recharts
  - `Dockerfile.frontend` — Build multi-stage (node → nginx)
  - `nginx.conf` — Proxy reverso: `/api/` → FastAPI, `/` → SPA React

### Removido
- `src/app/streamlit_app.py` — Substituído pela página JobMatch React
- `src/app/monitor_dashboard.py` — Substituído pela página Monitor React
- Serviço `streamlit` do `docker-compose.yml` — Substituído por `frontend` (nginx porta 80)

### Alterado
- `docker-compose.yml` — Serviço `streamlit` removido, serviço `frontend` adicionado
- `README.md` — Setup, arquitetura e stack atualizados para React
- `todo.md` — Fase 6 documentada (migração frontend)
- `.dockerignore` — Adicionado frontend/node_modules e frontend/dist

## [0.8.0] — 2026-06-26

### Corrigido
- **Travamento do `reload_eval.py` durante nested CV** — Causa: `StackingClassifier`/`StackingRegressor` dentro do nested CV criava treino exponencial (re-treinava 5-7 modelos internamente por fold). Adicionado `MLP` e `GaussianNB` que densificam a matriz sparse, agravando o problema. `n_jobs=-1` saturava a RAM.
  - `classifier.py`: Criado `NESTED_CV_CANDIDATES` — apenas modelos individuais compatíveis com sparse (exclui stacking, voting, MLP, GaussianNB). `N_JOBS=2` para evitar thrashing.
  - `salary_model.py`: Mesma estrutura — `NESTED_CV_CANDIDATES` sem stacking, voting, MLP. `N_JOBS=2`.
  - `reload_eval.py`: Agora usa `NESTED_CV_CANDIDATES` no nested CV. Stacking/Voting/MLP só entram no treino final via `ALL_CANDIDATES`.
- **`SentenceBertVectorizer.fit_transform` inexistente** — Script chamava `fit_transform()` que não existe na classe; corrigido para `fit()` + `transform()` separados.
- **Timestamps adicionados** — Cada etapa do `reload_eval.py` agora exibe `[HH:MM:SS]` para facilitar diagnóstico.

### Alterado
- `scripts/reload_eval.py` — Adicionados timestamps, correção SBERT fit/transform
- `src/models/classifier.py` — `NESTED_CV_CANDIDATES`, `N_JOBS=2`, `_make_candidate` aceita `n_jobs` opcional
- `src/models/salary_model.py` — `NESTED_CV_CANDIDATES`, `N_JOBS=2`
- `todo.md` — Diagnóstico de performance documentado, Fase 5 marcada como concluída

### Resultados (pós-correção)
- **`reload_eval.py` executou em ~10 min sem travamentos** (vs >30 min ou travamento antes)
- **Classificação**: ExtraTrees venceu nested CV — F1=68.11% (±0.55pp)
- **Regressão**: GradientBoosting venceu nested CV — RMSE=$39.651 (±$2.047)
- **85/85 testes unitários passando** (12 slow/SBERT desabilitados sem dependência NLP)

## [0.7.0] — 2026-06-25

### Adicionado
- **Novos algoritmos de classificação** — XGBoost, LightGBM, ExtraTrees, SGD, KNN, DecisionTree como candidatos no nested CV
- **Novos algoritmos de regressão** — LightGBM, ExtraTrees, SGD, KNN, DecisionTree como candidatos no nested CV
- **Ensembles expandidos** — Stacking e Voting agora incluem ExtraTrees e LightGBM como base estimators
- **Sinonímia de skills** — 40+ grupos de sinônimos (ex: "tf"↔"tensorflow", "k8s"↔"kubernetes", "ml"↔"machine learning")
- **`_RESOURCES` expandido** — De 20 para 70+ skills com cursos e tempo estimado
- **`skills_map.json` populado** — 922 títulos mapeados, 20.607 skills, 46.5% das vagas com required_skills
- **`scripts/build_skills_map.py`** — Script dedicado para construção do skills map com merge de datasets
- **Dependências opcionais** — LightGBM, CatBoost, spaCy, sentence-transformers em grupos opcionais

### Alterado
- `src/models/classifier.py` — 8 candidatos (LR, RF, SVM, XGB, ET, LGBM, stacking, voting); nested CV com inner_cv=3, n_iter=15; try/except em treino para robustez
- `src/models/salary_model.py` — 7 regressores (GB, RF, XGB, ET, LGBM, stacking, voting); mesma estrutura robusta
- `src/skills/skills_analyzer.py` — `_SYNONYMS` (40+ grupos), `_RESOURCES` (70+), extração multi-estratégia
- `src/pipeline/compose_datasets.py` — `build_skills_map` com merge + fallback + fuzzy_limit configurável
- `scripts/reload_eval.py` — inner_cv=3, n_iter=15, fallback de predict_proba/decision_function
- `pyproject.toml` — Versão 0.7.0, lightgbm como dependência, grupos opcionais NLP
- `README.md` — Stack, métricas e instruções atualizadas
- `todo.md` — Status atualizado com Fase 2 e 3 concluídas

### Resultados
- **Classificação**: XGBoost venceu — F1=72.33% (↑ vs 71.70%), Nested CV F1=70.92%
- **Regressão**: VotingRegressor venceu — RMSE=$33.452 (↓ 3.8%), R²=40.09% (↑ 4.85pp)

## [0.6.0] — 2026-06-25

### Adicionado
- **Gráficos Altair no dashboard ML** — `monitor_dashboard.py` agora exibe:
  - Heatmap da matriz de confusão (Altair, scale blues + contagens)
  - Barras de métricas (Acurácia, F1, Precisão, Recall) com linha meta 70%
  - Histograma dos scores de decisão por classe verdadeira (Fit × No Fit)
  - Scatter salário real × previsto com linha identidade
  - Histograma dos resíduos (previsto − real)
- **Dados de avaliação** — `scripts/reload_eval.py` salva `eval_clf.parquet` (1249 amostras com y_true, y_pred, y_prob) e `eval_reg.parquet` (179 amostras com y_true, y_pred)
- **Re-treino rápido** — `scripts/reload_eval.py` treina LogisticRegression + GradientBoosting em ~30s a partir dos dados já pré-processados

### Alterado
- `monitor_dashboard.py` — Aba "Modelo ML" com Altair (heatmap, barras, histograma, scatter, resíduos) + KPIs
- `pyproject.toml` — Versão 0.6.0
- `todo.md` — Etapa 7 documentada

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
