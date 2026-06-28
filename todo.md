# JobMatch AI — Plano de Implementação

## Status Geral

| Módulo | Status |
|--------|--------|
| Estrutura de diretórios | ✅ Concluído |
| Pipeline de dados (preprocess, load, compose) | ✅ Concluído |
| Modelos (classifier, salary, recommender, vectorizer) | ✅ Concluído |
| Skills analyzer (regex + gap analysis + sinônimos) | ✅ Concluído |
| skills_map.json (922 títulos, 20.6k skills) | ✅ Populado |
| Fase 2 (XGBoost + ExtraTrees + LightGBM + Ensemble) | ✅ Concluído |
| Fase 3 (Skills NLP + sinônimos + resources) | ✅ Concluído |
| Frontend Streamlit (híbrido API/direto) | ✅ Concluído |
| API REST FastAPI (predict, health, models/info, metrics) | ✅ Concluído |
| Testes (78 testes passando) | ✅ Concluído |
| Dashboard Monitoramento (Altair + nested CV) | ✅ Concluído |
| Docker / CI-CD | ✅ Concluído |
| Fase 4 (Neural: SBERT + MLP + Cross-encoder) | ✅ Concluído |
| Fase 5 (Testes Fase 4 — ~95 testes, 85 rápidos + 4 slow + SBERT) | ✅ Concluído |
| Correção travamento nested CV | ✅ Concluído |

## Resultados Atuais (Fase 2 — com XGBoost + Ensemble + LightGBM)

### Classificação
- **Melhor modelo**: **XGBoost** (max_depth=5, n_estimators=200, lr=0.05)
- Nested CV (3×3): F1 médio = **70.92%** (±2.39pp)
- Holdout: accuracy=71.82%, F1=**72.33%**, precision=70.55%, recall=74.19%

### Regressão
- **Melhor modelo**: **VotingRegressor** (GB + RF + XGB + ET + LGBM)
- Nested CV (3×2): RMSE médio = **$39.662** (±$2.295)
- Holdout: RMSE=**$33.452**, MAE=$22.688, R²=**40.09%**

## Skills NLP (Fase 3 — Concluída)
- ✅ `skills_map.json`: 922 títulos mapeados, 20.607 skills, 46.5% das vagas
- ✅ Sinonímia: 40+ grupos de sinônimos (ex: "tf" ↔ "tensorflow", "k8s" ↔ "kubernetes")
- ✅ `_RESOURCES` expandido de 20 → 70+ skills com cursos recomendados
- ✅ Matching multi-estratégia: regex + sinônimos + gap analysis
- ✅ `jobs_clean.parquet` atualizado com `required_skills` preenchido

---

## Fase 4 — Neural Approaches + Embeddings (CÓDIGO CONCLUÍDO)

---

### Sentence Embeddings
- [x] Adicionar `sentence-transformers` ao projeto (grupo `nlp` opcional)
- [x] `src/models/vectorizer.py`: `SentenceBertVectorizer` com `all-MiniLM-L6-v2`
- [x] Comparar desempenho TF-IDF vs Sentence-BERT salvo em `metrics.json["classification"]["sbert"]`
- [x] SBERT opcional na inferência (`use_sbert=True`) no `JobMatchPredictor` + API + frontend

### Redes Neurais
- [x] MLPClassifier + MLPRegressor como candidatos no nested CV
- [x] CatBoost + GaussianNB para classificação
- [x] Grids de hiperparâmetros para modelos densos

### Cross-encoder para Re-ranking
- [x] Re-ranking com cross-encoder (`cross-encoder/stsb-MiniLM-L-6-v2`) em `recommender.py`
- [x] Score normalizado em [0, 1] com `np.clip(scores, 0, 1) * 100`
- [x] Toggle no frontend + suporte na API

### Upload de Currículo + Empregabilidade
- [x] Upload de PDF/DOCX via `st.file_uploader` + PyPDF2 + python-docx
- [x] Comparar skills com múltiplas vagas (top_k)
- [x] Score de empregabilidade corrigido: média % de skills compatíveis por vaga

### Correções de Bugs (26/06)
- [x] `reload_eval.py`: `logger` import adicionado (crash no SBERT path)
- [x] `streamlit_app.py`: employability score corrigido (era métrica sem sentido)
- [x] `predictor.py`: adicionado `employability_score`, `use_sbert`, `use_cross_encoder`
- [x] `server.py`: `PredictRequest` com `use_sbert` + `use_cross_encoder`
- [x] `recommender.py`: cross-encoder trocado para STS (score em [0,1])
- [x] `monitor_dashboard.py`: exibe comparação TF-IDF vs SBERT
- [x] `reload_eval.py`: salva `classifier_sbert.pkl` + `jobs_sbert_embeddings.npy`

### Correções de Performance — Nested CV Travando (26/06)
**Diagnóstico:** `reload_eval.py` travava o computador durante o nested CV. Causas identificadas:

1. **StackingClassifier/StackingRegressor dentro do nested CV** — Cada fold testava o stacking, que internamente re-treina 5-7 modelos com cv=2, criando um loop exponencial de treinos
2. **Modelos demais** — 11 candidatos na classificação + 12 na regressão (incluindo MLP, GaussianNB que densificam a matriz, e ensembles)
3. **`n_jobs=-1` em todos os modelos** — Paralelismo máximo saturava a RAM em máquinas com pouca memória

**Soluções aplicadas:**

| Arquivo | Mudança |
|---------|---------|
| `src/models/classifier.py` | Criado `NESTED_CV_CANDIDATES` — só modelos individuais e compatíveis com sparse (exclui stacking, voting, MLP, GaussianNB). `N_JOBS=2` |
| `src/models/salary_model.py` | Criado `NESTED_CV_CANDIDATES` — só modelos individuais compatíveis (exclui stacking, voting, MLP). `N_JOBS=2` |
| `scripts/reload_eval.py` | Adicionados timestamps (`[HH:MM:SS]`) em cada etapa para diagnóstico |
| `train_nested_cv_clf/reg` | Usam `NESTED_CV_CANDIDATES` para dados sparse, `INDIVIDUAL_CANDIDATES` para densos. Retreino final ainda usa `ALL_CANDIDATES` |

**Impacto:** Redução drástica do número de fits por fold. Stacking/Voting só entram no treino final (não no nested CV). `n_jobs=2` evita thrashing de memória.

### Correção de Compatibilidade metrics.json (26/06)
- [x] `reload_eval.py` não salvava `vectorizer_features` no `model_info`, causando `KeyError` no `monitor_dashboard.py`
- [x] Adicionado `"vectorizer_features": vec.max_features` no `model_info` do `reload_eval.py`
- [x] `monitor_dashboard.py` passou a usar `.get('vectorizer_features', 'N/A')` como fallback defensivo

---

## Fase 5 — Testes da Fase 4 (CONCLUÍDA)

**78 testes atuais — 0 cobrem componentes da Fase 4.**

### Abordagem Híbrida: Mock vs Real

| Componente | Abordagem | Motivo |
|---|---|---|
| `SentenceBertVectorizer` (fit/transform) | **Mock** `sentence-transformers` | Dependência opcional (`poetry install --with nlp`). Mock no `conftest.py` com `unittest.mock.patch` |
| MLPClassifier / GaussianNB / MLPRegressor | **Real (sklearn)** | Zero dependência extra. Treino em 50 amostras reais, < 1s |
| Cross-encoder rerank | **Mock** `CrossEncoder` + 1 **slow real** | 80MB download + torch. Mock testa fallback e fluxo; slow testa semântica |
| Employability score | **Real (skills_map.json)** | Lógica pura Python, rápido, skills_map já existe |
| Nested CV (dense) | **Real (50 amostras)** | Valida convergência real sem mock |
| SBERT inference flag | **Mock SBERT** + fallback TF-IDF real | Testa branch `use_sbert=True` sem dependência |

---

### `conftest.py` — Novos Fixtures
- [x] `dense_matrix` (np.ndarray 50×10) — dado denso para MLP/GaussianNB
- [x] `sample_resume_with_skills` — texto real contendo skills do `skills_map.json`
- [x] `sample_job_titles` — lista de títulos existentes no `skills_map.json`
- [x] `mock_sbert` (fixture autouse) — `patch('sentence_transformers.SentenceTransformer')` que retorna embeddings 384-dim fixos

---

### `test_vectorizer.py` — SBERT (6 unit + 1 slow)
- [x] **Unit** `test_sbert_init`: `SentenceBertVectorizer(model_name="fake")` armazena nome
- [x] **Unit** `test_sbert_transform_mock`: com `mock_sbert`, `transform(["texto"])` retorna array shape (1, 384)
- [x] **Unit** `test_sbert_fit_save`: `fit(texts, save_path=tmp_path)` salva `.pkl` e `.npy`
- [x] **Unit** `test_sbert_load_embeddings`: `load_embeddings()` lê `.npy` salvo
- [x] **Unit** `test_load_sbert_vectorizer`: `load_sbert_vectorizer(tmp_path / "sbert.pkl")` carrega objeto
- [x] **Slow** `test_sbert_consistency_real`: **real model** — mesmo texto 2x → mesmo embedding (norma L2 < 1e-5)

---

### `test_classifier.py` — MLP + GaussianNB + nested CV (8 unit)
- [x] `test_mlp_in_candidates`: `"mlp"` em `INDIVIDUAL_CANDIDATES`
- [x] `test_gaussian_nb_in_candidates`: `"gaussian_nb"` em `INDIVIDUAL_CANDIDATES`
- [x] `test_dense_only_skips_sparse`: `_is_sparse_compatible("mlp", sparse=True)` → `False`
- [x] `test_dense_only_allows_dense`: `_is_sparse_compatible("mlp", dense=False)` → `True`
- [x] `test_nested_cv_clf_returns_tuple`: `train_nested_cv_clf(50 amostras densas, outer=2, inner=2)` retorna `(str, dict, list, model)`
- [x] `test_nested_cv_clf_all_dense_models`: nested CV com `dense_matrix` — MLP ou GaussianNB podem ser escolhidos
- [x] `test_mlp_grid_present`: `HYPERPARAM_GRIDS["mlp"]` com `hidden_layer_sizes`, `alpha`, `learning_rate_init`
- [x] `test_gaussian_nb_grid_present`: `HYPERPARAM_GRIDS["gaussian_nb"]` com `var_smoothing`

---

### `test_salary_model.py` — MLPRegressor (3 unit)
- [x] `test_mlp_reg_in_candidates`: `"mlp"` em `INDIVIDUAL_CANDIDATES` do salary_model
- [x] `test_mlp_reg_dense_only`: `DENSE_ONLY_MODELS` contém `"mlp"`
- [x] `test_nested_cv_reg_returns_tuple`: `train_nested_cv_reg(50 amostras densas, outer=2, inner=2)` retorna `(str, dict, list, model)`

---

### `test_recommender.py` — Cross-encoder (2 unit + 1 slow)
- [x] **Unit** `test_rerank_cross_encoder_mock`: com `mock_cross_encoder`, `rerank_with_cross_encoder()` retorna DataFrame ordenado
- [x] **Unit** `test_rank_jobs_cross_encoder_flag`: `rank_jobs(use_cross_encoder=True)` não quebra (fallback TF-IDF)
- [x] **Slow** `test_cross_encoder_real_scores`: **real model** — pares idênticos → score ≈ 100, pares opostos → score ≈ 0

---

### `test_predictor.py` — Employability + Flags (5 unit)
- [x] `test_employability_score_range`: `employability_score` entre 0–100
- [x] `test_employability_score_100`: texto com todas skills → score 100
- [x] `test_employability_score_0`: texto sem nenhuma skill → score 0
- [x] `test_predict_use_sbert_flag`: `predict(use_sbert=True)` funciona (fallback TF-IDF silencioso)
- [x] `test_predict_use_cross_encoder_flag`: `predict(use_cross_encoder=True)` funciona

---

### Resumo

| Métrica | Atual | Meta |
|---|---|---|
| Total de testes | **~95** | ~95 |
| `fail_under` | **60%** | 60% |
| Testes `@pytest.mark.slow` | **4** (2 SBERT + 2 cross-encoder) | 4 |
| Cobertura Fase 4 | **100% dos componentes** | 100% |
| Mock externo | `sentence-transformers` (SBERT + CrossEncoder) | `sentence-transformers` (SBERT + CrossEncoder) |

---

---

## Fase 6 — Migração Frontend: Streamlit → React + Vite (EM ANDAMENTO — branch `feat/frontend-react`)

### Motivação
- Streamlit é limitado para UI complexa, Two Streamlits separados fragmentam a experiência
- React + Vite + Tailwind oferece componentização, tipagem forte e performance
- Backend REST FastAPI já está pronto, frontend só adiciona consumo

### Plano de substituição

| Etapa | Tarefa | Status |
|-------|--------|--------|
| 1 | Scaffold Vite + React 18 + TypeScript + Tailwind | ⬜ |
| 2 | Serviço de API (`api.ts`) + tipos TypeScript (`models.ts`) | ⬜ |
| 3 | Página JobMatch (upload PDF/DOCX, formulário, resultados, skills) | ⬜ |
| 4 | Página Monitor (métricas ML, gráficos Recharts, métricas API) | ⬜ |
| 5 | `Dockerfile.frontend` + `nginx.conf` (proxy reverso) | ⬜ |
| 6 | Atualizar `docker-compose.yml` (api + frontend, remover streamlit) | ⬜ |
| 7 | Remover `src/app/streamlit_app.py` e `src/app/monitor_dashboard.py` | ⬜ |
| 8 | Atualizar `README.md` (setup, stack, testes) + `CHANGELOG.md` | ⬜ |

### Stack
| Camada | Escolha |
|--------|---------|
| Build | Vite |
| UI | React 18 + TypeScript |
| Estilo | Tailwind CSS |
| Gráficos | Recharts |
| Upload | react-dropzone |
| Router | React Router v6 |
| HTTP | fetch nativo |

### O que NÃO muda
- `src/api/server.py`, `src/api/predictor.py`, `src/monitoring/` — intactos
- Pipeline de dados, modelos, testes — intactos
- `pyproject.toml` — Streamlit pode virar dependência opcional ou ser removido

---

## Decisões Técnicas

| Decisão | Escolha | Motivo |
|---------|---------|--------|
| Vetorização principal | TF-IDF (sklearn) | Fase inicial, sem LLMs |
| Vetorização alternativa | Sentence-BERT (sentence-transformers) | Embeddings semânticos para Fase 4 |
| Serialização | joblib | Padrão sklearn |
| Formato dados | Parquet (pyarrow) | Eficiente para colunas textuais |
| Fuzzy match | rapidfuzz | Mais rápido que fuzzywuzzy |
| Frontend (atual) | React + Vite + Tailwind | Substituiu Streamlit por componentização e perfomance |
| Frontend (legado) | Streamlit + FastAPI | Híbrido: REST com fallback direto (removido) |
| Deploy Cloud | Render.com (Web Service API + Static Site Frontend) | Free tier, Docker API, Static Site React, disk persistente |
| Python | >=3.11,<3.13 | Compatibilidade com dependências |
| Gerenciamento | Poetry | Reprodutibilidade de ambiente |
| Testes | pytest + pytest-cov | 78 testes → ~95, fail_under=55% → 60% |
| Mock em testes | `unittest.mock.patch` | Apenas para dependências opcionais (sentence-transformers, catboost) |
| Dados de teste | Reais (skills_map) + dense numpy 50×10 | MLP/GaussianNB testados sem mock, lógica pura com dados reais |
| NLP | spaCy (Fase 3) + NLTK (atual) | POS-aware lemmatization + NER |

---

## Fase 7 — Deploy Cloud: Render.com (EM ANDAMENTO — branch `feat/frontend-react`)

### Estratégia

| Serviço | Plataforma | Tipo | Custo |
|---------|-----------|------|-------|
| API FastAPI | Render | Web Service (Docker) | Free |
| Frontend React | Render | Static Site | Free |
| Modelos treinados | Embutidos na imagem Docker (~14 MB) | `data/models/` + `data/processed/` incluídos no build | — |

> **Motivo:** Free tier do Render tem 512 MB de RAM, insuficiente para rodar
> `compose_datasets` + `train_pipeline` + `reload_eval`. Solução: treinar local,
> commitar `data/` (modelos + dados processados) no git, e embutir na imagem Docker.

### Arquivos de Deploy

| Arquivo | Função |
|---------|--------|
| `Dockerfile` (raiz) | Build da API Python (multistage, Poetry, python:3.11-slim) |
| `render.yaml` (raiz) | Blueprint Render: define Web Service (API). Frontend criado manualmente |
| `scripts/startup.sh` | Só inicia uvicorn. Modelos vêm embutidos na imagem (sem pipeline) |
| `frontend/src/services/api.ts` | Fallback: cloud → local. Usa `VITE_API_URL` se disponível |
| `.dockerignore` | Exclui `data/raw/` mas **inclui** `data/models/` e `data/processed/` |

### Bugs corrigidos nesta fase

| # | Bug | Arquivo | Fix | Data |
|---|-----|---------|-----|------|
| B1 | `TypeError: 'NoneType' object is not iterable` ao iniciar API | `src/api/server.py:192` | `app.openapi_tags` é `None` por padrão, não lista vazia. Substituído loop de dedup por `app.openapi_tags = TAGS_META` | 26/06 |
| B2 | `type: static` não suportado no Render Blueprint | `render.yaml` | Removido do `render.yaml`. Frontend Static Site criado manualmente via Dashboard | 26/06 |
| B3 | Env vars `PYTHONUNBUFFERED`/`PYTHONDONTWRITEBYTECODE` desnecessárias no Dashboard | `Dockerfile`, `render.yaml` | Movidas para `ENV` no Dockerfile. `render.yaml` e `docker-compose.yml` limpos | 26/06 |
| B4 | `Out of Memory (512 Mi)` no free tier ao executar pipeline de treino | `scripts/startup.sh`, `.dockerignore` | **Mudança de estratégia:** modelos embutidos na imagem Docker. `startup.sh` simplificado (só uvicorn). `.dockerignore` agora inclui `data/models/` e `data/processed/` | 27/06 |
| B5 | `Exited with status 1` no Render após inicio — NLTK data não baixada na imagem | `Dockerfile`, `scripts/startup.sh` | Adicionado `nltk.download` no builder stage. Adicionado import test no `startup.sh`. Removido `HEALTHCHECK` do Dockerfile (conflito com health check do Render) | 27/06 |
| B6 | `libgomp.so.1: cannot open shared object file` — lightgbm/xgboost crasham no import mesmo com try/except ImportError | `Dockerfile`, `classifier.py`, `salary_model.py` | `try/except ImportError` não captura `OSError` (ctypes `_dlopen`). Solução: `libgomp1` no apt-get + trocar `except ImportError` por `except (ImportError, OSError)` nos blocos de lightgbm e xgboost | 27/06 |
| B7 | CORS — frontend local não consegue bater na API do Render | `server.py` | Adicionado `CORSMiddleware` com origens `localhost:5173`, `localhost:3000`, `https://jobmatch-frontend.onrender.com` | 27/06 |
| B8 | SPA routing — aba Monitor crasha "Not Found" ao clicar (navegação nativa `<a>` em vez de `<Link>`) | `App.tsx` | Trocar `<a href>` por `<Link to>` + criar `_redirects` para SPA fallback no Render | 27/06 |
| B9 | Monitor mostra "API não disponível" — cold start do free tier + fallback localhost inválido + sem timeout/retry | `api.ts`, `Monitor.tsx` | Remover fallback localhost em produção. Adicionar AbortController timeout (15s) + retry (3x, backoff 2-4-8s). Loading state com spinner no Monitor | 27/06 |
| B10 | CORS bloqueado — Render Static Site tem hash no subdomínio (`u6vt`), mas API só permite origem sem hash | `server.py` | Trocar `allow_origins` por `["*"]` (API pública, CORS aberto não afeta segurança) | 27/06 |

### Checklist

| # | Tarefa | Status |
|---|--------|--------|
| 1 | Criar `render.yaml` (Blueprint) | ✅ |
| 2 | Criar `scripts/startup.sh` (treino automático no primeiro deploy) | ✅ |
| 3 | Ajustar `Dockerfile` CMD para usar `startup.sh` | ✅ |
| 4 | Ajustar `frontend/src/services/api.ts` para suportar `VITE_API_URL` | ✅ |
| 5 | Corrigir `app.openapi_tags` (None → lista) em `server.py` | ✅ |
| 6 | Corrigir `render.yaml` — remover `type: static` | ✅ |
| 7 | Mover env vars Python para `ENV` no Dockerfile (0 env vars no Render) | ✅ |
| 8 | Simplificar `startup.sh` (só uvicorn, sem pipeline de treino) | ✅ |
| 9 | Ajustar `.dockerignore` (incluir `data/models/` e `data/processed/`) | ✅ |
| 10 | Ajustar `.gitignore` (incluir modelos treinados) | ✅ |
| 11 | Ajustar `api.ts` com fallback cloud → local | ✅ |
| 12 | Adicionar NLTK data download no Dockerfile | ✅ |
| 13 | Adicionar import test no startup.sh + remover HEALTHCHECK | ✅ |
| 14 | Criar conta no Render + conectar GitHub | ⬜ (manual) |
| 15 | Deploy API (Web Service) no Render com modelos embutidos | ⬜ (manual) |
| 16 | Deploy Frontend (Static Site) no Render | ⬜ (manual) |
| 17 | Testar endpoints públicos | ⬜ (manual) |
| 18 | Atualizar `README.md` | ✅ |
| 19 | Adicionar `libgomp1` no apt-get do Dockerfile e capturar OSError no try/except de lightgbm/xgboost | ✅ |
| 20 | Criar `.env.local` e `.env.example` no frontend para testar local com API cloud | ✅ |
| 21 | Adicionar `CORSMiddleware` na API (permitir frontend local + Render) | ✅ |

### Checklist Aprimoramentos Frontend

| # | Tarefa | Prioridade | Status |
|---|--------|-----------|--------|
| 1 | Dark mode foundation — configurar `darkMode: 'class'`, ThemeContext, toggle no navbar | Alta | ✅ |
| 2 | Tokens de cor no tailwind.config.js — paleta centralizada light/dark | Alta | ✅ |
| 3 | Refatorar JobMatch dark mode — classes `dark:` em todos os elementos | Alta | ✅ |
| 4 | Refatorar Monitor dark mode — classes `dark:` em tabela, métricas, abas | Alta | ✅ |
| 5 | Refatorar Charts dark mode — cores via variáveis de tema | Média | ✅ |
| 6 | Exibir dados da API não usados — `score_pct`, `fit_label`, `estimated_annual_usd`, `best_params`, `best_candidate` | Média | ✅ |
| 7 | ScatterChart regressão — gráfico predito vs real com `/eval/regression` | Média | ✅ |
| 8 | ScoreDistribution classificação — histograma de scores com `/eval/classification` | Média | ✅ |
| 9 | Animações e transições — `transition-colors`, loading skeleton, fade-in | Baixa | ✅ |
| 10 | Responsividade — `min-h` nos gráficos, grid mobile adaptável | Baixa | ✅ |
| 11 | Badge Fit/No Fit por vaga nos resultados | Baixa | ✅ |

---

## Fase 8 — Dark Mode + Aprimoramentos Frontend (CONCLUÍDA — branch `feat/frontend-react`)

### Checklist Fase 8

| # | Tarefa | Prioridade | Status |
|---|--------|-----------|--------|
| 1 | Dark mode foundation (ThemeContext, toggle, localStorage) | Alta | ✅ |
| 2 | Tokens de cor no tailwind.config.js | Alta | ✅ |
| 3 | Refatorar JobMatch dark mode | Alta | ✅ |
| 4 | Refatorar Monitor dark mode | Alta | ✅ |
| 5 | Charts com CSS variables | Média | ✅ |
| 6 | Exibir score_pct, fit_label, estimated_annual_usd, best_params, best_candidate | Média | ✅ |
| 7 | RegressionScatter (scatter predito vs real) | Média | ✅ |
| 8 | ScoreDistribution (histograma de scores) | Média | ✅ |

---

## Fase 9 — Correções de Legibilidade e Overflow (EM ANDAMENTO — branch `feat/frontend-react`)

### Diagnóstico
Análise completa dos 15 arquivos de frontend revelou 11 problemas de layout e contraste.

| Prioridade | Problemas |
|-----------|-----------|
| 🔴 Crítico | `best_params` vaza da box, `best_candidate` sem truncate, título job quebra flex, endpoint tabela sem break-all, skills_desc sem break-words |
| 🟡 Médio | Badges skills contraste baixo dark, warning boxes ilegíveis, empty state invisível |
| 🟢 Baixo | Tooltip chart text, dev plan break-words, fit_label card adaptável |

### Checklist

| # | Tarefa | Prioridade | Status |
|---|--------|-----------|--------|
| 1 | `best_params` — trocar Metric por `<details>` expansível com chave-valor | 🔴 | ⬜ |
| 2 | `best_candidate` — `truncate` + `max-w` no Metric | 🔴 | ⬜ |
| 3 | Job title `<summary>` — `truncate` + `overflow-hidden` no flex | 🔴 | ⬜ |
| 4 | Endpoint tabela — `break-all` + `max-w-[200px]` | 🔴 | ⬜ |
| 5 | `skills_desc` — `break-words` no container | 🔴 | ⬜ |
| 6 | Badges skills dark — `*-900/60` + `*-200` + borda | 🟡 | ⬜ |
| 7 | Warning boxes dark — `*-900/50` + `*-200` | 🟡 | ⬜ |
| 8 | Empty state — `dark:text-gray-400` (era gray-500) | 🟡 | ⬜ |
| 9 | Tooltip chart text — `--tooltip-text: #d1d5db` | 🟢 | ⬜ |
| 10 | Dev plan — `break-words` nos parágrafos | 🟢 | ⬜ |
| 11 | `fit_label` MetricCard — `text-lg` se > 10 chars | 🟢 | ⬜ |

### Progresso

| Data | Avanço |
|------|--------|
| 27/06 | Análise completa, registro do plano |

### Motivação
- Frontend sem dark mode — experiência ruim em ambiente noturno
- API retorna dados ricos (`score_pct`, `best_params`, eval data) que não são exibidos
- Gráficos sub-aproveitados: scatter de regressão e distribuição de scores não existem

### Estratégia
1. `darkMode: 'class'` no Tailwind — toggle no navbar, persistência em localStorage
2. Tokens de cor centralizados no `tailwind.config.js` (extend colors)
3. Classes `dark:` adicionadas em todos os componentes, sem refatorar estrutura
4. Cores dos gráficos Recharts via variáveis CSS (mudam com o tema)
5. Novos componentes: `RegressionScatter.tsx`, `ScoreDistribution.tsx`
6. Dados não exibidos: `score_pct`, `fit_label`, `estimated_annual_usd`, `best_params`, `best_candidate`

### Paleta Dark Mode

| Token | Light | Dark |
|-------|-------|------|
| `bg-primary` | `gray-50` | `gray-950` |
| `bg-surface` | `white` | `gray-900` |
| `bg-surface-2` | `gray-100` | `gray-800` |
| `text-primary` | `gray-900` | `gray-100` |
| `text-secondary` | `gray-500` | `gray-400` |
| `accent` | `indigo-600` | `indigo-400` |
| `border` | `gray-200`/`gray-300` | `gray-700` |
| `success` | `emerald-600` | `emerald-400` |
| `danger` | `red-600` | `red-400` |
| `chart-line` | `#667eea` | `#818cf8` |

---

### Fluxo de acesso final

```
Navegador → https://jobmatch.onrender.com (Static Site React)
                              ↓
                    fetch("/api/predict") → https://jobmatch-api.onrender.com/predict (Web Service)
```

---

## Fase 10 — Cold Start + Timeout + Retry no Frontend (EM ANDAMENTO — branch `feat/frontend-react`)

### Problema
Monitor mostra "API não disponível" porque:
1. Render free tier dorme após 15 min de inatividade → cold start de ~30-60s
2. `api.ts` faz fallback para `localhost:8000` em produção (inútil e demorado)
3. `fetch()` sem timeout — espera até o timeout do navegador (60-300s)
4. Sem retry — se a primeira tentativa falha (container restartando), não tenta de novo
5. Monitor não tem loading state — mostra "erro" imediatamente sem distinção de "carregando"
6. CORS mismatch — Render Static Site recebe hash aleatório no subdomínio (`jobmatch-frontend-XXXX.onrender.com`), mas `server.py` só permite a origem sem hash. Navegador bloqueia a requisição.

### Checklist

| # | Tarefa | Status |
|---|--------|--------|
| 1 | `api.ts`: remover fallback localhost quando `VITE_API_URL` existir | ✅ |
| 2 | `api.ts`: adicionar `AbortController` com timeout configurável | ✅ |
| 3 | `api.ts`: adicionar retry com backoff exponencial (3 tentativas, 2-4-8s) | ✅ |
| 4 | `Monitor.tsx`: adicionar loading state, spinner durante carregamento | ✅ |
| 5 | `Monitor.tsx`: botão "Tentar novamente" após todas as tentativas esgotarem | ✅ |
| 6 | `vite.config.ts`: plugin inline que escreve `dist/_redirects` no `closeBundle` (garante SPA routing independente de cache) | ✅ |
| 7 | (Manual) Dashboard Render: Clear build cache + Deploy | ⬜ |
| 8 | Build + commit + push | ⬜ |
| 9 | `server.py`: trocar `allow_origins` por `["*"]` (CORS aberto — API pública) | ✅ |
