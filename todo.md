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

## Decisões Técnicas

| Decisão | Escolha | Motivo |
|---------|---------|--------|
| Vetorização principal | TF-IDF (sklearn) | Fase inicial, sem LLMs |
| Vetorização alternativa | Sentence-BERT (sentence-transformers) | Embeddings semânticos para Fase 4 |
| Serialização | joblib | Padrão sklearn |
| Formato dados | Parquet (pyarrow) | Eficiente para colunas textuais |
| Fuzzy match | rapidfuzz | Mais rápido que fuzzywuzzy |
| Frontend | Streamlit + FastAPI | Híbrido: REST com fallback direto |
| Python | >=3.11,<3.13 | Compatibilidade com dependências |
| Gerenciamento | Poetry | Reprodutibilidade de ambiente |
| Testes | pytest + pytest-cov | 78 testes → ~95, fail_under=55% → 60% |
| Mock em testes | `unittest.mock.patch` | Apenas para dependências opcionais (sentence-transformers, catboost) |
| Dados de teste | Reais (skills_map) + dense numpy 50×10 | MLP/GaussianNB testados sem mock, lógica pura com dados reais |
| NLP | spaCy (Fase 3) + NLTK (atual) | POS-aware lemmatization + NER |
