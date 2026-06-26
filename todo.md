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

---

## Fase 5 — Testes da Fase 4 (PENDENTE)

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
- [ ] `dense_matrix` (np.ndarray 50×10) — dado denso para MLP/GaussianNB
- [ ] `sample_resume_with_skills` — texto real contendo skills do `skills_map.json`
- [ ] `sample_job_titles` — lista de títulos existentes no `skills_map.json`
- [ ] `mock_sbert` (fixture autouse) — `patch('sentence_transformers.SentenceTransformer')` que retorna embeddings 384-dim fixos

---

### `test_vectorizer.py` — SBERT (6 unit + 1 slow)
- [ ] **Unit** `test_sbert_init`: `SentenceBertVectorizer(model_name="fake")` armazena nome
- [ ] **Unit** `test_sbert_transform_mock`: com `mock_sbert`, `transform(["texto"])` retorna array shape (1, 384)
- [ ] **Unit** `test_sbert_fit_save`: `fit(texts, save_path=tmp_path)` salva `.pkl` e `.npy`
- [ ] **Unit** `test_sbert_load_embeddings`: `load_embeddings()` lê `.npy` salvo
- [ ] **Unit** `test_load_sbert_vectorizer`: `load_sbert_vectorizer(tmp_path / "sbert.pkl")` carrega objeto
- [ ] **Slow** `test_sbert_consistency_real`: **real model** — mesmo texto 2x → mesmo embedding (norma L2 < 1e-5)

---

### `test_classifier.py` — MLP + GaussianNB + nested CV (8 unit)
- [ ] `test_mlp_in_candidates`: `"mlp"` em `INDIVIDUAL_CANDIDATES`
- [ ] `test_gaussian_nb_in_candidates`: `"gaussian_nb"` em `INDIVIDUAL_CANDIDATES`
- [ ] `test_dense_only_skips_sparse`: `_is_sparse_compatible("mlp", sparse=True)` → `False`
- [ ] `test_dense_only_allows_dense`: `_is_sparse_compatible("mlp", dense=False)` → `True`
- [ ] `test_nested_cv_clf_returns_tuple`: `train_nested_cv_clf(50 amostras densas, outer=2, inner=2)` retorna `(str, dict, list, model)`
- [ ] `test_nested_cv_clf_all_dense_models`: nested CV com `dense_matrix` — MLP ou GaussianNB podem ser escolhidos
- [ ] `test_mlp_grid_present`: `HYPERPARAM_GRIDS["mlp"]` com `hidden_layer_sizes`, `alpha`, `learning_rate_init`
- [ ] `test_gaussian_nb_grid_present`: `HYPERPARAM_GRIDS["gaussian_nb"]` com `var_smoothing`

---

### `test_salary_model.py` — MLPRegressor (3 unit)
- [ ] `test_mlp_reg_in_candidates`: `"mlp"` em `INDIVIDUAL_CANDIDATES` do salary_model
- [ ] `test_mlp_reg_dense_only`: `DENSE_ONLY_MODELS` contém `"mlp"`
- [ ] `test_nested_cv_reg_returns_tuple`: `train_nested_cv_reg(50 amostras densas, outer=2, inner=2)` retorna `(str, dict, list, model)`

---

### `test_recommender.py` — Cross-encoder (2 unit + 1 slow)
- [ ] **Unit** `test_rerank_cross_encoder_mock`: com `mock_cross_encoder`, `rerank_with_cross_encoder()` retorna DataFrame ordenado
- [ ] **Unit** `test_rank_jobs_cross_encoder_flag`: `rank_jobs(use_cross_encoder=True)` não quebra (fallback TF-IDF)
- [ ] **Slow** `test_cross_encoder_real_scores`: **real model** — pares idênticos → score ≈ 100, pares opostos → score ≈ 0

---

### `test_predictor.py` — Employability + Flags (5 unit)
- [ ] `test_employability_score_range`: `employability_score` entre 0–100
- [ ] `test_employability_score_100`: texto com todas skills → score 100
- [ ] `test_employability_score_0`: texto sem nenhuma skill → score 0
- [ ] `test_predict_use_sbert_flag`: `predict(use_sbert=True)` funciona (fallback TF-IDF silencioso)
- [ ] `test_predict_use_cross_encoder_flag`: `predict(use_cross_encoder=True)` funciona

---

### Resumo

| Métrica | Atual | Meta |
|---|---|---|
| Total de testes | 78 | **~95** |
| `fail_under` | 55% | **60%** |
| Testes `@pytest.mark.slow` | 2 | **4** (2 SBERT + 2 cross-encoder) |
| Cobertura Fase 4 | 0% | **100% dos componentes** |
| Mock externo | Nenhum | `sentence-transformers` (SBERT + CrossEncoder) |

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
