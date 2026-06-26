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

## Fase 4 — Neural Approaches + Embeddings (PRÓXIMA SESSÃO)

---

### Sentence Embeddings
- [ ] Adicionar `sentence-transformers` ao projeto
- [ ] `src/models/vectorizer.py`: alternativa `fit_sentence_embeddings()` usando Sentence-BERT (`all-MiniLM-L6-v2`)
- [ ] Comparar desempenho TF-IDF vs Sentence-BERT no classificador

### Redes Neurais (requer embeddings densos)
- [ ] **MLPClassifier** + **MLPRegressor** como candidatos no nested CV
- [ ] **CatBoost** + **GaussianNB** para classificação
- [ ] Grids de hiperparâmetros para modelos densos

### Cross-encoder para Re-ranking
- [ ] (Opcional) Re-ranking das top-10 vagas com cross-encoder

### Upload de Currículo
- [ ] Upload de PDF/DOCX via `st.file_uploader` + extração de texto (PyPDF2 + python-docx)
- [ ] Comparar skills com múltiplas vagas (não só a #1)
- [ ] Score de empregabilidade: % de skills do candidato vs mercado

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
| Testes | pytest + pytest-cov | 78 testes, fail_under=55% |
| NLP | spaCy (Fase 3) + NLTK (atual) | POS-aware lemmatization + NER |
