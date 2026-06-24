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
| Testes (71 testes, 67% cobertura) | ✅ Concluído |
| Config Kaggle (.env.local, .gitignore, config.py) | ✅ Concluído |
| Notebooks (EDA, pipeline, classificador, salário) | ✅ Concluído |
| `train_pipeline.py` | ✅ Concluído |

## Pendentes (próximas versões)

- [ ] Deploy da API (Docker / cloud)
- [ ] CI/CD (GitHub Actions)
- [ ] Dashboard de monitoramento
- [ ] Testes de integração com dados reais

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
| Testes | pytest + pytest-cov | 71 testes, fail_under=55% |
| Credenciais | python-dotenv + .env.local | Segurança, sem versionar segredos |
