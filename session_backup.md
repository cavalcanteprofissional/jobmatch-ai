# Backup de Sessão — 25/06/2026 16:30

## Status Geral
- **78/78 testes passando**
- **Fase 1 (Nested CV + HP Tuning): CONCLUÍDA**
- **Fase 2 (Ensemble Stacking/Voting/XGBoost): PARCIAL — código pronto, treino pendente**
- **Fase 3 (Skills NLP): Não iniciada**
- **Fase 4 (Neural): Não iniciada**

## Arquivos Modificados Nesta Sessão

| Arquivo | Mudança |
|---------|---------|
| `src/models/classifier.py` | `train_nested_cv_clf()` — nested CV + all candidates (LR, RF, SVM, XGBoost, stacking, voting) + GridSearchCV/ RandomizedSearchCV automático |
| `src/models/salary_model.py` | `train_nested_cv_reg()` — mesma estrutura para regressão (GB, RF, XGBoost, stacking, voting) |
| `scripts/reload_eval.py` | Usa `train_nested_cv_*`, salva nested_cv.scores + best_params + best_candidate no metrics.json |
| `src/app/monitor_dashboard.py` | `nested_cv_chart()` — gráfico Altair de barras com scores por fold + média, seção "Nested CV", expander com hiperparâmetros. `import numpy` adicionado |
| `pyproject.toml` | `xgboost = {version = ">=2.0", python = ">=3.12"}` adicionado. Versão 0.6.0 |
| `todo.md` | Plano das 4 fases registrado |
| `poetry.lock` | Atualizado com xgboost |

## Resultados Obtidos (Fase 1 — sem XGBoost/ensemble)

### Classificação
- **Melhor modelo**: LinearSVC (C=5.0, loss=squared_hinge)
- Nested CV (5×3): F1 médio = **71.91%** (±1.86pp)
- Scores por fold: [71.46%, 72.49%, 69.87%, 75.19%, 70.52%]
- Holdout: accuracy=70.86%, F1=71.70%, precision=69.22%, recall=74.35%
- Matriz: VN=424, FP=205, FN=159, VP=461

### Regressão
- **Melhor modelo**: GradientBoosting (n_est=200, max_depth=5, lr=0.05)
- Nested CV (3×3): RMSE médio = **$40.031** (±$2.479)
- Scores por fold: [$38.024, $43.524, $38.545]
- Holdout: RMSE=$34.779, MAE=$24.015, R²=0.3524

## Código Fase 2 (Já Escrito, Não Treinado)

### classifier.py — Novos candidatos em `ALL_CANDIDATES`
```python
ALL_CANDIDATES = ["logistic_regression", "random_forest", "svm", "xgboost", "stacking", "voting"]
```
- `_make_stacking()`: StackingClassifier(base=[LR, RF, SVM, XGB], meta=LR, cv=3, n_jobs=-1)
- `_make_voting()`: VotingClassifier([LR, RF, XGB], voting="soft", n_jobs=-1) — SVM excluído (sem predict_proba)
- XGBoost: `tree_method='hist'` (rápido CPU), grid: n_est=[100,200], max_depth=[3,5], lr=[0.05]

### salary_model.py — Novos candidatos
```python
ALL_CANDIDATES = ["gradient_boosting", "random_forest", "xgboost", "stacking", "voting"]
```
- `_make_stacking()`: StackingRegressor(base=[GB, RF, XGB], meta=Ridge, cv=3, n_jobs=-1)
- `_make_voting()`: VotingRegressor([GB, RF, XGB], n_jobs=-1)
- XGBoost: mesmo grid do classifier

### Grid sizes (após redução)
- LR: 4 combos → GridSearchCV
- RF: 8 combos → GridSearchCV
- SVM: 4 combos → GridSearchCV
- XGBoost: 4 combos → GridSearchCV
- Stacking: sem grid
- Voting: sem grid

### Config reload_eval.py
- Classifier: outer_cv=3, inner_cv=2, n_iter=8
- Regressor: outer_cv=3, inner_cv=2, n_iter=6

## Problemas Conhecidos
1. **XGBoost lento** com n_iter grande — grid reduzido para 4 combos (GridSearchCV). Com `tree_method='hist'` e outer_cv=3, inner_cv=2, cada fold deve levar ~2-3 min
2. **GridSearchCV não aceita `random_state`** — fix: só passa o param quando não é grid search
3. **VotingClassifier soft voting** — SVM (LinearSVC) não tem predict_proba, excluído do voting
4. **Ensemble candidates são singletons** — `_make_stacking()` é chamada no import. `set_params()` pode mutar estado global. `_get_model("stacking")` sempre retorna o mesmo objeto
5. **poetry install** falha com "No file/folder found for package jobmatch-ai" — erro pré-existente, não afeta execução com `poetry run`
6. **Window codec cp1252** — prints com Unicode (→, ç, ã) falham. Todo código usa ASCII-safe prints

## Para Retomar
```bash
# Treinar com ensemble (leva ~30 min)
PYTHONPATH=. poetry run python scripts/reload_eval.py

# Rodar testes
poetry run pytest -x -q

# Rodar dashboard
streamlit run src/app/monitor_dashboard.py

# Se xgboost falhar
poetry add xgboost --python ">=3.12"
```
