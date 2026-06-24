---
name: jobmatch-ai
description: >
  Guia completo para implementar o projeto JobMatch AI — sistema de ML que calcula score de aderência
  entre currículos e vagas, classifica Fit/No Fit, recomenda Top-5 vagas e identifica skills faltantes.
  Use esta skill sempre que o usuário mencionar JobMatch, matching de currículo e vaga, score de aderência,
  pipeline de dados LinkedIn + HuggingFace, classificação Fit/No Fit, recomendação de vagas com ML,
  NLP com TF-IDF para currículos, ou qualquer tarefa de implementação deste projeto de dados.
  Acione também quando o usuário pedir ajuda com a pipeline de composição dos datasets, frontend Streamlit
  para análise de compatibilidade, ou modelos de regressão salarial para vagas.
---

# JobMatch AI — Guia de Implementação

Sistema de ML para matching inteligente entre currículos e vagas de emprego.

## Visão Geral do Projeto

| Componente | Descrição |
|---|---|
| **Entrada** | Perfil/currículo em texto livre (usuário digita ou cola) |
| **Saída** | Score de aderência, classificação Fit/No Fit, Top-5 vagas, skills gap, faixa salarial |
| **ML Core** | TF-IDF + Cosine Similarity + Classificadores + Regressor de salário |
| **Frontend** | Streamlit (padrão) ou alternativa |
| **Dados** | 3 datasets compostos (ver seção Datasets) |

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────┐
│                    JOBMATCH AI                          │
│                                                         │
│  [Input: Perfil/Currículo Texto]                        │
│           │                                             │
│           ▼                                             │
│  ┌─────────────────┐    ┌──────────────────────────┐   │
│  │  Preprocessamento│    │   Pipeline de Dados       │   │
│  │  NLP (TF-IDF)   │◄───│   (Datasets Compostos)    │   │
│  └────────┬────────┘    └──────────────────────────┘   │
│           │                                             │
│    ┌──────┴──────┐                                      │
│    │             │                                      │
│    ▼             ▼                                      │
│  [Classificador] [Similarity Engine]                    │
│  Fit / No Fit    Cosine → Ranking                       │
│    │             │                                      │
│    └──────┬──────┘                                      │
│           │                                             │
│           ▼                                             │
│  ┌─────────────────────────────┐                        │
│  │  Resultados para o usuário  │                        │
│  │  • Score de aderência       │                        │
│  │  • Fit / No Fit             │                        │
│  │  • Top-5 vagas              │                        │
│  │  • Skills compatíveis       │                        │
│  │  • Skills faltantes + plano │                        │
│  │  • Faixa salarial estimada  │                        │
│  └─────────────────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

---

## Datasets

> ⚠️ **Passo crítico**: Os 3 datasets precisam ser compostos em uma pipeline unificada.
> Consulte `references/pipeline-dados.md` para o guia detalhado de composição.

### 1. LinkedIn Job Postings 2023-2024
- **Fonte**: Kaggle — `arshkon/linkedin-job-postings`
- **Volume**: ~124.000 vagas reais
- **Colunas relevantes**: `title`, `description`, `skills_desc`, `max_salary`, `min_salary`, `pay_period`, `location`, `company_name`
- **Uso no projeto**: Base principal de vagas para ranking e regressão de salário

### 2. Resume-JD-Match
- **Fonte**: HuggingFace — `cnamuangtoun/resume-job-description-fit`
- **Conteúdo**: Pares currículo ↔ vaga com rótulo `Fit` / `No Fit`
- **Uso no projeto**: Treino do classificador binário

### 3. Job Skill Set Dataset
- **Fonte**: Kaggle — dataset de habilidades por cargo
- **Conteúdo**: Mapeamento `cargo → lista de skills exigidas`
- **Uso no projeto**: Comparação de skills para identificar gaps e gerar plano de desenvolvimento

---

## Estrutura de Arquivos Recomendada

```
jobmatch-ai/
├── data/
│   ├── raw/                      # Datasets originais baixados
│   │   ├── linkedin_jobs.csv
│   │   ├── resume_jd_match/      # HuggingFace dataset
│   │   └── job_skills.csv
│   ├── processed/                # Após pipeline de composição
│   │   ├── jobs_clean.parquet
│   │   ├── training_pairs.parquet
│   │   └── skills_map.json
│   └── models/                   # Modelos serializados
│       ├── tfidf_vectorizer.pkl
│       ├── classifier.pkl
│       └── salary_regressor.pkl
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── load_data.py          # Download e carregamento dos datasets
│   │   ├── compose_datasets.py   # Composição e merge dos 3 datasets
│   │   └── preprocess.py         # Limpeza e normalização de texto
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vectorizer.py         # TF-IDF fitting e transform
│   │   ├── classifier.py         # Treino Fit/No Fit
│   │   ├── recommender.py        # Cosine similarity + ranking Top-5
│   │   └── salary_model.py       # Regressão de faixa salarial
│   ├── skills/
│   │   ├── __init__.py
│   │   └── skills_analyzer.py    # Gap analysis e plano de desenvolvimento
│   └── app/
│       ├── __init__.py
│       └── streamlit_app.py      # Frontend Streamlit
├── notebooks/
│   ├── 01_eda.ipynb              # Exploração dos dados
│   ├── 02_pipeline_composicao.ipynb
│   ├── 03_modelo_classificador.ipynb
│   └── 04_modelo_salario.ipynb
├── requirements.txt
└── README.md
```

---

## Implementação por Módulo

### MÓDULO 1 — Carregamento e Composição dos Datasets

**Arquivo**: `src/pipeline/compose_datasets.py`

> Consulte `references/pipeline-dados.md` para o passo-a-passo completo de composição.

Resumo do fluxo:
1. Baixar datasets (Kaggle API + HuggingFace `datasets`)
2. Padronizar schema: normalizar nomes de colunas, unificar campo de `skills`
3. Normalizar salários (converter pay_period → salário anual em USD)
4. Criar coluna `full_text = title + description + skills_desc` para vetorização
5. Merge com `job_skills.csv` via `title` (fuzzy match com `rapidfuzz`)
6. Salvar como Parquet em `data/processed/`

---

### MÓDULO 2 — Preprocessamento NLP

**Arquivo**: `src/pipeline/preprocess.py`

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Limpa e normaliza texto de currículo ou vaga."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in text.split()
        if t not in STOP_WORDS and len(t) > 2
    ]
    return ' '.join(tokens)
```

---

### MÓDULO 3 — Vetorização TF-IDF

**Arquivo**: `src/models/vectorizer.py`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib, os

VECTORIZER_PATH = "data/models/tfidf_vectorizer.pkl"

def fit_vectorizer(corpus: list[str], max_features: int = 15_000) -> TfidfVectorizer:
    """Treina o vetorizador TF-IDF no corpus de vagas."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # escala logarítmica de TF
        min_df=3,              # remove termos muito raros
        max_df=0.85,           # remove termos muito comuns
    )
    vec.fit(corpus)
    joblib.dump(vec, VECTORIZER_PATH)
    return vec

def load_vectorizer() -> TfidfVectorizer:
    return joblib.load(VECTORIZER_PATH)

def transform(texts: list[str], vectorizer: TfidfVectorizer):
    return vectorizer.transform(texts)
```

---

### MÓDULO 4 — Classificador Fit/No Fit

**Arquivo**: `src/models/classifier.py`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import joblib

CLASSIFIER_PATH = "data/models/classifier.pkl"

# Modelos candidatos — avaliar com cross-validation
CANDIDATES = {
    "logistic_regression": LogisticRegression(C=1.0, max_iter=500, class_weight='balanced'),
    "random_forest":       RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1),
    "svm":                 LinearSVC(C=0.5, class_weight='balanced', max_iter=2000),
}

def train_best(X_train, y_train, cv: int = 5) -> tuple:
    """Seleciona e treina o melhor classificador via cross-val F1."""
    best_name, best_score, best_model = None, 0, None
    for name, model in CANDIDATES.items():
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
        print(f"  {name}: F1={score:.4f}")
        if score > best_score:
            best_score, best_name, best_model = score, name, model
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, CLASSIFIER_PATH)
    print(f"\n✅ Melhor modelo: {best_name} (F1={best_score:.4f})")
    return best_name, best_model

def predict(resume_vec, classifier) -> tuple[str, float]:
    """Retorna ('Fit'|'No Fit', probabilidade)."""
    label = classifier.predict(resume_vec)[0]
    # LinearSVC não tem predict_proba — usar decision_function normalizado
    if hasattr(classifier, 'predict_proba'):
        prob = classifier.predict_proba(resume_vec).max()
    else:
        from scipy.special import expit
        prob = float(expit(classifier.decision_function(resume_vec)[0]))
    return ("Fit" if label == 1 else "No Fit"), round(prob, 4)
```

---

### MÓDULO 5 — Recommender (Top-5 por Cosine Similarity)

**Arquivo**: `src/models/recommender.py`

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def rank_jobs(
    resume_vec,          # sparse matrix (1, vocab)
    jobs_matrix,         # sparse matrix (n_jobs, vocab)
    jobs_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Calcula similaridade cosseno entre currículo e todas as vagas.
    Retorna DataFrame com Top-K vagas e score de aderência.
    """
    scores = cosine_similarity(resume_vec, jobs_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    result = jobs_df.iloc[top_indices].copy()
    result['adherence_score'] = np.round(scores[top_indices] * 100, 1)  # em %
    result['fit_label'] = result['adherence_score'].apply(
        lambda s: "✅ Fit" if s >= 40 else "❌ No Fit"
    )
    return result[['title', 'company_name', 'location', 'adherence_score', 'fit_label',
                    'min_salary_annual', 'max_salary_annual', 'skills_desc']]
```

> **Nota sobre threshold Fit/No Fit**: O valor 40% é um ponto de partida.
> Calibre via curva ROC sobre o conjunto de validação do Resume-JD-Match.

---

### MÓDULO 6 — Skills Analyzer (Gap Analysis)

**Arquivo**: `src/skills/skills_analyzer.py`

```python
import json, re
from pathlib import Path

SKILLS_MAP_PATH = Path("data/processed/skills_map.json")

def extract_skills_from_text(text: str, skills_vocabulary: set) -> set:
    """Extrai skills mencionadas no texto por matching de vocabulário."""
    text_lower = text.lower()
    found = set()
    for skill in skills_vocabulary:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)
    return found

def analyze_gap(resume_text: str, job_title: str) -> dict:
    """
    Compara skills do currículo com as exigidas pela vaga.
    Retorna: {compatible, missing, development_plan}
    """
    with open(SKILLS_MAP_PATH) as f:
        skills_map = json.load(f)
    
    # Normaliza título para buscar no mapa
    norm_title = job_title.lower().strip()
    required = set(skills_map.get(norm_title, []))
    
    if not required:
        return {"compatible": [], "missing": [], "development_plan": []}
    
    all_skills = set(skills_map.values().__iter__().__next__())  # vocabulário global
    candidate_skills = extract_skills_from_text(resume_text, required)
    missing = required - candidate_skills
    
    return {
        "compatible": sorted(candidate_skills),
        "missing": sorted(missing),
        "development_plan": _generate_plan(missing),
    }

def _generate_plan(missing_skills: set) -> list[dict]:
    """Gera sugestão de desenvolvimento para cada skill faltante."""
    RESOURCES = {
        "python": {"curso": "Python for Everybody (Coursera)", "tempo": "~6 semanas"},
        "sql": {"curso": "SQL for Data Science (Coursera)", "tempo": "~4 semanas"},
        "machine learning": {"curso": "ML Specialization - Andrew Ng (Coursera)", "tempo": "~3 meses"},
        "power bi": {"curso": "Microsoft Power BI Desktop (Udemy)", "tempo": "~3 semanas"},
        "docker": {"curso": "Docker & Kubernetes: The Practical Guide (Udemy)", "tempo": "~4 semanas"},
    }
    plan = []
    for skill in sorted(missing_skills):
        resource = RESOURCES.get(skill.lower(), {
            "curso": f"Buscar curso de {skill} no Coursera/Udemy",
            "tempo": "~2-6 semanas"
        })
        plan.append({"skill": skill, **resource})
    return plan
```

---

### MÓDULO 7 — Regressor de Salário

**Arquivo**: `src/models/salary_model.py`

```python
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib
import numpy as np

SALARY_MODEL_PATH = "data/models/salary_regressor.pkl"

def train_salary_model(X_train, y_train) -> object:
    """
    Treina regressor de salário anual (em USD) com base no vetor TF-IDF da vaga.
    Target: salário médio anual = (min_salary_annual + max_salary_annual) / 2
    """
    candidates = {
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05),
        "random_forest":     RandomForestRegressor(n_estimators=200, n_jobs=-1),
    }
    best_name, best_score, best_model = None, float('inf'), None
    for name, model in candidates.items():
        rmse = -cross_val_score(model, X_train, y_train, cv=5,
                                scoring='neg_root_mean_squared_error').mean()
        print(f"  {name}: RMSE=${rmse:,.0f}")
        if rmse < best_score:
            best_score, best_name, best_model = rmse, name, model
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, SALARY_MODEL_PATH)
    print(f"\n✅ Melhor regressor: {best_name} (RMSE=${best_score:,.0f})")
    return best_model

def predict_salary_range(job_vec, model) -> dict:
    """Prediz faixa salarial com intervalo de ±15%."""
    pred = float(model.predict(job_vec)[0])
    return {
        "estimated_annual_usd": round(pred),
        "range_low":  round(pred * 0.85),
        "range_high": round(pred * 1.15),
    }
```

---

### MÓDULO 8 — Frontend Streamlit

**Arquivo**: `src/app/streamlit_app.py`

> Consulte `references/streamlit-ui.md` para o código completo do frontend.

**Fluxo de telas**:
1. **Sidebar** — Configurações (top-k, threshold de Fit)
2. **Área principal**:
   - `st.text_area` para o usuário colar o currículo/perfil
   - Botão "🔍 Analisar Compatibilidade"
   - Loading spinner durante processamento
3. **Resultados**:
   - Métrica grande: Score médio de aderência
   - Badge Fit / No Fit
   - Tabela das Top-5 vagas com scores e salários
   - Expander com Skills Compatíveis ✅ e Skills Faltantes ❌
   - Plano de desenvolvimento em cards

---

## Pipeline de Treinamento (Script Principal)

**Arquivo**: `train_pipeline.py` (na raiz do projeto)

```python
"""
Executa toda a pipeline de treino do JobMatch AI.
Uso: python train_pipeline.py
"""
import pandas as pd
from src.pipeline.load_data import load_all
from src.pipeline.compose_datasets import compose
from src.pipeline.preprocess import clean_text
from src.models.vectorizer import fit_vectorizer, transform
from src.models.classifier import train_best
from src.models.salary_model import train_salary_model
from sklearn.model_selection import train_test_split

def main():
    print("📦 1/5 — Carregando datasets...")
    raw = load_all()

    print("🔧 2/5 — Compondo e limpando datasets...")
    jobs_df, pairs_df = compose(raw)

    print("🔤 3/5 — Vetorizando com TF-IDF...")
    corpus = jobs_df['full_text'].tolist()
    vec = fit_vectorizer(corpus)
    
    # Vetorizar pares para classificador
    pairs_df['resume_clean'] = pairs_df['resume'].apply(clean_text)
    pairs_df['job_clean']    = pairs_df['job_description'].apply(clean_text)
    combined = (pairs_df['resume_clean'] + ' ' + pairs_df['job_clean']).tolist()
    X = transform(combined, vec)
    y = (pairs_df['label'] == 'Fit').astype(int).values

    print("🤖 4/5 — Treinando classificador Fit/No Fit...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    _, clf = train_best(X_train, y_train)

    print("💰 5/5 — Treinando regressor de salário...")
    jobs_with_salary = jobs_df.dropna(subset=['salary_annual_avg'])
    X_sal = transform(jobs_with_salary['full_text'].tolist(), vec)
    y_sal = jobs_with_salary['salary_annual_avg'].values
    train_salary_model(X_sal, y_sal)

    print("\n🎉 Pipeline concluída! Modelos salvos em data/models/")

if __name__ == "__main__":
    main()
```

---

## Requirements

```txt
# requirements.txt
# Core ML
scikit-learn>=1.4
numpy>=1.26
pandas>=2.2
scipy>=1.12
joblib>=1.3

# NLP
nltk>=3.8

# Data pipeline
kaggle>=1.6
datasets>=2.18        # HuggingFace
pyarrow>=15.0
rapidfuzz>=3.6        # fuzzy match para títulos de cargo
pyarrow>=15.0

# Frontend
streamlit>=1.33

# Utilitários
python-dotenv>=1.0
tqdm>=4.66
```

---

## Métricas de Avaliação

| Tarefa | Métrica Principal | Meta Mínima |
|---|---|---|
| Classificação Fit/No Fit | F1-Score (macro) | ≥ 0.75 |
| Ranking Top-5 | NDCG@5 | ≥ 0.70 |
| Regressão Salarial | RMSE | < $15.000 |
| Similaridade | Precisão@5 | ≥ 0.60 |

---

## Referências Detalhadas

- `references/pipeline-dados.md` — Passo-a-passo completo de composição dos 3 datasets
- `references/streamlit-ui.md` — Código completo do frontend Streamlit com todos os componentes

Leia estes arquivos quando for implementar os módulos correspondentes.
