# Pipeline de Composição dos Datasets — JobMatch AI

Este arquivo detalha como baixar, inspecionar e compor os 3 datasets em uma estrutura unificada.

---

## Pré-requisitos

```bash
pip install kaggle datasets pyarrow rapidfuzz pandas tqdm -q

# Configurar credenciais Kaggle
# 1. Acesse https://www.kaggle.com/settings → "API" → "Create New Token"
# 2. Salve kaggle.json em ~/.kaggle/kaggle.json
# 3. chmod 600 ~/.kaggle/kaggle.json
```

---

## Passo 1 — Download dos Datasets

### 1a. LinkedIn Job Postings (Kaggle)

```python
import subprocess, zipfile, os

def download_linkedin_jobs(dest: str = "data/raw/") -> None:
    os.makedirs(dest, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "arshkon/linkedin-job-postings",
        "-p", dest, "--unzip"
    ], check=True)
    print("✅ LinkedIn jobs baixado")
```

Colunas úteis após download:
```
job_id, title, description, skills_desc,
max_salary, min_salary, pay_period,
location, company_name, work_type,
formatted_experience_level
```

### 1b. Resume-JD-Match (HuggingFace)

```python
from datasets import load_dataset

def download_resume_jd(dest: str = "data/raw/") -> None:
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")
    # Salva como parquet para uso posterior
    os.makedirs(dest, exist_ok=True)
    ds['train'].to_parquet(f"{dest}resume_jd_train.parquet")
    if 'test' in ds:
        ds['test'].to_parquet(f"{dest}resume_jd_test.parquet")
    print(f"✅ Resume-JD-Match baixado: {len(ds['train'])} pares de treino")
```

Schema esperado: `resume`, `job_description`, `label` (Fit / No Fit)

### 1c. Job Skill Set (Kaggle)

```bash
# Buscar dataset de skills por cargo no Kaggle:
# Exemplo: "job skills dataset" → verificar o slug exato disponível
kaggle datasets download -d <slug-do-dataset> -p data/raw/ --unzip
```

Schema esperado: `job_title`, `skills` (lista separada por vírgula ou JSON)

---

## Passo 2 — Normalização de Salários

O LinkedIn Job Postings usa `pay_period` com valores como:
`HOURLY`, `DAILY`, `WEEKLY`, `MONTHLY`, `YEARLY`

```python
import pandas as pd
import numpy as np

PERIOD_TO_ANNUAL = {
    'HOURLY':  40 * 52,    # 40h/semana × 52 semanas
    'DAILY':   5 * 52,     # 5 dias/semana × 52 semanas
    'WEEKLY':  52,
    'MONTHLY': 12,
    'YEARLY':  1,
}

def normalize_salary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['pay_period'] = df['pay_period'].str.upper().fillna('YEARLY')
    multiplier = df['pay_period'].map(PERIOD_TO_ANNUAL).fillna(1)
    
    df['min_salary_annual'] = df['min_salary'] * multiplier
    df['max_salary_annual'] = df['max_salary'] * multiplier
    df['salary_annual_avg'] = (df['min_salary_annual'] + df['max_salary_annual']) / 2
    
    # Remover outliers grosseiros
    q_low  = df['salary_annual_avg'].quantile(0.01)
    q_high = df['salary_annual_avg'].quantile(0.99)
    df = df[(df['salary_annual_avg'] >= q_low) & (df['salary_annual_avg'] <= q_high)]
    
    return df
```

---

## Passo 3 — Criação do Campo `full_text`

Este campo é o que será vetorizado pelo TF-IDF. Combine title + description + skills:

```python
from src.pipeline.preprocess import clean_text

def build_full_text(row: pd.Series) -> str:
    parts = [
        str(row.get('title', '')),
        str(row.get('description', '')),
        str(row.get('skills_desc', '')),
    ]
    combined = ' '.join(p for p in parts if p and p != 'nan')
    return clean_text(combined)

def add_full_text(jobs_df: pd.DataFrame) -> pd.DataFrame:
    jobs_df = jobs_df.copy()
    jobs_df['full_text'] = jobs_df.apply(build_full_text, axis=1)
    # Remover vagas com texto muito curto (menos de 30 tokens)
    jobs_df = jobs_df[jobs_df['full_text'].str.split().str.len() >= 30]
    return jobs_df
```

---

## Passo 4 — Merge com Skills Map (Fuzzy Match)

Como os títulos no LinkedIn e no Skills Dataset podem divergir, use fuzzy matching:

```python
from rapidfuzz import process, fuzz
import json

def build_skills_map(jobs_df: pd.DataFrame, skills_df: pd.DataFrame,
                     score_threshold: int = 80) -> dict:
    """
    Para cada título único de vaga no LinkedIn, encontra o título
    mais próximo no Skills Dataset e mapeia as skills.
    """
    skill_titles = skills_df['job_title'].tolist()
    skills_map = {}
    
    unique_titles = jobs_df['title'].dropna().unique().tolist()
    for title in unique_titles:
        match, score, _ = process.extractOne(
            title, skill_titles, scorer=fuzz.token_sort_ratio
        )
        if score >= score_threshold:
            matched_row = skills_df[skills_df['job_title'] == match].iloc[0]
            raw_skills = matched_row.get('skills', '')
            # Normalizar para lista
            if isinstance(raw_skills, str):
                skills_list = [s.strip().lower() for s in raw_skills.split(',') if s.strip()]
            else:
                skills_list = list(raw_skills)
            skills_map[title.lower()] = skills_list
    
    # Salvar para uso posterior
    with open("data/processed/skills_map.json", "w") as f:
        json.dump(skills_map, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Skills map criado: {len(skills_map)} títulos mapeados")
    return skills_map
```

---

## Passo 5 — Composição Final

```python
def compose(raw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entrada: dict com chaves 'linkedin', 'resume_jd', 'job_skills'
    Saída: (jobs_df limpo e enriquecido, pairs_df para classificador)
    """
    # ── Jobs DataFrame ──────────────────────────────────────────────
    jobs_df = raw['linkedin'].copy()
    jobs_df = normalize_salary(jobs_df)
    jobs_df = add_full_text(jobs_df)
    
    # ── Skills Map ──────────────────────────────────────────────────
    skills_map = build_skills_map(jobs_df, raw['job_skills'])
    
    # Adicionar skills como coluna
    jobs_df['required_skills'] = jobs_df['title'].str.lower().map(
        lambda t: skills_map.get(t, [])
    )
    
    # Salvar
    os.makedirs("data/processed", exist_ok=True)
    jobs_df.to_parquet("data/processed/jobs_clean.parquet", index=False)
    
    # ── Pairs DataFrame (para classificador) ────────────────────────
    pairs_df = raw['resume_jd'].copy()
    pairs_df.to_parquet("data/processed/training_pairs.parquet", index=False)
    
    print(f"""
    ✅ Composição concluída:
       • Vagas limpas:     {len(jobs_df):,}
       • Pares treino:     {len(pairs_df):,}
       • Títulos c/ skills: {len(skills_map):,}
    """)
    return jobs_df, pairs_df
```

---

## Passo 6 — Verificação de Qualidade

```python
def quality_check(jobs_df: pd.DataFrame, pairs_df: pd.DataFrame) -> None:
    print("=== QUALITY CHECK ===")
    
    # Jobs
    print(f"\n📊 Jobs DataFrame ({len(jobs_df):,} linhas)")
    print(f"  Nulos em full_text: {jobs_df['full_text'].isna().sum()}")
    print(f"  Com salário:        {jobs_df['salary_annual_avg'].notna().sum():,}")
    print(f"  Com skills:         {(jobs_df['required_skills'].str.len() > 0).sum():,}")
    print(f"  Salário médio:      ${jobs_df['salary_annual_avg'].median():,.0f}/ano")
    
    # Pairs
    print(f"\n📊 Pairs DataFrame ({len(pairs_df):,} linhas)")
    print(f"  Distribuição labels:\n{pairs_df['label'].value_counts()}")
    
    # Verificar balanceamento
    balance = pairs_df['label'].value_counts(normalize=True)
    if balance.min() < 0.30:
        print("  ⚠️  Dataset desbalanceado — considere class_weight='balanced' nos modelos")
```

Rode a verificação ao final de cada composição antes de treinar os modelos.
