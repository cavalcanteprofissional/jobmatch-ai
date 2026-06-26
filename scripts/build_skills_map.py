"""
Script rápido para construir skills_map.json com matching exato.
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

print("Carregando e mergeando datasets de skills...")
skills_raw = pd.read_csv(RAW_DIR / "job_skills.csv", low_memory=False)
linkedin = pd.read_csv(
    RAW_DIR / "linkedin_job_postings.csv",
    usecols=["job_link", "job_title"],
    low_memory=False,
)
skills_df = skills_raw.merge(linkedin, on="job_link", how="left")
skills_df = skills_df.rename(columns={"job_skills": "skills"})

skill_titles_set = set(skills_df["job_title"].dropna().str.lower().unique())
print(f"Skills: {len(skill_titles_set):,} títulos únicos")

jobs_df = pd.read_parquet(PROCESSED_DIR / "jobs_clean.parquet")
unique_titles = jobs_df["title"].dropna().unique().tolist()
print(f"Jobs: {len(unique_titles):,} títulos únicos")

# Exact match
matched = [t for t in unique_titles if t.lower() in skill_titles_set]
print(f"Match exato: {len(matched)} títulos")

# Build map
skill_lookup = (
    skills_df.dropna(subset=["job_title"])
    .copy()
    .drop_duplicates(subset="job_title")
)
skill_lookup["key"] = skill_lookup["job_title"].str.lower()

skills_map = {}
for title in matched:
    row = skill_lookup[skill_lookup["key"] == title.lower()]
    if row.empty:
        continue
    raw = row.iloc[0]["skills"]
    if isinstance(raw, str):
        skills = [s.strip().lower() for s in raw.split(",") if s.strip()]
    elif isinstance(raw, list):
        skills = [s.strip().lower() for s in raw if s]
    else:
        skills = []
    if skills:
        skills_map[title.lower()] = skills

with open(PROCESSED_DIR / "skills_map.json", "w", encoding="utf-8") as f:
    json.dump(skills_map, f, ensure_ascii=False, indent=2)
print(f"Skills map salvo: {len(skills_map)} títulos, {sum(len(v) for v in skills_map.values())} skills")

jobs_df["required_skills"] = jobs_df["title"].str.lower().map(
    lambda t: skills_map.get(t, [])
)
n_with = (jobs_df["required_skills"].str.len() > 0).sum()
print(f"Vagas com skills: {n_with}/{len(jobs_df)} ({n_with/len(jobs_df)*100:.1f}%)")
jobs_df.to_parquet(PROCESSED_DIR / "jobs_clean.parquet", index=False)

all_skills = set()
for sl in skills_map.values():
    all_skills.update(sl)
print(f"\nSkills únicas: {len(all_skills):,}")
skill_counter = Counter()
for sl in skills_map.values():
    skill_counter.update(sl)
print("Top 20:")
for sk, ct in skill_counter.most_common(20):
    print(f"  {sk}: {ct}")
