"""
Re-treina modelos e gera dados de avaliação para o dashboard.
Roda em segundos (dados já pré-processados).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from scipy.sparse import save_npz, vstack, csr_matrix

DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Classificação (Fit x No Fit) ──────────────────────────────
print("[1/5] Carregando pares currículo-vaga...")
pairs = pd.read_parquet(RAW_DIR / "resume_jd_train.parquet")
label_map = {"Good Fit": 1, "Potential Fit": 1, "No Fit": 0}
pairs["label_bin"] = pairs["label"].map(label_map)
pairs["text"] = pairs["resume"].fillna("") + " " + pairs["job_description"].fillna("")

X_text = pairs["text"].tolist()
y = pairs["label_bin"].values

print(f"  -> {len(pairs)} pares, {y.sum()} Fit / {len(y)-y.sum()} No Fit")

print("[2/5] Vetorizando TF-IDF (15k features)...")
vec = TfidfVectorizer(max_features=15_000, stop_words="english")
X_vec = vec.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=42
)

print("[3/5] Treinando LogisticRegression...")
clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.decision_function(X_test)

cm = confusion_matrix(y_test, y_pred).tolist()
clf_metrics = {
    "model_type": type(clf).__name__,
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    "precision": round(float(precision_score(y_test, y_pred)), 4),
    "recall": round(float(recall_score(y_test, y_pred)), 4),
    "test_samples": int(len(y_test)),
    "confusion_matrix": cm,
}

eval_clf = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "y_prob": y_prob,
})
eval_clf["y_true_label"] = eval_clf["y_true"].map({1: "Fit", 0: "No Fit"})
eval_clf["y_pred_label"] = eval_clf["y_pred"].map({1: "Fit", 0: "No Fit"})
eval_clf.to_parquet(MODELS_DIR / "eval_clf.parquet", index=False)
print(f"  -> eval_clf.parquet ({len(eval_clf)} linhas)")

# ── 2. Regressão Salarial ────────────────────────────────────────
print("[4/5] Treinando regressão salarial...")
jobs = pd.read_parquet(PROCESSED_DIR / "jobs_clean.parquet")
has_salary = jobs.dropna(subset=["salary_annual_avg"])
X_sal = vec.transform(has_salary["full_text"].fillna(""))
y_sal = has_salary["salary_annual_avg"].values

Xst, Xste, yst, yste = train_test_split(
    X_sal, y_sal, test_size=0.2, random_state=42
)

sal = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
sal.fit(Xst, yst)
ysp = sal.predict(Xste)

reg_metrics = {
    "model_type": type(sal).__name__,
    "rmse": round(float(np.sqrt(mean_squared_error(yste, ysp))), 2),
    "mae": round(float(mean_absolute_error(yste, ysp)), 2),
    "r2": round(float(r2_score(yste, ysp)), 4),
    "test_samples": int(len(yste)),
}

eval_reg = pd.DataFrame({"y_true": yste, "y_pred": ysp})
eval_reg.to_parquet(MODELS_DIR / "eval_reg.parquet", index=False)
print(f"  -> eval_reg.parquet ({len(eval_reg)} linhas)")

# ── 3. Salvar ────────────────────────────────────────────────────
print("[5/5] Salvando modelos e métricas...")

joblib.dump(vec, MODELS_DIR / "tfidf_vectorizer.pkl")
joblib.dump(clf, MODELS_DIR / "classifier.pkl")
joblib.dump(sal, MODELS_DIR / "salary_regressor.pkl")

# jobs_matrix para o dashboard de vagas
X_jobs = vec.transform(jobs["full_text"].fillna(""))
save_npz(MODELS_DIR / "jobs_matrix.npz", X_jobs.tocsr())
print(f"  -> jobs_matrix.npz ({X_jobs.shape})")

metrics = {
    "classification": clf_metrics,
    "regression": reg_metrics,
    "model_info": {
        "vectorizer_features": vec.max_features,
        "total_jobs": len(jobs),
        "jobs_with_salary": len(has_salary),
        "training_pairs": len(pairs),
    },
}
with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# Tamanhos
for p in ["classifier.pkl", "tfidf_vectorizer.pkl", "salary_regressor.pkl", "jobs_matrix.npz",
          "eval_clf.parquet", "eval_reg.parquet", "metrics.json"]:
    f = MODELS_DIR / p
    print(f"  -> {p}: {f.stat().st_size/1024:.1f} KB")

print("OK")
