"""
Re-treina modelos e gera dados de avaliacao para o dashboard.
Usa nested cross-validation + hyperparameter tuning.
Roda em ~2-5 minutos (dados ja pre-processados).
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.model_selection import train_test_split

from src.models.classifier import train_nested_cv_clf, DENSE_ONLY_MODELS as CLF_DENSE
from src.models.salary_model import train_nested_cv_reg, DENSE_ONLY_MODELS as REG_DENSE
from src.models.vectorizer import SentenceBertVectorizer, load_sbert_vectorizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _ts():
    return datetime.now().strftime("%H:%M:%S")

# ── 1. Classificacao (Fit x No Fit) ──────────────────────────────
print(f"[{_ts()}] [1/5] Carregando pares curriculo-vaga...")
pairs = pd.read_parquet(RAW_DIR / "resume_jd_train.parquet")
label_map = {"Good Fit": 1, "Potential Fit": 1, "No Fit": 0}
pairs["label_bin"] = pairs["label"].map(label_map)
pairs["text"] = pairs["resume"].fillna("") + " " + pairs["job_description"].fillna("")

X_text = pairs["text"].tolist()
y = pairs["label_bin"].values

print(f"  -> {len(pairs)} pares, {y.sum()} Fit / {len(y)-y.sum()} No Fit")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── 1a. TF-IDF path ──────────────────────────────────────────────
print(f"[{_ts()}] [2/5] Vetorizando TF-IDF (15k features)...")
vec = TfidfVectorizer(max_features=15_000, stop_words="english")
X_vec = vec.fit_transform(X_text)

print(f"[{_ts()}] [3/5] Treinando classificador com TF-IDF (nested CV)...")
clf_name, clf_params, clf_cv_scores, clf = train_nested_cv_clf(
    X_vec, y, outer_cv=2, inner_cv=2, n_iter=8, random_state=42,
)

# Holdout test set for final metrics
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, stratify=y, random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
if hasattr(clf, "predict_proba"):
    y_prob = clf.predict_proba(X_test)[:, 1] if clf.classes_[1] == 1 else clf.predict_proba(X_test)[:, 0]
elif hasattr(clf, "decision_function"):
    y_prob = expit(clf.decision_function(X_test))
else:
    y_prob = y_pred.astype(float)

cm = confusion_matrix(y_test, y_pred).tolist()
clf_metrics = {
    "vectorizer": "tfidf",
    "model_type": type(clf).__name__,
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    "precision": round(float(precision_score(y_test, y_pred)), 4),
    "recall": round(float(recall_score(y_test, y_pred)), 4),
    "test_samples": int(len(y_test)),
    "confusion_matrix": cm,
    "nested_cv": {
        "scores": [round(s, 4) for s in clf_cv_scores],
        "mean": round(float(np.mean(clf_cv_scores)), 4),
        "std": round(float(np.std(clf_cv_scores)), 4),
    },
    "best_params": clf_params,
    "best_candidate": clf_name,
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
print(f"  -> Nested CV F1: {clf_cv_scores}")

# ── 1b. Sentence-BERT path (todos os modelos, incluindo MLP/GaussianNB) ──
print(f"[{_ts()}] [3b/5] Treinando classificador com Sentence-BERT...")
try:
    sbert = SentenceBertVectorizer()
    sbert.fit(X_text)
    X_dense = sbert.transform(X_text)
    clf_sbert_name, clf_sbert_params, clf_sbert_scores, clf_sbert = train_nested_cv_clf(
        X_dense, y, outer_cv=2, inner_cv=2, n_iter=8, random_state=42,
    )
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_dense, y, test_size=0.2, stratify=y, random_state=42
    )
    clf_sbert.fit(X_tr_s, y_tr_s)
    yp_s = clf_sbert.predict(X_te_s)
    clf_metrics["sbert"] = {
        "model_type": type(clf_sbert).__name__,
        "accuracy": round(float(accuracy_score(y_te_s, yp_s)), 4),
        "f1_score": round(float(f1_score(y_te_s, yp_s)), 4),
        "nested_cv_mean": round(float(np.mean(clf_sbert_scores)), 4),
        "best_candidate": clf_sbert_name,
    }
    joblib.dump(sbert, MODELS_DIR / "sentence_bert.pkl")
    joblib.dump(clf_sbert, MODELS_DIR / "classifier_sbert.pkl")
    X_jobs_sbert = sbert.transform(jobs["full_text"].fillna("").tolist())
    np.save(MODELS_DIR / "jobs_sbert_embeddings.npy", X_jobs_sbert)
    print(f"  -> SBERT classifier: {clf_sbert_name} F1={clf_metrics['sbert']['f1_score']:.4f}")
    print(f"  -> classifier_sbert.pkl + jobs_sbert_embeddings.npy salvos")
except Exception as e:
    logger.warning("Sentence-BERT path falhou: %s", e)

# ── 2. Regressao Salarial ────────────────────────────────────────
print(f"[{_ts()}] [4/5] Treinando regressao salarial (nested CV)...")
jobs = pd.read_parquet(PROCESSED_DIR / "jobs_clean.parquet")
has_salary = jobs.dropna(subset=["salary_annual_avg"])
X_sal = vec.transform(has_salary["full_text"].fillna(""))
y_sal = has_salary["salary_annual_avg"].values

print(f"  -> {len(has_salary)} vagas com salario")
reg_name, reg_params, reg_cv_scores, sal = train_nested_cv_reg(
    X_sal, y_sal, outer_cv=3, inner_cv=2, n_iter=6, random_state=42,
)

Xst, Xste, yst, yste = train_test_split(
    X_sal, y_sal, test_size=0.2, random_state=42
)
sal.fit(Xst, yst)
ysp = sal.predict(Xste)

reg_metrics = {
    "model_type": type(sal).__name__,
    "rmse": round(float(np.sqrt(mean_squared_error(yste, ysp))), 2),
    "mae": round(float(mean_absolute_error(yste, ysp)), 2),
    "r2": round(float(r2_score(yste, ysp)), 4),
    "test_samples": int(len(yste)),
    "nested_cv": {
        "scores": [round(s, 2) for s in reg_cv_scores],
        "mean": round(float(np.mean(reg_cv_scores)), 2),
        "std": round(float(np.std(reg_cv_scores)), 2),
    },
    "best_params": reg_params,
    "best_candidate": reg_name,
}

eval_reg = pd.DataFrame({"y_true": yste, "y_pred": ysp})
eval_reg.to_parquet(MODELS_DIR / "eval_reg.parquet", index=False)
print(f"  -> eval_reg.parquet ({len(eval_reg)} linhas)")
print(f"  -> Nested CV RMSE: {reg_cv_scores}")

# ── 3. Salvar ────────────────────────────────────────────────────
print(f"[{_ts()}] [5/5] Salvando modelos e metricas...")

joblib.dump(vec, MODELS_DIR / "tfidf_vectorizer.pkl")
joblib.dump(clf, MODELS_DIR / "classifier.pkl")
joblib.dump(sal, MODELS_DIR / "salary_regressor.pkl")

X_jobs = vec.transform(jobs["full_text"].fillna(""))
save_npz(MODELS_DIR / "jobs_matrix.npz", X_jobs.tocsr())
print(f"  -> jobs_matrix.npz ({X_jobs.shape})")

metrics = {
    "classification": clf_metrics,
    "regression": reg_metrics,
    "model_info": {
        "vectorizer": "tfidf",
        "total_jobs": len(jobs),
        "jobs_with_salary": len(has_salary),
        "training_pairs": len(pairs),
    },
}
with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

for p in ["classifier.pkl", "tfidf_vectorizer.pkl", "salary_regressor.pkl",
          "jobs_matrix.npz", "eval_clf.parquet", "eval_reg.parquet", "metrics.json"]:
    f = MODELS_DIR / p
    print(f"  -> {p}: {f.stat().st_size/1024:.1f} KB" if f.exists() else f"  -> {p}: NOT FOUND")

print("OK")
