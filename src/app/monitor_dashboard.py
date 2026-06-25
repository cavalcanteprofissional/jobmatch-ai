"""
JobMatch AI — Dashboard de Métricas dos Modelos ML

Uso: streamlit run src/app/monitor_dashboard.py

Exibe métricas de classificação (Fit x No Fit) e regressão salarial
calculadas durante o treino e salvas em data/models/metrics.json.
NÃO depende da API FastAPI.
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd

METRICS_PATH = Path("data/models/metrics.json")


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


st.set_page_config(
    page_title="JobMatch AI — Métricas dos Modelos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📊 JobMatch AI — Métricas dos Modelos ML")
st.markdown("*Métricas calculadas no conjunto de teste durante o treino.*")
st.markdown("---")

metrics = load_metrics()

if not metrics:
    st.warning(
        "Arquivo data/models/metrics.json não encontrado. "
        "Execute o treino dos modelos primeiro."
    )
    st.stop()

# ===== CLASSIFICAÇÃO =====
st.subheader("🔍 Classificação Fit x No Fit")

cm = metrics["classification"]["confusion_matrix"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo", metrics["classification"]["model_type"])
c2.metric("Acurácia", f"{metrics['classification']['accuracy']:.2%}")
c3.metric("F1-Score", f"{metrics['classification']['f1_score']:.2%}")
c4.metric("Amostras de Teste", metrics["classification"]["test_samples"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precisão", f"{metrics['classification']['precision']:.2%}")
c2.metric("Recall", f"{metrics['classification']['recall']:.2%}")
c3.metric("Verdadeiros Negativos", cm[0][0])
c4.metric("Falsos Positivos", cm[0][1])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Falsos Negativos", cm[1][0])
c2.metric("Verdadeiros Positivos", cm[1][1])

st.markdown("**Matriz de Confusão**")
cm_df = pd.DataFrame(
    cm,
    index=["No Fit (Real)", "Fit (Real)"],
    columns=["No Fit (Previsto)", "Fit (Previsto)"],
)
st.dataframe(cm_df, use_container_width=True)

st.markdown("---")

# ===== REGRESSÃO =====
st.subheader("💰 Regressão Salarial")

r = metrics["regression"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo", r["model_type"])
c2.metric("RMSE", f"${r['rmse']:,.2f}")
c3.metric("MAE", f"${r['mae']:,.2f}")
c4.metric("R²", f"{r['r2']:.2%}")

c1, c2 = st.columns(2)
c1.metric("Amostras de Teste", r["test_samples"])

st.markdown("---")

# ===== INFORMAÇÕES DO MODELO =====
st.subheader("⚙️ Informações do Modelo")

info = metrics["model_info"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Features do Vetorizador", f"{info['vectorizer_features']:,}")
c2.metric("Total de Vagas", f"{info['total_jobs']:,}")
c3.metric("Vagas com Salário", f"{info['jobs_with_salary']:,}")
c4.metric("Pares de Treino", f"{info['training_pairs']:,}")
