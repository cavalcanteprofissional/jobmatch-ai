"""
JobMatch AI — Dashboard de Monitoramento (Modelo ML + API)

Uso: streamlit run src/app/monitor_dashboard.py

Abas:
  - Modelo ML: métricas de classificação (Fit x No Fit) e regressão salarial
  - API: métricas de uso da API (requisições, latência, erros) se FastAPI estiver rodando
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

METRICS_PATH = Path("data/models/metrics.json")
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="JobMatch AI — Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📊 JobMatch AI — Monitor")
st.markdown("---")


def load_ml_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def fetch_api_metrics() -> dict | None:
    try:
        import httpx
        resp = httpx.get(f"{API_URL}/metrics", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ================================================================
# ABA 1: Modelo ML
# ================================================================
def render_ml_tab(metrics: dict) -> None:
    st.subheader("🔍 Classificação Fit x No Fit")
    clf = metrics["classification"]
    cm = clf["confusion_matrix"]

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Modelo", clf["model_type"])
    cc2.metric("Acurácia", f"{clf['accuracy']:.2%}")
    cc3.metric("F1-Score", f"{clf['f1_score']:.2%}")
    cc4.metric("Amostras", clf["test_samples"])

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Precisão", f"{clf['precision']:.2%}")
    cc2.metric("Recall", f"{clf['recall']:.2%}")
    cc3.metric("VN", cm[0][0])
    cc4.metric("FP", cm[0][1])

    cc1, cc2 = st.columns(2)
    cc1.metric("FN", cm[1][0])
    cc2.metric("VP", cm[1][1])

    st.markdown("**Matriz de Confusão**")
    cm_df = pd.DataFrame(
        cm,
        index=["No Fit (Real)", "Fit (Real)"],
        columns=["No Fit (Previsto)", "Fit (Previsto)"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("---")
    st.subheader("💰 Regressão Salarial")
    reg = metrics["regression"]
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Modelo", reg["model_type"])
    rc2.metric("RMSE", f"${reg['rmse']:,.2f}")
    rc3.metric("MAE", f"${reg['mae']:,.2f}")
    rc4.metric("R²", f"{reg['r2']:.2%}")
    st.metric("Amostras de Teste", reg["test_samples"])

    st.markdown("---")
    st.subheader("⚙️ Informações do Modelo")
    info = metrics["model_info"]
    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("Features TF-IDF", f"{info['vectorizer_features']:,}")
    ic2.metric("Total de Vagas", f"{info['total_jobs']:,}")
    ic3.metric("Vagas c/ Salário", f"{info['jobs_with_salary']:,}")
    ic4.metric("Pares de Treino", f"{info['training_pairs']:,}")


# ================================================================
# ABA 2: API
# ================================================================
def render_api_tab(api_metrics: dict) -> None:
    if not api_metrics:
        st.warning(
            "API não disponível. "
            "Execute `poetry run uvicorn src.api.server:app --host 0.0.0.0 --port 8000` "
            "para ativar o monitoramento."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Requisições Totais", api_metrics["total_requests"])
    with c2:
        st.metric("Erros Totais", api_metrics["total_errors"])
    with c3:
        st.metric("Taxa de Erro", f"{api_metrics['error_rate_pct']}%")
    with c4:
        uptime = api_metrics["uptime_seconds"]
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        st.metric("Uptime", f"{int(hours)}h {int(minutes)}m {int(seconds)}s")

    st.markdown("---")
    st.subheader("📈 Métricas por Endpoint")
    if api_metrics["endpoints"]:
        rows = []
        for ep, data in api_metrics["endpoints"].items():
            rows.append({
                "Endpoint": ep,
                "Requisições": data["requests"],
                "Erros": data["errors"],
                "Latência Média (ms)": data["latency_ms_avg"],
                "Latência P99 (ms)": data["latency_ms_p99"],
                "Latência Min (ms)": data["latency_ms_min"],
                "Latência Max (ms)": data["latency_ms_max"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("📉 Latência por Endpoint (ms)")
        chart_data = df.set_index("Endpoint")[
            ["Latência Média (ms)", "Latência P99 (ms)",
             "Latência Min (ms)", "Latência Max (ms)"]
        ]
        st.bar_chart(chart_data)
    else:
        st.info("Nenhuma requisição registrada ainda.")


# ================================================================
# RENDER
# ================================================================

tab_ml, tab_api = st.tabs(["📊 Modelo ML", "⚡ API"])

with tab_ml:
    ml_metrics = load_ml_metrics()
    if ml_metrics:
        render_ml_tab(ml_metrics)
    else:
        st.warning(
            "Arquivo data/models/metrics.json não encontrado. "
            "Execute o treino dos modelos primeiro."
        )

with tab_api:
    api_metrics = fetch_api_metrics()
    render_api_tab(api_metrics)
