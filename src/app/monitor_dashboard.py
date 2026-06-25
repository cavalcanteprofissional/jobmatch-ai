"""
JobMatch AI — Streamlit Monitoring Dashboard

Uso: streamlit run src/app/monitor_dashboard.py

Exibe métricas de uso da API: requisições, latência, erros.
"""

import os
from pathlib import Path

import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="JobMatch AI — Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def fetch_metrics() -> dict | None:
    try:
        import httpx
        resp = httpx.get(f"{API_URL}/metrics", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Erro ao conectar na API: {e}")
        return None


st.title("📊 JobMatch AI — Dashboard de Monitoramento")
st.markdown("---")

metrics = fetch_metrics()

if not metrics:
    st.warning(
        "API não disponível. Certifique-se de que o servidor FastAPI está rodando "
        "em %s", API_URL,
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Requisições Totais", metrics["total_requests"])
with c2:
    st.metric("Erros Totais", metrics["total_errors"])
with c3:
    st.metric("Taxa de Erro", f"{metrics['error_rate_pct']}%")
with c4:
    uptime = metrics["uptime_seconds"]
    hours, remainder = divmod(uptime, 3600)
    minutes, seconds = divmod(remainder, 60)
    st.metric("Uptime", f"{int(hours)}h {int(minutes)}m {int(seconds)}s")

st.markdown("---")
st.subheader("📈 Métricas por Endpoint")

if metrics["endpoints"]:
    rows = []
    for ep, data in metrics["endpoints"].items():
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
        ["Latência Média (ms)", "Latência P99 (ms)", "Latência Min (ms)", "Latência Max (ms)"]
    ]
    st.bar_chart(chart_data)
else:
    st.info("Nenhuma requisição registrada ainda.")
