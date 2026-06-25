"""
JobMatch AI — Dashboard de Monitoramento (Modelo ML + API)

Uso: streamlit run src/app/monitor_dashboard.py

Abas:
  - Modelo ML: métricas de classificação + regressão + gráficos Altair
  - API: métricas de uso da API (requisições, latência, erros)
"""

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

MODELS_DIR = Path("data/models")
METRICS_PATH = MODELS_DIR / "metrics.json"
EVAL_CLF_PATH = MODELS_DIR / "eval_clf.parquet"
EVAL_REG_PATH = MODELS_DIR / "eval_reg.parquet"
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="JobMatch AI — Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📊 JobMatch AI — Monitor")
st.markdown("---")


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_parquet(path: Path):
    if not path.exists():
        return None
    return pd.read_parquet(path)


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

def confusion_heatmap(cm: list[list[int]]) -> alt.Chart:
    rows = []
    for i, label_real in enumerate(["No Fit", "Fit"]):
        for j, label_pred in enumerate(["No Fit", "Fit"]):
            rows.append({
                "Real": label_real,
                "Previsto": label_pred,
                "Contagem": cm[i][j],
            })
    df = pd.DataFrame(rows)
    total = sum(cm[0]) + sum(cm[1])
    df["Porcentagem"] = (df["Contagem"] / total * 100).round(1)

    heatmap = (
        alt.Chart(df)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("Previsto:O", title=None, sort=None),
            y=alt.Y("Real:O", title=None, sort=None),
            color=alt.Color("Contagem:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Real", "Previsto", "Contagem", alt.Tooltip("Porcentagem:Q", format=".1f")],
        )
        .properties(width=400, height=300, title="Matriz de Confusão")
    )
    text = (
        alt.Chart(df)
        .mark_text(baseline="middle", fontSize=14, fontWeight="bold")
        .encode(
            x="Previsto:O",
            y="Real:O",
            text=alt.Text("Contagem:Q"),
            color=alt.value("white"),
        )
    )
    return heatmap + text


def classification_bars(metrics: dict) -> alt.Chart:
    df = pd.DataFrame({
        "Métrica": ["Acurácia", "F1-Score", "Precisão", "Recall"],
        "Valor": [
            metrics["accuracy"],
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
        ],
    })
    df["Valor %"] = (df["Valor"] * 100).round(1)
    bar = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Métrica:O", sort=None),
            y=alt.Y("Valor:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Métrica:O", scale=alt.Scale(scheme="category10"), legend=None),
            tooltip=["Métrica", alt.Tooltip("Valor %:Q", title="Score (%)")],
        )
        .properties(width=500, height=300, title="Métricas de Classificação")
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [0.7]}))
        .mark_rule(stroke="red", strokeDash=[6, 3], strokeWidth=1.5)
        .encode(y="y:Q")
    )
    label = (
        alt.Chart(pd.DataFrame({"label": ["Meta 70%"]}))
        .mark_text(x=510, y=0.7, align="left", fontSize=11, color="red")
        .encode(y=alt.value(0))
    )
    return bar + rule


def probability_hist(eval_clf: pd.DataFrame) -> alt.Chart:
    df = eval_clf.copy()
    df["Classe"] = df["y_true_label"]
    hist = (
        alt.Chart(df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("y_prob:Q", bin=alt.Bin(maxbins=40), title="Score de Decisão"),
            y=alt.Y("count()", title="Frequência"),
            color=alt.Color("Classe:N", scale=alt.Scale(
                domain=["Fit", "No Fit"],
                range=["#2ca02c", "#d62728"],
            )),
            tooltip=["count()"],
        )
        .properties(width=500, height=300, title="Distribuição dos Scores por Classe Verdadeira")
    )
    return hist


def salary_scatter(eval_reg: pd.DataFrame) -> alt.Chart:
    df = eval_reg.copy()
    pts = (
        alt.Chart(df)
        .mark_circle(opacity=0.5, size=60)
        .encode(
            x=alt.X("y_true:Q", title="Salário Real (R$)"),
            y=alt.Y("y_pred:Q", title="Salário Previsto (R$)"),
            tooltip=[
                alt.Tooltip("y_true:Q", title="Real", format=",.0f"),
                alt.Tooltip("y_pred:Q", title="Previsto", format=",.0f"),
            ],
        )
        .properties(width=500, height=400, title="Salário Real × Previsto")
    )
    line_df = pd.DataFrame({
        "x": [df["y_true"].min(), df["y_true"].max()],
        "y": [df["y_true"].min(), df["y_true"].max()],
    })
    identity = (
        alt.Chart(line_df)
        .mark_line(color="red", strokeDash=[6, 3])
        .encode(x="x:Q", y="y:Q")
    )
    return pts + identity


def residual_hist(eval_reg: pd.DataFrame) -> alt.Chart:
    df = eval_reg.copy()
    df["residual"] = df["y_pred"] - df["y_true"]
    hist = (
        alt.Chart(df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("residual:Q", bin=alt.Bin(maxbins=30), title="Resíduo (Previsto − Real)"),
            y=alt.Y("count()", title="Frequência"),
            tooltip=["count()"],
        )
        .properties(width=500, height=300, title="Distribuição dos Resíduos")
    )
    return hist


def render_ml_tab(metrics: dict, eval_clf: pd.DataFrame | None, eval_reg: pd.DataFrame | None) -> None:
    clf = metrics["classification"]
    reg = metrics["regression"]
    info = metrics["model_info"]

    # ── KPIs ──
    st.subheader("🔍 Classificação Fit x No Fit")
    col = st.columns(6)
    col[0].metric("Modelo", clf["model_type"])
    col[1].metric("Acurácia", f"{clf['accuracy']:.2%}")
    col[2].metric("F1-Score", f"{clf['f1_score']:.2%}")
    col[3].metric("Precisão", f"{clf['precision']:.2%}")
    col[4].metric("Recall", f"{clf['recall']:.2%}")
    col[5].metric("Amostras", clf["test_samples"])

    # ── Matriz de Confusão + Barras de Métricas ──
    c1, c2 = st.columns(2)
    with c1:
        cm = clf["confusion_matrix"]
        heat = confusion_heatmap(cm)
        st.altair_chart(heat, use_container_width=True)

    with c2:
        bars = classification_bars(clf)
        st.altair_chart(bars, use_container_width=True)

    # ── Histograma de Scores ──
    if eval_clf is not None:
        st.markdown("---")
        c1, _ = st.columns([1, 1])
        with c1:
            hist = probability_hist(eval_clf)
            st.altair_chart(hist, use_container_width=True)

    # ── Regressão ──
    st.markdown("---")
    st.subheader("💰 Regressão Salarial")
    col = st.columns(5)
    col[0].metric("Modelo", reg["model_type"])
    col[1].metric("RMSE", f"${reg['rmse']:,.2f}")
    col[2].metric("MAE", f"${reg['mae']:,.2f}")
    col[3].metric("R²", f"{reg['r2']:.2%}")
    col[4].metric("Amostras", reg["test_samples"])

    if eval_reg is not None:
        c1, c2 = st.columns(2)
        with c1:
            scatter = salary_scatter(eval_reg)
            st.altair_chart(scatter, use_container_width=True)
        with c2:
            resid = residual_hist(eval_reg)
            st.altair_chart(resid, use_container_width=True)

    # ── Info ──
    st.markdown("---")
    st.subheader("⚙️ Informações do Modelo")
    col = st.columns(4)
    col[0].metric("Features TF-IDF", f"{info['vectorizer_features']:,}")
    col[1].metric("Total de Vagas", f"{info['total_jobs']:,}")
    col[2].metric("Vagas c/ Salário", f"{info['jobs_with_salary']:,}")
    col[3].metric("Pares de Treino", f"{info['training_pairs']:,}")


# ================================================================
# ABA 2: API
# ================================================================
def render_api_tab(api_metrics: dict | None) -> None:
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
    ml_metrics = load_json(METRICS_PATH)
    if ml_metrics:
        eval_clf = load_parquet(EVAL_CLF_PATH)
        eval_reg = load_parquet(EVAL_REG_PATH)
        render_ml_tab(ml_metrics, eval_clf, eval_reg)
    else:
        st.warning(
            "Arquivo data/models/metrics.json não encontrado. "
            "Execute o treino dos modelos primeiro."
        )

with tab_api:
    api_metrics = fetch_api_metrics()
    render_api_tab(api_metrics)
