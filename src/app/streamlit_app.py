"""
JobMatch AI — Frontend Streamlit
Uso: streamlit run src/app/streamlit_app.py

Modo API (padrão): faz requisições HTTP para FastAPI em localhost:8000.
Fallback direto: carrega modelos localmente se API estiver offline.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger("streamlit_app")

API_URL = os.getenv("API_URL", "http://localhost:8000")
USE_API = os.getenv("STREAMLIT_USE_API", "true").lower() == "true"


def extract_text_from_pdf(file) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.warning("Erro extraindo PDF: %s", e)
        return ""


def extract_text_from_docx(file) -> str:
    try:
        import docx
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.warning("Erro extraindo DOCX: %s", e)
        return ""


if USE_API:
    try:
        import httpx
        client = httpx.Client(base_url=API_URL, timeout=30)
        client.post("/health")
        API_AVAILABLE = True
    except Exception:
        API_AVAILABLE = False
        logger.warning("API não disponível em %s. Usando fallback direto.", API_URL)
        from src.models.classifier import predict as classify
        from src.models.recommender import rank_jobs
        from src.models.salary_model import predict_salary_range
        from src.models.vectorizer import load_vectorizer, transform
        from src.pipeline.preprocess import clean_text
        from src.skills.skills_analyzer import analyze_gap
else:
    API_AVAILABLE = False
    from src.models.classifier import predict as classify
    from src.models.recommender import rank_jobs
    from src.models.salary_model import predict_salary_range
    from src.models.vectorizer import load_vectorizer, transform
    from src.pipeline.preprocess import clean_text
    from src.skills.skills_analyzer import analyze_gap


st.set_page_config(
    page_title="JobMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
    margin-bottom: 1rem;
}
.metric-card h1 { font-size: 3rem; margin: 0; }
.metric-card p  { font-size: 1rem; margin: 0; opacity: 0.9; }
.fit-badge   { background: #10b981; color: white; padding: 4px 14px;
               border-radius: 20px; font-weight: bold; }
.nofit-badge { background: #ef4444; color: white; padding: 4px 14px;
               border-radius: 20px; font-weight: bold; }
.skill-chip-ok  { display:inline-block; background:#d1fae5; color:#065f46;
                  padding:2px 10px; border-radius:12px; margin:2px; font-size:0.85rem; }
.skill-chip-gap { display:inline-block; background:#fee2e2; color:#991b1b;
                  padding:2px 10px; border-radius:12px; margin:2px; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Carregando modelos de ML…")
def load_models_direct():
    from src.models.vectorizer import load_vectorizer, transform
    import joblib
    vec = load_vectorizer()
    clf = joblib.load("data/models/classifier.pkl")
    sal = joblib.load("data/models/salary_regressor.pkl")
    jobs = pd.read_parquet("data/processed/jobs_clean.parquet")
    jobs_matrix = transform(jobs["full_text"].tolist(), vec)
    logger.info("Modelos carregados em modo direto")
    return vec, clf, sal, jobs, jobs_matrix


def predict_via_api(resume_text: str, top_k: int, threshold: float,
                    use_sbert: bool = False, use_cross_encoder: bool = False) -> dict:
    response = client.post("/predict", json={
        "resume_text": resume_text,
        "top_k": top_k,
        "fit_threshold": threshold,
        "use_sbert": use_sbert,
        "use_cross_encoder": use_cross_encoder,
    })
    response.raise_for_status()
    return response.json()


def predict_direct(resume_text: str, top_k: int, threshold: float,
                   use_sbert: bool = False, use_cross_encoder: bool = False) -> dict:
    vec, clf, sal, jobs_df, jobs_matrix = load_models_direct()

    resume_clean = clean_text(resume_text)

    if use_sbert:
        try:
            from src.models.vectorizer import SentenceBertVectorizer
            sbert = joblib.load("data/models/sentence_bert.pkl")
            clf_sbert = joblib.load("data/models/classifier_sbert.pkl")
            jobs_sbert = np.load("data/models/jobs_sbert_embeddings.npy")
            resume_vec = sbert.transform([resume_clean])
            jobs_matrix = jobs_sbert
            clf = clf_sbert
        except Exception as e:
            logger.warning("SBERT não disponível, usando TF-IDF: %s", e)

    fit_label, fit_prob = classify(resume_vec, clf)
    score_pct = round(fit_prob * 100, 1)

    top_jobs = rank_jobs(
        resume_vec, jobs_matrix, jobs_df,
        top_k=top_k, fit_threshold=threshold,
        resume_text=resume_text,
        use_cross_encoder=use_cross_encoder,
    )

    best_job_idx = top_jobs.index[0]
    best_job_vec = transform([jobs_df.loc[best_job_idx, "full_text"]], vec)
    salary_est = predict_salary_range(best_job_vec, sal)

    gap = analyze_gap(resume_text, top_jobs.iloc[0]["title"])

    job_scores = []
    all_unique_required = set()
    all_unique_compat = set()
    for _, row in top_jobs.iterrows():
        title = row.get("title", "")
        g = analyze_gap(resume_text, title)
        compat = set(g["compatible"])
        missing = set(g["missing"])
        total = len(compat) + len(missing)
        all_unique_required.update(compat, missing)
        all_unique_compat.update(compat)
        if total > 0:
            job_scores.append(len(compat) / total * 100)
    if job_scores:
        employability_score = round(
            sum(job_scores) / len(job_scores), 1
        )
    elif all_unique_required:
        employability_score = round(
            len(all_unique_compat) / len(all_unique_required) * 100, 1
        )
    else:
        employability_score = 0.0

    return {
        "fit_label": fit_label,
        "score_pct": score_pct,
        "avg_adherence": round(top_jobs["adherence_score"].mean(), 1),
        "fit_count": int((top_jobs["adherence_score"] >= threshold).sum()),
        "top_k": top_k,
        "employability_score": employability_score,
        "salary_est": salary_est,
        "gap": gap,
        "top_jobs": top_jobs.reset_index().to_dict(orient="records"),
    }


with st.sidebar:
    st.markdown("## 🎯 JobMatch AI")
    mode_text = "🔵 Modo API" if API_AVAILABLE else "🟠 Modo Direto"
    st.caption(mode_text)
    st.markdown("---")
    st.subheader("⚙️ Configurações")
    top_k = st.slider("Número de vagas recomendadas", 1, 10, 5)
    fit_threshold = st.slider(
        "Threshold de Fit (%)", 20, 80, 40,
        help="Score mínimo para classificar como Fit",
    )
    show_plan = st.checkbox("Mostrar plano de desenvolvimento", value=True)
    use_sbert = st.checkbox(
        "Sentence-BERT (embedding semântico, ~3s extra)",
        value=False,
        help="Usa Sentence-BERT em vez de TF-IDF para maior precisão semântica",
    )
    use_cross_encoder = st.checkbox(
        "Re-ranking com Cross-Encoder (mais preciso, ~2s extra)",
        value=False,
        help="Re-ordena as vagas usando um modelo neural cross-encoder",
    )
    st.markdown("---")
    st.markdown("**Como usar:**")
    st.markdown(
        "1. Faça upload do currículo (PDF/DOCX) ou cole o texto\n"
        "2. Clique em **Analisar**\n"
        "3. Veja suas vagas ideais!"
    )

st.title("🎯 JobMatch AI")
st.markdown("*Descubra sua compatibilidade real com vagas e acelere sua contratação.*")
st.markdown("---")

col_input, col_tip = st.columns([3, 1])
with col_input:
    uploaded_file = st.file_uploader(
        "📎 Ou faça upload do currículo (PDF/DOCX):",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            resume_text = extract_text_from_docx(uploaded_file)
        else:
            resume_text = ""
        st.success(f"✅ Arquivo '{uploaded_file.name}' extraído ({len(resume_text)} caracteres)")
    else:
        resume_text = st.text_area(
            "📄 Cole seu currículo ou descreva seu perfil profissional:",
            height=280,
            placeholder="Exemplo:\n"
            "Sou Analista de Dados com 3 anos de experiência em Python, SQL e Power BI...",
        )
with col_tip:
    st.info(
        "💡 **Dicas:**\n\n"
        "• Faça upload do PDF/DOCX ou cole o texto\n"
        "• Inclua suas tecnologias\n"
        "• Mencione cargos anteriores\n"
        "• Liste certificações\n"
        "• Descreva projetos relevantes"
    )

run_btn = st.button(
    "🔍 Analisar Compatibilidade",
    type="primary",
    disabled=(not resume_text.strip()),
    use_container_width=True,
)

if run_btn and resume_text.strip():
    with st.spinner("Analisando seu perfil nas 124k+ vagas…"):
        try:
            if API_AVAILABLE:
                result = predict_via_api(resume_text, top_k, fit_threshold,
                                         use_sbert=use_sbert,
                                         use_cross_encoder=use_cross_encoder)
                logger.info("Predição via API concluída")
            else:
                result = predict_direct(resume_text, top_k, fit_threshold,
                                        use_sbert=use_sbert,
                                        use_cross_encoder=use_cross_encoder)
                logger.info("Predição via modo direto concluída")
        except Exception as e:
            logger.error("Erro na análise: %s", e)
            st.error(f"Erro durante a análise: {e}")
            st.stop()

    st.success("✅ Análise concluída!")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <p>Score Médio de Aderência</p>
                <h1>{result['avg_adherence']:.1f}%</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card" style="background:linear-gradient(135deg,#11998e,#38ef7d)">
                <p>Vagas com Fit</p>
                <h1>{result['fit_count']}/{result['top_k']}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card" style="background:linear-gradient(135deg,#667eea,#764ba2)">
                <p>Empregabilidade</p>
                <h1>{result.get('employability_score', 0):.1f}%</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        sal = result["salary_est"]
        sal_fmt = f"${sal['range_low']:,.0f} – ${sal['range_high']:,.0f}"
        st.markdown(
            f"""
            <div class="metric-card" style="background:linear-gradient(135deg,#f093fb,#f5576c)">
                <p>Faixa Salarial Estimada (Top Vaga)</p>
                <h1 style="font-size:1.6rem">{sal_fmt}/ano</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader(f"🏆 Top-{result['top_k']} Vagas Mais Compatíveis")
    for i, row in enumerate(result["top_jobs"]):
        with st.expander(
            f"**#{i+1} {row['title']}** @ {row.get('company_name', 'N/A')}  |  "
            f"Score: **{row['adherence_score']:.1f}%**  {row['fit_label']}",
            expanded=(i == 0),
        ):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"📍 **Localização**: {row.get('location', 'N/A')}")
                st.markdown(
                    f"💰 **Salário**: "
                    f"${row.get('min_salary_annual', 0):,.0f} – "
                    f"${row.get('max_salary_annual', 0):,.0f} /ano"
                )
            with col_b:
                score = row["adherence_score"]
                color = "#10b981" if score >= fit_threshold else "#ef4444"
                st.markdown(
                    f"<div style='text-align:center;font-size:2rem;color:{color}'>"
                    f"<b>{score:.1f}%</b><br><small>aderência</small></div>",
                    unsafe_allow_html=True,
                )
            if row.get("skills_desc"):
                st.markdown(f"**Skills exigidas:** {row['skills_desc'][:300]}...")

    st.markdown("---")
    st.subheader("🔍 Análise de Skills (Vaga #1)")
    gap = result["gap"]

    col_ok, col_gap = st.columns(2)
    with col_ok:
        st.markdown("**✅ Skills Compatíveis**")
        if gap["compatible"]:
            chips = " ".join(
                f'<span class="skill-chip-ok">{s}</span>' for s in gap["compatible"]
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info("Nenhuma skill mapeada no seu perfil para esta vaga.")

    with col_gap:
        st.markdown("**❌ Skills Faltantes**")
        if gap["missing"]:
            chips = " ".join(
                f'<span class="skill-chip-gap">{s}</span>' for s in gap["missing"]
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.success("Você tem todas as skills mapeadas!")

    if show_plan and gap.get("development_plan"):
        st.markdown("---")
        st.subheader("📚 Plano de Desenvolvimento Sugerido")
        for item in gap["development_plan"]:
            with st.expander(f"📖 {item['skill'].title()}"):
                st.markdown(f"**Curso sugerido**: {item['curso']}")
                st.markdown(f"**Tempo estimado**: {item['tempo']}")

    st.markdown("---")
    if st.button("🔄 Nova Análise", use_container_width=True):
        st.rerun()

elif not run_btn:
    st.markdown(
        """
        <div style="text-align:center; padding:3rem; color:#6b7280;">
            <h2>👆 Cole seu currículo acima e clique em Analisar</h2>
            <p>O sistema vai buscar as vagas mais compatíveis com seu perfil em 124k+ ofertas reais.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
