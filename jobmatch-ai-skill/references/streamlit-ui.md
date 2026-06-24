# Frontend Streamlit — JobMatch AI

Código completo do frontend. Salve como `src/app/streamlit_app.py`.

## Execução

```bash
streamlit run src/app/streamlit_app.py
```

---

## Código Completo

```python
"""
JobMatch AI — Frontend Streamlit
Uso: streamlit run src/app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ── Imports internos ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.preprocess import clean_text
from src.models.vectorizer import load_vectorizer, transform
from src.models.classifier import predict as classify
from src.models.recommender import rank_jobs
from src.models.salary_model import predict_salary_range
from src.skills.skills_analyzer import analyze_gap

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobMatch AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Customizado ──────────────────────────────────────────────────────────
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

# ── Carregamento de modelos (cache) ──────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelos de ML…")
def load_models():
    vec  = load_vectorizer()
    clf  = joblib.load("data/models/classifier.pkl")
    sal  = joblib.load("data/models/salary_regressor.pkl")
    jobs = pd.read_parquet("data/processed/jobs_clean.parquet")
    # Pré-vetorizar todas as vagas
    jobs_matrix = transform(jobs['full_text'].tolist(), vec)
    return vec, clf, sal, jobs, jobs_matrix

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=JobMatch+AI", use_column_width=True)
    st.markdown("---")
    st.subheader("⚙️ Configurações")
    top_k = st.slider("Número de vagas recomendadas", 1, 10, 5)
    fit_threshold = st.slider("Threshold de Fit (%)", 20, 80, 40,
                              help="Score mínimo para classificar como Fit")
    show_plan = st.checkbox("Mostrar plano de desenvolvimento", value=True)
    st.markdown("---")
    st.markdown("**Como usar:**")
    st.markdown("1. Cole seu currículo ou descreva seu perfil\n"
                "2. Clique em **Analisar**\n"
                "3. Veja suas vagas ideais!")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🎯 JobMatch AI")
st.markdown("*Descubra sua compatibilidade real com vagas e acelere sua contratação.*")
st.markdown("---")

# ── Input ────────────────────────────────────────────────────────────────────
col_input, col_tip = st.columns([3, 1])
with col_input:
    resume_text = st.text_area(
        "📄 Cole seu currículo ou descreva seu perfil profissional:",
        height=280,
        placeholder=(
            "Exemplo:\n"
            "Sou Analista de Dados com 3 anos de experiência em Python, SQL e Power BI. "
            "Trabalhei com pipelines ETL, dashboards de BI e modelos de machine learning "
            "usando scikit-learn e pandas. Tenho experiência em análise de dados de marketing "
            "e falo inglês intermediário..."
        )
    )
with col_tip:
    st.info("💡 **Dicas para melhores resultados:**\n\n"
            "• Inclua suas tecnologias\n"
            "• Mencione cargos anteriores\n"
            "• Liste certificações\n"
            "• Descreva projetos relevantes")

run_btn = st.button("🔍 Analisar Compatibilidade", type="primary",
                    disabled=(not resume_text.strip()), use_container_width=True)

# ── Processamento ────────────────────────────────────────────────────────────
if run_btn and resume_text.strip():
    with st.spinner("Analisando seu perfil nas 124k+ vagas…"):
        try:
            vec, clf, sal, jobs_df, jobs_matrix = load_models()
            
            # 1. Preprocessar
            resume_clean = clean_text(resume_text)
            resume_vec   = transform([resume_clean], vec)
            
            # 2. Classificar
            fit_label, fit_prob = classify(resume_vec, clf)
            score_pct = round(fit_prob * 100, 1)
            
            # 3. Ranking Top-K
            top_jobs = rank_jobs(resume_vec, jobs_matrix, jobs_df, top_k=top_k)
            top_jobs['fit_label'] = top_jobs['adherence_score'].apply(
                lambda s: "✅ Fit" if s >= fit_threshold else "❌ No Fit"
            )
            
            # 4. Salário estimado para a vaga #1
            best_job_idx = top_jobs.index[0]
            best_job_vec = transform([jobs_df.loc[best_job_idx, 'full_text']], vec)
            salary_est   = predict_salary_range(best_job_vec, sal)
            
            # 5. Skills gap para a vaga #1
            gap = analyze_gap(resume_text, top_jobs.iloc[0]['title'])

        except Exception as e:
            st.error(f"Erro durante a análise: {e}")
            st.stop()

    # ── Resultados ────────────────────────────────────────────────────────────
    st.success("✅ Análise concluída!")
    st.markdown("---")

    # Métricas principais
    c1, c2, c3 = st.columns(3)
    with c1:
        avg_score = top_jobs['adherence_score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <p>Score Médio de Aderência</p>
            <h1>{avg_score:.1f}%</h1>
        </div>""", unsafe_allow_html=True)
    with c2:
        fit_count = (top_jobs['fit_label'].str.contains('Fit ✅') |
                     top_jobs['adherence_score'] >= fit_threshold).sum()
        badge_cls = "fit-badge" if fit_count >= top_k // 2 else "nofit-badge"
        st.markdown(f"""
        <div class="metric-card" style="background:linear-gradient(135deg,#11998e,#38ef7d)">
            <p>Vagas com Fit</p>
            <h1>{fit_count}/{top_k}</h1>
        </div>""", unsafe_allow_html=True)
    with c3:
        sal_fmt = f"${salary_est['range_low']:,.0f} – ${salary_est['range_high']:,.0f}"
        st.markdown(f"""
        <div class="metric-card" style="background:linear-gradient(135deg,#f093fb,#f5576c)">
            <p>Faixa Salarial Estimada (Top Vaga)</p>
            <h1 style="font-size:1.6rem">{sal_fmt}/ano</h1>
        </div>""", unsafe_allow_html=True)

    # Top-K Vagas
    st.subheader(f"🏆 Top-{top_k} Vagas Mais Compatíveis")
    for i, row in top_jobs.reset_index().iterrows():
        with st.expander(
            f"**#{i+1} {row['title']}** @ {row['company_name']}  |  "
            f"Score: **{row['adherence_score']:.1f}%**  {row['fit_label']}",
            expanded=(i == 0)
        ):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"📍 **Localização**: {row.get('location', 'N/A')}")
                st.markdown(f"💰 **Salário**: "
                            f"${row.get('min_salary_annual', 0):,.0f} – "
                            f"${row.get('max_salary_annual', 0):,.0f} /ano")
            with col_b:
                score = row['adherence_score']
                color = "#10b981" if score >= fit_threshold else "#ef4444"
                st.markdown(
                    f"<div style='text-align:center;font-size:2rem;color:{color}'>"
                    f"<b>{score:.1f}%</b><br><small>aderência</small></div>",
                    unsafe_allow_html=True
                )
            if row.get('skills_desc'):
                st.markdown(f"**Skills exigidas:** {row['skills_desc'][:300]}...")

    # Skills Analysis
    st.markdown("---")
    st.subheader("🔍 Análise de Skills (Vaga #1)")

    col_ok, col_gap = st.columns(2)
    with col_ok:
        st.markdown("**✅ Skills Compatíveis**")
        if gap['compatible']:
            chips = ' '.join(
                f'<span class="skill-chip-ok">{s}</span>' for s in gap['compatible']
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info("Nenhuma skill mapeada no seu perfil para esta vaga.")

    with col_gap:
        st.markdown("**❌ Skills Faltantes**")
        if gap['missing']:
            chips = ' '.join(
                f'<span class="skill-chip-gap">{s}</span>' for s in gap['missing']
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.success("Você tem todas as skills mapeadas!")

    # Plano de Desenvolvimento
    if show_plan and gap.get('development_plan'):
        st.markdown("---")
        st.subheader("📚 Plano de Desenvolvimento Sugerido")
        for item in gap['development_plan']:
            with st.expander(f"📖 {item['skill'].title()}"):
                st.markdown(f"**Curso sugerido**: {item['curso']}")
                st.markdown(f"**Tempo estimado**: {item['tempo']}")

    # Botão de nova análise
    st.markdown("---")
    if st.button("🔄 Nova Análise", use_container_width=True):
        st.rerun()

elif not run_btn:
    # Estado inicial
    st.markdown("""
    <div style="text-align:center; padding:3rem; color:#6b7280;">
        <h2>👆 Cole seu currículo acima e clique em Analisar</h2>
        <p>O sistema vai buscar as vagas mais compatíveis com seu perfil em 124k+ ofertas reais.</p>
    </div>
    """, unsafe_allow_html=True)
```

---

## Notas de UX

- **Tempo de resposta**: A vetorização de 124k vagas é pesada. Se necessário, pré-calcule e salve `jobs_matrix` em disco com `scipy.sparse.save_npz`.
- **Cache de recursos**: `@st.cache_resource` garante que modelos são carregados apenas uma vez por sessão.
- **Modo offline**: Se quiser rodar sem internet, garanta que todos os arquivos em `data/` já estejam presentes antes de subir o app.
- **Deploy**: Para deploy no Streamlit Cloud, adicione `secrets.toml` com credenciais se necessário.

## Alternativas de Frontend

Se preferir não usar Streamlit, as alternativas viáveis são:

| Tecnologia | Prós | Contras |
|---|---|---|
| **Gradio** | Mais simples que Streamlit | Menos flexível para layouts customizados |
| **FastAPI + React** | Frontend profissional | Muito mais trabalho de implementação |
| **Flask + Jinja2** | Leve e simples | HTML manual, menos componentes prontos |

Para o escopo deste projeto acadêmico, **Streamlit é a recomendação**.
