"""Job Copilot UI - Streamlit frontend over the same core functions."""
import streamlit as st
from match import analyze
from cover_letter import write_cover_letter
from tracker import log_application, list_applications, update_status

st.set_page_config(page_title="Job Copilot", page_icon="💼", layout="wide")
theme_choice = st.sidebar.radio("🎨 Theme", ["System", "Dark", "Light"], index=0)

DARK = {"bg":"#0E1117","panel":"#1A1D29","text":"#E8E8F0","sub":"#D8D4FF"}
LIGHT = {"bg":"#FFFFFF","panel":"#F2F1FA","text":"#1A1A2E","sub":"#EDEBFF"}

if theme_choice == "Dark":
    c = DARK; media = ""
elif theme_choice == "Light":
    c = LIGHT; media = ""
else:
    c = DARK  # fallback values; media query below handles the real switching
    media = """
    @media (prefers-color-scheme: light) {
      .stApp { background: #FFFFFF !important; color: #1A1A2E !important; }
      .themed-panel, .stTabs [data-baseweb="tab"] { background: #F2F1FA !important; }
      [data-testid="stMetric"] { background: #F2F1FA !important; }
    }"""

st.markdown(f"""
<style>
.stApp {{ background: {c["bg"]}; color: {c["text"]}; }}

.hero {{
  background: linear-gradient(135deg, #7C6FF0 0%, #4A3FBF 60%, #2E2A72 100%);
  padding: 2.2rem 2.5rem; border-radius: 18px; margin-bottom: 1.5rem;
}}
.hero h1 {{ color: white; margin: 0; font-size: 2.3rem; }}
.hero p {{ color: {c["sub"]}; margin: 0.4rem 0 0 0; font-size: 1.05rem; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
.stTabs [data-baseweb="tab"] {{
  background: {c["panel"]}; border-radius: 10px 10px 0 0; padding: 10px 18px;
}}
.stTabs [aria-selected="true"] {{ background: #7C6FF0 !important; color: white !important; }}

.stTextInput input, .stTextArea textarea {{ border-radius: 10px !important; }}
.stButton button {{
  border-radius: 10px; padding: 0.5rem 1.6rem; font-weight: 600;
  box-shadow: 0 4px 14px rgba(124,111,240,0.35);
}}

[data-testid="stMetric"] {{
  background: {c["panel"]}; border: 1px solid #7C6FF0; border-radius: 14px;
  padding: 1rem 1.4rem; width: fit-content;
}}
{media}
</style>

<div class="hero">
  <h1>💼 Job Copilot</h1>
  <p>Local AI job-search automation — RAG matching, cover letters, tracking. Zero API costs.</p>
</div>
""", unsafe_allow_html=True)

tab_match, tab_letter, tab_tracker = st.tabs(["📊 Match", "✍️ Cover Letter", "📋 Tracker"])

with tab_match:
    col1, col2 = st.columns(2)
    company = col1.text_input("Company")
    role = col2.text_input("Role")
    posting = st.text_area("Job posting", height=250, placeholder="Paste the full job description here...")

    if st.button("Analyze", type="primary"):
        if len(posting.strip()) < 50:
            st.error("Posting too short to analyze.")
        else:
            with st.spinner("Scoring against your profile..."):
                report = analyze(posting)
            st.metric("Match score", f"{report.match_score}%")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("✅ Matching skills")
                for s in report.matching_skills:
                    st.markdown(f"- {s}")
            with c2:
                st.subheader("❌ Missing keywords")
                for k in report.missing_keywords:
                    st.markdown(f"- {k}")
            st.subheader("⭐ Emphasize")
            st.write(", ".join(report.projects_to_emphasize))
            st.info(report.one_line_verdict)
            if company and role:
                app_id = log_application(company, role, report)
                st.success(f"Logged to tracker as #{app_id}")

with tab_letter:
    company_l = st.text_input("Company name", key="letter_company")
    posting_l = st.text_area("Job posting", height=250, key="letter_posting")
    if st.button("Draft letter", type="primary"):
        if len(posting_l.strip()) < 50:
            st.error("Posting too short.")
        else:
            with st.spinner("Writing from your real experience..."):
                letter = write_cover_letter(posting_l, company_l or "the company")
            st.text_area("Draft (edit before sending!)", letter, height=350)

with tab_tracker:
    apps = list_applications()
    if not apps:
        st.info("Nothing tracked yet - analyze a posting with company + role filled in.")
    else:
        st.dataframe(apps, use_container_width=True)
        c1, c2 = st.columns(2)
        app_id = c1.number_input("Application #", min_value=1, step=1)
        new_status = c2.selectbox("Status", ["analyzed", "applied", "interview", "offer", "rejected"])
        if st.button("Update status"):
            update_status(int(app_id), new_status)
            st.success(f"#{int(app_id)} → {new_status}")
            st.rerun()
