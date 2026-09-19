import os
import io
import json
import time
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pymupdf
import docx

# Optional direct backend imports as resilient fallback
import jd_index
from analysis_pipeline import AnalysisPipeline

# Page configuration
st.set_page_config(
    page_title="AI Placement Analyzer — Test Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# ── Session State Initialization ──
if "resumes" not in st.session_state:
    st.session_state.resumes = [] # list of dicts: {"id", "name", "text", "words", "status"}
if "jds" not in st.session_state:
    st.session_state.jds = []     # list of dicts: {"id", "title", "company", "location", "text", "words", "status", "backend_id"}
if "rank_results" not in st.session_state:
    st.session_state.rank_results = None
if "selected_jd_id" not in st.session_state:
    st.session_state.selected_jd_id = None
if "analyses" not in st.session_state:
    st.session_state.analyses = {} # keyed by jd_id
if "selected_resume_idx" not in st.session_state:
    st.session_state.selected_resume_idx = 0


# ── Helper Functions: Document Parsing ──

def extract_text_from_file(uploaded_file) -> str:
    """Extracts raw text from PDF, DOCX, or TXT uploaded files."""
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        return text.strip()

    elif filename.endswith(".docx") or filename.endswith(".doc"):
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip()

    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="replace").strip()

    else:
        raise ValueError(f"Unsupported file format: {uploaded_file.name}")


def validate_doc_text(text: str) -> tuple[str, str]:
    """Returns (status, warning_msg). Status: 'Valid', 'Empty', 'Short', 'Excessive'."""
    word_count = len(text.split())
    if word_count == 0:
        return "Empty", "Document contains no text."
    elif word_count < 30:
        return "Short", f"Only {word_count} words (<30). May trigger Insufficient Context gate."
    elif word_count > 4000:
        return "Excessive", f"Document has {word_count} words (>4000). May increase latency."
    return "Valid", "Ready for analysis."


# ── Backend Communication (API with Direct Fallback) ──

def check_backend_alive() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/jds/count", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def api_reset_jds():
    """Resets JDs on backend."""
    try:
        requests.delete(f"{API_BASE_URL}/jds", timeout=3.0)
    except Exception:
        jd_index.clear_store_for_testing()


def api_add_jd(title: str, company: str, location: str, text: str) -> dict:
    """Ingests JD via API or local fallback."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/jds",
            json={"title": title, "company": company, "location": location, "jd_text": text},
            timeout=10.0
        )
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 409:
            return {"error": "STORE_CAP_REACHED", "detail": r.json().get("detail")}
    except Exception:
        pass

    # Direct fallback
    res = jd_index.add_jd(title, company, location, text)
    if hasattr(res, "error_code"):
        return {"error": res.error_code, "detail": res.message}
    return res.model_dump()


def api_rank(resume_text: str, resume_id: str) -> dict:
    """Ranks stored JDs."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/rank",
            json={"resume_text": resume_text, "resume_id": resume_id},
            timeout=10.0
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    import ranking
    return ranking.rank_jds(resume_text, resume_id).model_dump()


@st.cache_resource
def get_local_pipeline():
    return AnalysisPipeline()


def api_analyze(resume_text: str, resume_id: str, jd_id: str) -> dict:
    """Runs detailed Tier 2 analysis."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/analyze",
            json={"resume_text": resume_text, "resume_id": resume_id, "jd_id": jd_id},
            timeout=45.0
        )
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 404:
            return {"error": "JD_NOT_FOUND", "detail": r.json().get("detail")}
    except Exception:
        pass

    pipeline = get_local_pipeline()
    return pipeline.analyze(resume_text, resume_id, jd_id)


# ── Charting Helpers ──

def make_gauge_chart(score: int) -> go.Figure:
    """Creates a circular gauge chart for match score."""
    color = "#2ecc71" if score >= 75 else ("#f39c12" if score >= 50 else "#e74c3c")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Overall Match Score", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#bdc3c7"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#ecf0f1",
            "borderwidth": 1,
            "steps": [
                {"range": [0, 50], "color": "rgba(231, 76, 60, 0.15)"},
                {"range": [50, 75], "color": "rgba(243, 156, 18, 0.15)"},
                {"range": [75, 100], "color": "rgba(46, 204, 113, 0.15)"}
            ]
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def make_skill_donut_chart(skill_gap: dict) -> go.Figure:
    """Donut chart for skill match categories."""
    labels = ["Matched", "Missing (Essential)", "Missing (Preferred)", "Candidate Extra"]
    values = [
        len(skill_gap.get("matched", [])),
        len(skill_gap.get("missing_essential", [])),
        len(skill_gap.get("missing_preferred", [])),
        len(skill_gap.get("extra", []))
    ]
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+value"
    )])
    fig.update_layout(
        title_text="Skill Match Composition",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig


def make_requirement_bar_chart(req_matches: list) -> go.Figure:
    """Bar chart for requirement categories."""
    categories = [r.get("category", "") for r in req_matches]
    percentages = [r.get("match_percent", 0) for r in req_matches]
    colors = ["#3498db" if c != "Skills" else "#2ecc71" for c in categories]

    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=percentages,
        marker_color=colors,
        text=[f"{p}%" for p in percentages],
        textposition="auto"
    )])
    fig.update_layout(
        title_text="Requirement Category Alignment (%)",
        yaxis=dict(range=[0, 100]),
        height=280,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def make_skill_comparison_bar(skill_gap: dict) -> go.Figure:
    """Compares candidate skills vs JD requirements count."""
    matched_count = len(skill_gap.get("matched", []))
    miss_ess = len(skill_gap.get("missing_essential", []))
    miss_pref = len(skill_gap.get("missing_preferred", []))
    extra_count = len(skill_gap.get("extra", []))

    candidate_total = matched_count + extra_count
    jd_total = matched_count + miss_ess + miss_pref

    fig = go.Figure(data=[
        go.Bar(name="Candidate Skills", x=["Resume Skills", "JD Requirements"], y=[candidate_total, 0], marker_color="#3498db"),
        go.Bar(name="JD Requirements", x=["Resume Skills", "JD Requirements"], y=[0, jd_total], marker_color="#9b59b6"),
    ])
    fig.update_layout(
        barmode="group",
        title_text="Total Skill Volume Comparison",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


# ── SIDEBAR: System Status & Session Controls ──

with st.sidebar:
    st.title("🎯 AI Placement Analyzer")
    st.caption("Phase 1 Testing & Verification Dashboard")
    st.divider()

    # Backend status badge
    backend_ok = check_backend_alive()
    if backend_ok:
        st.success("🟢 FastAPI Backend: Connected (127.0.0.1:8000)")
    else:
        st.warning("🟠 Backend Server Offline — using Direct Pipeline fallback")

    st.subheader("📊 Session Counters (Feature #3)")
    col1, col2 = st.columns(2)
    col1.metric("Resumes", f"{len(st.session_state.resumes)} / 10")
    col2.metric("JDs", f"{len(st.session_state.jds)} / 10")

    st.progress(len(st.session_state.resumes) / 10, text="Resume Cap")
    st.progress(len(st.session_state.jds) / 10, text="JD Cap")

    st.divider()

    # Reset Action (Feature #20)
    if st.button("🔄 Reset Session & Clear Store", use_container_width=True, type="secondary"):
        api_reset_jds()
        st.session_state.resumes = []
        st.session_state.jds = []
        st.session_state.rank_results = None
        st.session_state.selected_jd_id = None
        st.session_state.analyses = {}
        st.session_state.selected_resume_idx = 0
        st.success("Session reset! Store cleared.")
        st.rerun()

    # Quick Fixture Loader
    st.divider()
    st.subheader("⚡ Quick Test Fixtures")
    if st.button("📥 Load 5 Sample Resumes & 5 JDs", use_container_width=True):
        if os.path.exists("tests/fixtures/sample_data.json"):
            with open("tests/fixtures/sample_data.json", "r", encoding="utf-8") as f:
                fixtures = json.load(f)

            # Load Resumes
            st.session_state.resumes = []
            for r in fixtures.get("resumes", []):
                text = r["text"]
                status, warn = validate_doc_text(text)
                st.session_state.resumes.append({
                    "id": r["id"],
                    "name": r["id"].replace("_", " ").title(),
                    "text": text,
                    "words": len(text.split()),
                    "status": status,
                    "warning": warn
                })

            # Load JDs
            api_reset_jds()
            st.session_state.jds = []
            for j in fixtures.get("jds", []):
                text = j["text"]
                status, warn = validate_doc_text(text)
                st.session_state.jds.append({
                    "id": j["id"],
                    "title": j["title"],
                    "company": j["company"],
                    "location": j["location"],
                    "text": text,
                    "words": len(text.split()),
                    "status": status,
                    "warning": warn,
                    "backend_id": None
                })
            st.success("5 Resumes and 5 JDs loaded from fixtures!")
            st.rerun()


# ── MAIN PANEL ──

tab_upload, tab_results, tab_detail = st.tabs([
    "📁 1. Upload & Manage Documents",
    "🏆 2. Top-5 Matching Results",
    "🔬 3. In-Depth Gap Analysis"
])


# ════════════════════════════════════════════════════════════════
# TAB 1: RESUME & JD UPLOAD PANEL (Features #1, #2, #3, #4)
# ════════════════════════════════════════════════════════════════

with tab_upload:
    st.header("Upload & Document Management Panel")
    st.markdown("Upload up to **10 candidate resumes** and **10 job descriptions** via PDF, DOCX, TXT, or manual entry.")

    col_res, col_jd = st.columns(2)

    # ── Left Column: Resumes ──
    with col_res:
        st.subheader("📄 Candidate Resumes (Max 10)")
        res_upload_mode = st.radio("Resume Upload Mode", ["Files (PDF / DOCX / TXT)", "Manual Paste"], horizontal=True, key="res_mode")

        if res_upload_mode == "Files (PDF / DOCX / TXT)":
            res_files = st.file_uploader(
                "Select Resume Files",
                type=["pdf", "docx", "doc", "txt"],
                accept_multiple_files=True,
                key="res_file_input"
            )
            if res_files and st.button("➕ Add Uploaded Resumes"):
                for f in res_files:
                    if len(st.session_state.resumes) >= 10:
                        st.warning("Resume cap reached (10 max).")
                        break
                    try:
                        raw_text = extract_text_from_file(f)
                        status, warn = validate_doc_text(raw_text)
                        st.session_state.resumes.append({
                            "id": f"res_{int(time.time()*1000)%100000}",
                            "name": f.name,
                            "text": raw_text,
                            "words": len(raw_text.split()),
                            "status": status,
                            "warning": warn
                        })
                    except Exception as e:
                        st.error(f"Error reading {f.name}: {e}")
                st.rerun()

        else:
            with st.form("manual_resume_form"):
                m_res_name = st.text_input("Candidate Name / ID", value="Candidate John Doe")
                m_res_text = st.text_area("Resume Full Text", height=160, placeholder="Paste resume contents here...")
                submit_res = st.form_submit_button("➕ Add Manual Resume")
                if submit_res:
                    if len(st.session_state.resumes) >= 10:
                        st.error("Resume cap reached (10 max).")
                    elif not m_res_text.strip():
                        st.error("Resume text cannot be empty.")
                    else:
                        status, warn = validate_doc_text(m_res_text)
                        st.session_state.resumes.append({
                            "id": f"res_{int(time.time()*1000)%100000}",
                            "name": m_res_name,
                            "text": m_res_text,
                            "words": len(m_res_text.split()),
                            "status": status,
                            "warning": warn
                        })
                        st.rerun()

        # Preview Resumes (Feature #2 & #4)
        st.markdown(f"**Uploaded Resumes ({len(st.session_state.resumes)}/10)**")
        if not st.session_state.resumes:
            st.info("No resumes uploaded yet.")
        for idx, r in enumerate(st.session_state.resumes):
            with st.expander(f"{r['name']} ({r['words']} words) — Status: {r['status']}"):
                if r['status'] == "Valid":
                    st.success(f"Status: {r['status']} ({r['warning']})")
                else:
                    st.warning(f"Status: {r['status']} — {r['warning']}")
                st.text_area("Content Preview", r['text'][:400] + ("..." if len(r['text']) > 400 else ""), height=100, key=f"prev_r_{idx}")
                if st.button("🗑️ Remove Resume", key=f"del_r_{idx}"):
                    st.session_state.resumes.pop(idx)
                    st.rerun()

    # ── Right Column: Job Descriptions ──
    with col_jd:
        st.subheader("💼 Job Descriptions (Max 10)")
        jd_upload_mode = st.radio("JD Upload Mode", ["Files (PDF / DOCX / TXT)", "Manual Form"], horizontal=True, key="jd_mode")

        if jd_upload_mode == "Files (PDF / DOCX / TXT)":
            jd_files = st.file_uploader(
                "Select JD Files",
                type=["pdf", "docx", "doc", "txt"],
                accept_multiple_files=True,
                key="jd_file_input"
            )
            if jd_files and st.button("➕ Add Uploaded JDs"):
                for f in jd_files:
                    if len(st.session_state.jds) >= 10:
                        st.warning("JD cap reached (10 max).")
                        break
                    try:
                        raw_text = extract_text_from_file(f)
                        status, warn = validate_doc_text(raw_text)
                        title = f.name.replace(".pdf", "").replace(".docx", "").replace(".txt", "")
                        st.session_state.jds.append({
                            "id": f"jd_{int(time.time()*1000)%100000}",
                            "title": title.title(),
                            "company": "Company Inc.",
                            "location": "Remote / Hybrid",
                            "text": raw_text,
                            "words": len(raw_text.split()),
                            "status": status,
                            "warning": warn,
                            "backend_id": None
                        })
                    except Exception as e:
                        st.error(f"Error reading {f.name}: {e}")
                st.rerun()

        else:
            with st.form("manual_jd_form"):
                m_jd_title = st.text_input("Job Title", value="Senior Backend Engineer")
                m_jd_company = st.text_input("Company Name", value="CloudScale Inc")
                m_jd_location = st.text_input("Location", value="Remote")
                m_jd_text = st.text_area("Job Description Full Text", height=120, placeholder="Requirements, qualifications, skills...")
                submit_jd = st.form_submit_button("➕ Add Manual JD")
                if submit_jd:
                    if len(st.session_state.jds) >= 10:
                        st.error("JD cap reached (10 max).")
                    elif not m_jd_text.strip():
                        st.error("JD text cannot be empty.")
                    else:
                        status, warn = validate_doc_text(m_jd_text)
                        st.session_state.jds.append({
                            "id": f"jd_{int(time.time()*1000)%100000}",
                            "title": m_jd_title,
                            "company": m_jd_company,
                            "location": m_jd_location,
                            "text": m_jd_text,
                            "words": len(m_jd_text.split()),
                            "status": status,
                            "warning": warn,
                            "backend_id": None
                        })
                        st.rerun()

        # Preview JDs (Feature #2 & #4)
        st.markdown(f"**Ingested Job Descriptions ({len(st.session_state.jds)}/10)**")
        if not st.session_state.jds:
            st.info("No JDs added yet.")
        for idx, j in enumerate(st.session_state.jds):
            with st.expander(f"{j['title']} @ {j['company']} ({j['words']} words) — {j['status']}"):
                if j['status'] == "Valid":
                    st.success(f"Status: {j['status']} ({j['warning']})")
                else:
                    st.warning(f"Status: {j['status']} — {j['warning']}")
                st.text_area("Content Preview", j['text'][:400] + ("..." if len(j['text']) > 400 else ""), height=100, key=f"prev_j_{idx}")
                if st.button("🗑️ Remove JD", key=f"del_j_{idx}"):
                    st.session_state.jds.pop(idx)
                    st.rerun()

    # ── Action Bar: Analysis Trigger (Feature #5 & #6) ──
    st.divider()
    st.subheader("🚀 Run Alignment Analysis")

    if not st.session_state.resumes:
        st.info("Upload at least one resume to run analysis.")
    elif not st.session_state.jds:
        st.info("Upload at least one job description to run analysis.")
    else:
        resume_choices = [f"{idx+1}. {r['name']} ({r['words']} words)" for idx, r in enumerate(st.session_state.resumes)]
        selected_res_str = st.selectbox("Select Candidate Resume for Analysis", resume_choices)
        selected_idx = int(selected_res_str.split(".")[0]) - 1
        st.session_state.selected_resume_idx = selected_idx

        target_resume = st.session_state.resumes[selected_idx]

        if st.button("🔥 Analyze Placement Fit (Embed & Rank)", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Step 1/5: Uploading & Parsing Documents...")
            time.sleep(0.3)

            try:
                # Step 1: Upload / Ingest JDs to Backend
                progress_bar.progress(25, text="Step 2/5: Ingesting & Embedding Job Descriptions...")
                api_reset_jds()
                for j in st.session_state.jds:
                    res = api_add_jd(j["title"], j["company"], j["location"], j["text"])
                    if "jd_id" in res:
                        j["backend_id"] = res["jd_id"]
                    elif "error" in res:
                        st.error(f"Error adding JD '{j['title']}': {res.get('detail')}")

                # Step 2: Tier 1 Ranking
                progress_bar.progress(50, text="Step 3/5: Vectorizing Resume & Running Cosine Similarity Matching...")
                rank_resp = api_rank(target_resume["text"], target_resume["id"])
                st.session_state.rank_results = rank_resp.get("results", [])

                progress_bar.progress(75, text="Step 4/5: Ranking Top-5 Matches (Zero Gemini Calls)...")
                time.sleep(0.3)

                # Step 3: Trigger Tier 2 analysis on top match if available
                if st.session_state.rank_results:
                    top_jd = st.session_state.rank_results[0]
                    top_id = top_jd["jd_id"]
                    progress_bar.progress(90, text=f"Step 5/5: Running Gemini Deep Analysis on #1 match ({top_jd['title']})...")
                    analysis_res = api_analyze(target_resume["text"], target_resume["id"], top_id)
                    st.session_state.analyses[top_id] = analysis_res
                    st.session_state.selected_jd_id = top_id

                progress_bar.progress(100, text="Complete! Results ready.")
                st.success("Analysis complete! View the results in Tab 2 and Tab 3.")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")


# ════════════════════════════════════════════════════════════════
# TAB 2: TOP-5 JD RESULTS (Features #7, #8, #9, #10)
# ════════════════════════════════════════════════════════════════

with tab_results:
    st.header("🏆 Top-5 Ranked Job Descriptions")

    if not st.session_state.rank_results:
        st.info("No ranking results yet. Go to Tab 1 and click 'Analyze Placement Fit'.")
    else:
        target_res = st.session_state.resumes[st.session_state.selected_resume_idx]
        st.markdown(f"**Target Resume:** `{target_res['name']}` ({target_res['words']} words)")
        st.caption(f"Showing Top {len(st.session_state.rank_results[:5])} matching roles ranked via deterministic cosine similarity (0 Gemini calls).")

        top_5 = st.session_state.rank_results[:5]

        for card in top_5:
            rank_num = card.get("rank", 1)
            sim_score = card.get("similarity_score", 0)
            medal = "🥇" if rank_num == 1 else ("🥈" if rank_num == 2 else ("🥉" if rank_num == 3 else f"#{rank_num}"))
            jd_id = card.get("jd_id")

            with st.container():
                st.markdown(f"### {medal} {card.get('title')} — {card.get('company')}")
                c1, c2, c3 = st.columns([2, 5, 3])

                with c1:
                    st.metric("Similarity Score", f"{sim_score}%")
                    st.caption(f"📍 Location: {card.get('location')}")

                with c2:
                    st.markdown("**Overview Snippet:**")
                    st.write(card.get("snippet", ""))

                with c3:
                    st.markdown("**Actions:**")
                    if st.button(f"🔍 View Full Gap Analysis", key=f"btn_analyze_{jd_id}"):
                        if jd_id not in st.session_state.analyses:
                            with st.spinner(f"Running Gemini Tier 2 analysis on {card.get('title')}..."):
                                analysis_data = api_analyze(target_res["text"], target_res["id"], jd_id)
                                st.session_state.analyses[jd_id] = analysis_data
                        st.session_state.selected_jd_id = jd_id
                        st.success(f"Loaded analysis for {card.get('title')}! Switch to Tab 3.")

                st.divider()


# ════════════════════════════════════════════════════════════════
# TAB 3: IN-DEPTH GAP ANALYSIS (Features #11 to #19)
# ════════════════════════════════════════════════════════════════

with tab_detail:
    st.header("🔬 Detailed Match & Skill Gap Analysis")

    if not st.session_state.selected_jd_id or st.session_state.selected_jd_id not in st.session_state.analyses:
        st.info("Select a job description in Tab 2 by clicking 'View Full Gap Analysis' to inspect deep results.")
    else:
        active_jd_id = st.session_state.selected_jd_id
        analysis = st.session_state.analyses[active_jd_id]

        # Check Insufficient Context State (Feature #17)
        if not analysis.get("context_sufficient", True):
            st.error("⚠️ Insufficient Context Detected")
            st.warning(f"Reason: {analysis.get('insufficient_reason', 'Document too short')}")
            st.info("The backend determined that either the resume or job description does not contain sufficient text for statistically reliable skill grounding. Automated generation was safely bypassed without errors.")
        else:
            # ── Top Row: Gauge & Scores (Feature #9) ──
            score = analysis.get("match_score", 0)
            score_col, summary_col = st.columns([1, 2])

            with score_col:
                st.plotly_chart(make_gauge_chart(score), use_container_width=True)

            with summary_col:
                st.subheader("Executive Fit Summary")
                st.markdown(f"> {analysis.get('summary', 'No summary available.')}")
                st.caption(f"JD ID: `{active_jd_id}` | Resume ID: `{analysis.get('resume_id')}` | Context Sufficient: ✅ True")

            st.divider()

            # ── Row 2: Strengths & Weaknesses (Feature #12) ──
            st.subheader("📋 Strengths & Qualification Gaps")
            str_col, weak_col = st.columns(2)

            with str_col:
                st.markdown("#### ✅ Key Strengths")
                strengths = analysis.get("strengths", [])
                if strengths:
                    for s in strengths:
                        st.markdown(f"- {s}")
                else:
                    st.write("No specific strengths listed.")

            with weak_col:
                st.markdown("#### ⚠️ Areas for Improvement / Gaps")
                weaknesses = analysis.get("weaknesses", [])
                if weaknesses:
                    for w in weaknesses:
                        st.markdown(f"- {w}")
                else:
                    st.write("No major gaps identified.")

            st.divider()

            # ── Row 3: Interactive Visualizations (Features #10, #14, #15) ──
            st.subheader("📊 Visual Alignment Analytics")
            chart_c1, chart_c2, chart_c3 = st.columns(3)

            gap_data = analysis.get("skill_gap", {})
            req_data = analysis.get("requirement_match", [])

            with chart_c1:
                st.plotly_chart(make_skill_donut_chart(gap_data), use_container_width=True)

            with chart_c2:
                st.plotly_chart(make_requirement_bar_chart(req_data), use_container_width=True)

            with chart_c3:
                st.plotly_chart(make_skill_comparison_bar(gap_data), use_container_width=True)

            st.divider()

            # ── Row 4: Skill Gap Breakdown (Feature #13) ──
            st.subheader("🎯 Grounded Skill Gap Breakdown")

            matched_skills = gap_data.get("matched", [])
            miss_essential = gap_data.get("missing_essential", [])
            miss_preferred = gap_data.get("missing_preferred", [])
            extra_skills = gap_data.get("extra", [])

            g_c1, g_c2, g_c3, g_c4 = st.columns(4)

            with g_c1:
                st.markdown(f"**Matched Skills ({len(matched_skills)})**")
                if matched_skills:
                    for m in matched_skills:
                        badge = "🟢" if m.get("match_type") == "exact" else "🟡 (close)"
                        st.markdown(f"- **{m.get('skill_name')}** {badge}")
                else:
                    st.write("None")

            with g_c2:
                st.markdown(f"**Missing Essential ({len(miss_essential)})**")
                if miss_essential:
                    for s in miss_essential:
                        st.markdown(f"- 🔴 **{s}**")
                else:
                    st.write("None")

            with g_c3:
                st.markdown(f"**Missing Preferred ({len(miss_preferred)})**")
                if miss_preferred:
                    for s in miss_preferred:
                        st.markdown(f"- 🟠 **{s}**")
                else:
                    st.write("None")

            with g_c4:
                st.markdown(f"**Candidate Extra ({len(extra_skills)})**")
                if extra_skills:
                    for s in extra_skills:
                        st.markdown(f"- 🔵 **{s}**")
                else:
                    st.write("None")

            st.divider()

            # ── Row 5: Actionable Skill Recommendations (Feature #16) ──
            st.subheader("💡 Targeted Skill Recommendations")
            recs = analysis.get("recommendations", [])
            if recs:
                r_cols = st.columns(min(len(recs), 3))
                for i, r in enumerate(recs):
                    col_idx = i % len(r_cols)
                    with r_cols[col_idx]:
                        with st.container(border=True):
                            st.markdown(f"**🎯 {r.get('skill', 'Skill')}**")
                            st.write(r.get("reason", ""))
            else:
                st.info("No specific recommendations generated.")

            st.divider()

            # ── Raw JSON Inspection (Feature #18) ──
            with st.expander("🛠️ Raw Backend API Response JSON"):
                st.json(analysis)
