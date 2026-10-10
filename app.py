import os
from dotenv import load_dotenv
load_dotenv(override=True)

import io
import json
import time
import glob
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pymupdf
import docx

# Optional direct backend imports as resilient fallback
import jd_index
from analysis_pipeline import AnalysisPipeline
from report_generator import generate_markdown_report

# 5MB upload limit
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Page configuration
st.set_page_config(
    page_title="AI Placement Analyzer - Test Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Session State Initialization
if "resumes" not in st.session_state:
    st.session_state.resumes = [] # list of dicts: {"id", "name", "text", "words", "status", "warning"}
if "jds" not in st.session_state:
    st.session_state.jds = []     # list of dicts: {"id", "title", "company", "location", "text", "words", "status", "warning", "backend_id"}
if "rank_results" not in st.session_state:
    st.session_state.rank_results = None
if "selected_jd_id" not in st.session_state:
    st.session_state.selected_jd_id = None
if "analyses" not in st.session_state:
    st.session_state.analyses = {} # keyed by jd_id
if "selected_resume_idx" not in st.session_state:
    st.session_state.selected_resume_idx = 0
if "pref_location" not in st.session_state:
    st.session_state.pref_location = ""
if "pref_arrangement" not in st.session_state:
    st.session_state.pref_arrangement = "Any"
if "active_ranked_resume_id" not in st.session_state:
    st.session_state.active_ranked_resume_id = None


# Helper Functions: Document Parsing

def extract_text_from_file(uploaded_file) -> str:
    """Extracts raw text from PDF, DOCX, or TXT uploaded files."""
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
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


# Backend Communication (API with Direct Fallback)

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
            timeout=15.0
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


from typing import Optional


def api_rank(
    resume_text: str,
    resume_id: str,
    top_n: Optional[int] = None,
    preferred_location: Optional[str] = None,
    preferred_arrangement: Optional[str] = None
) -> dict:
    """Ranks stored JDs."""
    payload = {"resume_text": resume_text, "resume_id": resume_id}
    if top_n is not None:
        payload["top_n"] = top_n
    else:
        payload["top_n"] = None
    if preferred_location:
        payload["preferred_location"] = preferred_location
    if preferred_arrangement:
        payload["preferred_arrangement"] = preferred_arrangement
    try:
        r = requests.post(
            f"{API_BASE_URL}/rank",
            json=payload,
            timeout=10.0
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    import ranking
    return ranking.rank_jds(
        resume_text,
        resume_id,
        top_n=top_n,
        preferred_location=preferred_location,
        preferred_arrangement=preferred_arrangement
    ).model_dump()


@st.cache_resource
def get_local_pipeline():
    return AnalysisPipeline()


def api_analyze(resume_text: str, resume_id: str, jd_id: str) -> dict:
    """Runs detailed Tier 2 analysis."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/analyze",
            json={"resume_text": resume_text, "resume_id": resume_id, "jd_id": jd_id},
            timeout=90.0
        )
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 404:
            return {"error": "JD_NOT_FOUND", "detail": r.json().get("detail")}
    except Exception:
        pass

    pipeline = get_local_pipeline()
    return pipeline.analyze(resume_text, resume_id, jd_id)


# Charting Helpers

def make_gauge_chart(score: int) -> go.Figure:
    """Creates a circular gauge chart for match score."""
    color = "#2ecc71" if score >= 75 else ("#f39c12" if score >= 50 else "#e74c3c")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0.08, 0.92], "y": [0, 1]},
        title={"text": "Overall Match Score", "font": {"size": 18}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#bdc3c7",
                "tickmode": "array",
                "tickvals": [0, 25, 50, 75, 100],
                "ticktext": ["0", "25", "50", "75", "100"]
            },
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
    fig.update_layout(height=240, margin=dict(l=35, r=45, t=40, b=25))
    return fig



def make_skill_donut_chart(skill_gap: dict) -> go.Figure:
    """Donut chart for skill match categories."""
    if not isinstance(skill_gap, dict):
        skill_gap = {}
    labels = ["Matched", "Missing (Essential)", "Missing (Preferred)", "Candidate Extra"]
    values = [
        len(skill_gap.get("matched", []) or []),
        len(skill_gap.get("missing_essential", []) or []),
        len(skill_gap.get("missing_preferred", []) or []),
        len(skill_gap.get("extra", []) or [])
    ]
    if sum(values) == 0:
        fig = go.Figure()
        fig.update_layout(
            title_text="No skill composition data available",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=280
        )
        return fig

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
    if not req_matches:
        fig = go.Figure()
        fig.update_layout(
            title_text="No requirement data available",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=280
        )
        return fig

    categories = [r.get("category", "") if isinstance(r, dict) else str(r) for r in req_matches]
    percentages = [r.get("match_percent", 0) if isinstance(r, dict) else 0 for r in req_matches]
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
    if not isinstance(skill_gap, dict):
        skill_gap = {}
    matched_count = len(skill_gap.get("matched", []) or [])
    miss_ess = len(skill_gap.get("missing_essential", []) or [])
    miss_pref = len(skill_gap.get("missing_preferred", []) or [])
    extra_count = len(skill_gap.get("extra", []) or [])

    candidate_total = matched_count + extra_count
    jd_total = matched_count + miss_ess + miss_pref

    if candidate_total == 0 and jd_total == 0:
        fig = go.Figure()
        fig.update_layout(
            title_text="No skill volume data available",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=280
        )
        return fig

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


def get_analysis_cache_key(resume_id: str, jd_id: str) -> str:
    """Returns a composite cache key to strictly isolate analyses across candidate resumes."""
    return f"{str(resume_id).strip()}::{str(jd_id).strip()}"


def get_cached_analysis(resume_id: str, jd_id: str) -> Optional[dict]:
    """Retrieves cached analysis using composite key with backward-compatible lookup."""
    key = get_analysis_cache_key(resume_id, jd_id)
    if "analyses" in st.session_state and key in st.session_state.analyses:
        return st.session_state.analyses[key]
    if "analyses" in st.session_state and jd_id in st.session_state.analyses:
        legacy = st.session_state.analyses[jd_id]
        if isinstance(legacy, dict) and legacy.get("resume_id") == str(resume_id).strip():
            return legacy
    return None


def set_cached_analysis(resume_id: str, jd_id: str, analysis: dict):
    """Caches analysis using composite key and keeps active pointer."""
    if "analyses" not in st.session_state:
        st.session_state.analyses = {}
    key = get_analysis_cache_key(resume_id, jd_id)
    st.session_state.analyses[key] = analysis
    st.session_state.analyses[jd_id] = analysis


def matches_work_arrangement(location: str, chk_remote: bool, chk_hybrid: bool, chk_onsite: bool) -> bool:
    """
    Evaluates whether a JD's location matches active work arrangement checkboxes.
    If none are selected, all arrangements pass (returns True).
    """
    if not (chk_remote or chk_hybrid or chk_onsite):
        return True

    loc_lower = (location or "").lower()
    is_remote = "remote" in loc_lower
    is_hybrid = "hybrid" in loc_lower
    onsite_keywords = ["on-site", "onsite", "in-person", "in person", "office", "in office"]
    is_onsite = any(kw in loc_lower for kw in onsite_keywords) or (not is_remote and not is_hybrid and bool(loc_lower.strip()))

    if chk_remote and is_remote:
        return True
    if chk_hybrid and is_hybrid:
        return True
    if chk_onsite and is_onsite:
        return True

    return False


def make_multi_jd_grouped_bar(analyses_map: dict, jd_titles_map: dict) -> go.Figure:
    """Grouped bar chart for Multi-JD Comparison across requirement categories with empty guards."""
    categories = ["Skills", "Experience", "Education", "Responsibilities"]
    fig = go.Figure()
    if not analyses_map:
        fig.update_layout(
            title_text="No JD analysis data available to compare",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=200
        )
        return fig

    colors = ["#3498db", "#2ecc71", "#e67e22"]
    valid_count = 0

    for idx, (jd_id, analysis) in enumerate(analyses_map.items()):
        if not isinstance(analysis, dict):
            continue
        valid_count += 1
        jd_label = jd_titles_map.get(jd_id, jd_id)
        raw_matches = analysis.get("requirement_match", [])
        req_matches = {r.get("category"): r.get("match_percent", 0) for r in raw_matches if isinstance(r, dict)}
        y_vals = [req_matches.get(cat, 0) for cat in categories]
        color = colors[idx % len(colors)]
        fig.add_trace(go.Bar(
            name=jd_label,
            x=categories,
            y=y_vals,
            marker_color=color,
            text=[f"{v}%" for v in y_vals],
            textposition="auto"
        ))

    if valid_count == 0:
        fig.update_layout(
            title_text="No valid requirement data found across selected JDs",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=200
        )
        return fig

    fig.update_layout(
        barmode="group",
        title_text="Requirement Alignment Comparison (%)",
        yaxis=dict(range=[0, 100], title="Match %"),
        xaxis=dict(title="Category"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig


def aggregate_skill_gaps(analyses: list[dict]) -> tuple[dict[str, int], dict[str, int], dict[str, int], int]:
    """
    Aggregates essential missing, preferred missing, and matched skills across valid analyses.
    Returns: (essential_counts, preferred_counts, matched_counts, successful_count)
    """
    essential_counts: dict[str, int] = {}
    preferred_counts: dict[str, int] = {}
    matched_counts: dict[str, int] = {}
    successful_count = 0

    for a in analyses:
        if not isinstance(a, dict) or not a.get("context_sufficient", True) or "error" in a or "error_code" in a:
            continue
        successful_count += 1
        sg = a.get("skill_gap", {})
        for s in sg.get("missing_essential", []):
            essential_counts[s] = essential_counts.get(s, 0) + 1
        for s in sg.get("missing_preferred", []):
            preferred_counts[s] = preferred_counts.get(s, 0) + 1
        for m in sg.get("matched", []):
            s_name = m.get("skill_name", str(m)) if isinstance(m, dict) else str(m)
            matched_counts[s_name] = matched_counts.get(s_name, 0) + 1

    return essential_counts, preferred_counts, matched_counts, successful_count


def make_skill_gap_dashboard_chart(essential_counts: dict[str, int], preferred_counts: dict[str, int]) -> go.Figure:
    """
    Horizontal bar chart for most frequently missing skills.
    Color = red for essential, orange for preferred.
    Sorted by count descending.
    """
    all_missing = list(set(list(essential_counts.keys()) + list(preferred_counts.keys())))
    if not all_missing:
        fig = go.Figure()
        fig.update_layout(
            title_text="No skill gaps found across analyzed JDs",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=200
        )
        return fig

    # Sort descending by essential count, then preferred count
    all_missing.sort(key=lambda s: (essential_counts.get(s, 0), preferred_counts.get(s, 0)), reverse=True)
    top_missing = all_missing[:15]
    top_missing_rev = list(reversed(top_missing))

    ess_vals = [essential_counts.get(s, 0) for s in top_missing_rev]
    pref_vals = [preferred_counts.get(s, 0) for s in top_missing_rev]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_missing_rev,
        x=ess_vals,
        name="Missing (Essential)",
        orientation="h",
        marker_color="#e74c3c",
        text=[str(v) if v > 0 else "" for v in ess_vals],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        y=top_missing_rev,
        x=pref_vals,
        name="Missing (Preferred)",
        orientation="h",
        marker_color="#f39c12",
        text=[str(v) if v > 0 else "" for v in pref_vals],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title_text="Most Frequently Missing Skills Across Analyzed Roles",
        xaxis=dict(title="Number of Job Descriptions", dtick=1),
        yaxis=dict(title="Skill Name"),
        height=max(360, len(top_missing_rev) * 32 + 100),
        margin=dict(l=120, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


# SIDEBAR: System Status & Session Controls

with st.sidebar:
    st.title("AI Placement Analyzer")
    st.caption("Phase 1 Testing Dashboard (Main Branch)")
    st.divider()

    # Backend status badge
    backend_ok = check_backend_alive()
    if backend_ok:
        st.success("[Connected] FastAPI Backend (127.0.0.1:8000)")
    else:
        st.warning("[Offline] Backend Server - using Direct Pipeline fallback")

    st.subheader("Session Counters")
    col1, col2 = st.columns(2)
    col1.metric("Resumes", f"{len(st.session_state.resumes)} / 10")
    col2.metric("JDs", f"{len(st.session_state.jds)} / 10")

    st.progress(len(st.session_state.resumes) / 10, text="Resume Cap")
    st.progress(len(st.session_state.jds) / 10, text="JD Cap")

    st.divider()

    # Reset Action
    if st.button("Reset Session and Clear Store", use_container_width=True, type="secondary"):
        api_reset_jds()
        st.session_state.resumes = []
        st.session_state.jds = []
        st.session_state.rank_results = None
        st.session_state.selected_jd_id = None
        st.session_state.analyses = {}
        st.session_state.selected_resume_idx = 0
        st.session_state.pref_location = ""
        st.session_state.pref_arrangement = "Any"
        st.success("Session reset! Store cleared.")
        st.rerun()

    # Preferences Section (Feature 6)
    st.divider()
    st.subheader("Preferences")
    st.session_state.pref_location = st.text_input(
        "Preferred Location",
        value=st.session_state.pref_location,
        placeholder="e.g., Remote, New York, London",
        key="pref_location_input"
    )
    arr_options = ["Any", "Remote", "Hybrid", "On-site"]
    arr_idx = arr_options.index(st.session_state.pref_arrangement) if st.session_state.pref_arrangement in arr_options else 0
    st.session_state.pref_arrangement = st.selectbox(
        "Work Arrangement",
        arr_options,
        index=arr_idx,
        key="pref_arrangement_input"
    )

    # Quick Fixture Loader
    st.divider()
    st.subheader("Quick Test Fixtures")

    # Load from Sample PDFs
    if st.button("Load 5 PDF Resumes and 5 PDF JDs", use_container_width=True):
        resume_pdfs = sorted(glob.glob("sample_pdfs/resumes/*.pdf"))
        jd_pdfs = sorted(glob.glob("sample_pdfs/job_descriptions/*.pdf"))

        if resume_pdfs and jd_pdfs:
            st.session_state.resumes = []
            for path in resume_pdfs[:10]:
                with pymupdf.open(path) as doc:
                    raw_text = "\n".join([page.get_text() for page in doc]).strip()
                status, warn = validate_doc_text(raw_text)
                name = os.path.basename(path).replace(".pdf", "").replace("_", " ")
                st.session_state.resumes.append({
                    "id": f"res_{int(time.time()*1000)%100000}_{len(st.session_state.resumes)}",
                    "name": name,
                    "text": raw_text,
                    "words": len(raw_text.split()),
                    "status": status,
                    "warning": warn
                })

            api_reset_jds()
            st.session_state.jds = []
            for path in jd_pdfs[:10]:
                with pymupdf.open(path) as doc:
                    raw_text = "\n".join([page.get_text() for page in doc]).strip()
                status, warn = validate_doc_text(raw_text)
                title = os.path.basename(path).replace(".pdf", "").replace("JD_", "").replace("_", " ")
                st.session_state.jds.append({
                    "id": f"jd_{int(time.time()*1000)%100000}_{len(st.session_state.jds)}",
                    "title": title,
                    "company": "Partner Company",
                    "location": "Bengaluru / Remote",
                    "text": raw_text,
                    "words": len(raw_text.split()),
                    "status": status,
                    "warning": warn,
                    "backend_id": None
                })
            st.success("Loaded 5 PDF Resumes and 5 PDF Job Descriptions!")
            st.rerun()
        else:
            st.error("Sample PDF directory not found. Please run the PDF generator.")

    if st.button("Load JSON Fixtures (5 Resumes / 5 JDs)", use_container_width=True):
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
            st.success("JSON Fixtures loaded successfully.")
            st.rerun()


# MAIN PANEL

tab_upload, tab_results, tab_detail, tab_compare, tab_dashboard = st.tabs([
    "1. Upload and Manage Documents",
    "2. Ranked Matching Results",
    "3. In-Depth Gap Analysis",
    "4. Compare JDs",
    "5. Skill Gap Dashboard"
])


# TAB 1: RESUME & JD UPLOAD PANEL

with tab_upload:
    st.header("Upload and Document Management Panel")
    st.markdown("Upload candidate resumes and job descriptions (Max 10 documents each, **5MB max file size**).")

    col_res, col_jd = st.columns(2)

    # Left Column: Resumes
    with col_res:
        st.subheader("Candidate Resumes (Max 10, Limit 5MB)")
        res_upload_mode = st.radio("Resume Upload Mode", ["Files (PDF / DOCX / TXT)", "Manual Paste"], horizontal=True, key="res_mode")

        if res_upload_mode == "Files (PDF / DOCX / TXT)":
            res_files = st.file_uploader(
                "Select Resume Files (Max 5MB each)",
                type=["pdf", "docx", "doc", "txt"],
                accept_multiple_files=True,
                key="res_file_input"
            )
            if res_files and st.button("Add Uploaded Resumes"):
                for f in res_files:
                    if len(st.session_state.resumes) >= 10:
                        st.warning("Resume cap reached (10 max).")
                        break
                    if f.size > MAX_FILE_SIZE_BYTES:
                        st.error(f"File '{f.name}' exceeds the 5MB size limit ({f.size / (1024*1024):.2f}MB). Upload rejected.")
                        continue
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
                submit_res = st.form_submit_button("Add Manual Resume")
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

        # Preview Resumes
        st.markdown(f"**Uploaded Resumes ({len(st.session_state.resumes)}/10)**")
        if not st.session_state.resumes:
            st.info("No resumes uploaded yet.")
        for idx, r in enumerate(st.session_state.resumes):
            with st.expander(f"{r['name']} ({r['words']} words) - Status: {r['status']}"):
                if r['status'] == "Valid":
                    st.success(f"Status: {r['status']} ({r['warning']})")
                else:
                    st.warning(f"Status: {r['status']} - {r['warning']}")
                st.text_area("Content Preview", r['text'][:400] + ("..." if len(r['text']) > 400 else ""), height=100, key=f"prev_r_{idx}")
                if st.button("Remove Resume", key=f"del_r_{idx}"):
                    st.session_state.resumes.pop(idx)
                    st.rerun()

    # Right Column: Job Descriptions
    with col_jd:
        st.subheader("Job Descriptions (Max 10, Limit 5MB)")
        jd_upload_mode = st.radio("JD Upload Mode", ["Files (PDF / DOCX / TXT)", "Manual Form"], horizontal=True, key="jd_mode")

        if jd_upload_mode == "Files (PDF / DOCX / TXT)":
            jd_files = st.file_uploader(
                "Select JD Files (Max 5MB each)",
                type=["pdf", "docx", "doc", "txt"],
                accept_multiple_files=True,
                key="jd_file_input"
            )
            if jd_files and st.button("Add Uploaded JDs"):
                for f in jd_files:
                    if len(st.session_state.jds) >= 10:
                        st.warning("JD cap reached (10 max).")
                        break
                    if f.size > MAX_FILE_SIZE_BYTES:
                        st.error(f"File '{f.name}' exceeds the 5MB size limit ({f.size / (1024*1024):.2f}MB). Upload rejected.")
                        continue
                    try:
                        raw_text = extract_text_from_file(f)
                        status, warn = validate_doc_text(raw_text)
                        title = f.name.replace(".pdf", "").replace(".docx", "").replace(".txt", "")
                        st.session_state.jds.append({
                            "id": f"jd_{int(time.time()*1000)%100000}",
                            "title": title.title(),
                            "company": "Partner Company",
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
                submit_jd = st.form_submit_button("Add Manual JD")
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

        # Preview JDs
        st.markdown(f"**Ingested Job Descriptions ({len(st.session_state.jds)}/10)**")
        if not st.session_state.jds:
            st.info("No JDs added yet.")
        for idx, j in enumerate(st.session_state.jds):
            with st.expander(f"{j['title']} @ {j['company']} ({j['words']} words) - {j['status']}"):
                if j['status'] == "Valid":
                    st.success(f"Status: {j['status']} ({j['warning']})")
                else:
                    st.warning(f"Status: {j['status']} - {j['warning']}")
                st.text_area("Content Preview", j['text'][:400] + ("..." if len(j['text']) > 400 else ""), height=100, key=f"prev_j_{idx}")
                if st.button("Remove JD", key=f"del_j_{idx}"):
                    st.session_state.jds.pop(idx)
                    st.rerun()

    # Action Bar: Analysis Trigger
    st.divider()
    st.subheader("Run Alignment Analysis")

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

        if st.button("Analyze Placement Fit (Embed and Rank)", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Step 1/5: Uploading and Parsing Documents...")
            time.sleep(0.3)

            try:
                # Step 1: Upload / Ingest JDs to Backend
                progress_bar.progress(25, text="Step 2/5: Ingesting and Embedding Job Descriptions...")
                api_reset_jds()
                for j in st.session_state.jds:
                    res = api_add_jd(j["title"], j["company"], j["location"], j["text"])
                    if "jd_id" in res:
                        j["backend_id"] = res["jd_id"]
                    elif "error" in res:
                        st.error(f"Error adding JD '{j['title']}': {res.get('detail')}")

                # Step 2: Tier 1 Ranking
                progress_bar.progress(50, text="Step 3/5: Vectorizing Resume and Running Cosine Similarity Matching...")
                pref_loc = st.session_state.pref_location.strip() if st.session_state.pref_location.strip() else None
                pref_arr = st.session_state.pref_arrangement if st.session_state.pref_arrangement != "Any" else None
                rank_resp = api_rank(
                    target_resume["text"],
                    target_resume["id"],
                    top_n=None,
                    preferred_location=pref_loc,
                    preferred_arrangement=pref_arr
                )
                st.session_state.rank_results = rank_resp.get("results", [])
                st.session_state.active_ranked_resume_id = target_resume["id"]

                progress_bar.progress(75, text="Step 4/5: Ranking Matches (Zero Gemini Calls)...")
                time.sleep(0.3)

                # Step 3: Trigger Tier 2 analysis on top match if available
                if st.session_state.rank_results:
                    top_jd = st.session_state.rank_results[0]
                    top_id = top_jd["jd_id"]
                    progress_bar.progress(90, text=f"Step 5/5: Running Gemini Deep Analysis on #1 match ({top_jd['title']})...")
                    try:
                        analysis_res = api_analyze(target_resume["text"], target_resume["id"], top_id)
                        set_cached_analysis(target_resume["id"], top_id, analysis_res)
                        st.session_state.selected_jd_id = top_id
                    except Exception as exc:
                        st.warning(f"Could not automatically analyze #1 match: {exc}")

                progress_bar.progress(100, text="Complete! Results ready.")
                st.success("Analysis complete! View the results in Tab 2 and Tab 3.")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")


# TAB 2: RANKED JD RESULTS

with tab_results:
    st.header("Ranked Job Descriptions")

    if not st.session_state.rank_results:
        st.info("No ranking results yet. Go to Tab 1 and click 'Analyze Placement Fit'.")
    else:
        target_res = st.session_state.resumes[st.session_state.selected_resume_idx]
        st.markdown(f"**Target Resume:** `{target_res['name']}` ({target_res['words']} words)")

        # Feature 1: Top-N Selector
        top_n_choice = st.radio(
            "Number of Results to Display",
            ["Top 5", "Top 10", "All"],
            index=0,
            horizontal=True,
            key="top_n_choice"
        )

        # Feature 2: Filters and Sorting
        with st.expander("Filter and Sort Results", expanded=True):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                loc_filter = st.text_input(
                    "Filter by Location",
                    value="",
                    placeholder="Search location (e.g. Remote, London)...",
                    key="loc_filter_input"
                )
                st.write("Work Arrangement:")
                col_w1, col_w2, col_w3 = st.columns(3)
                chk_remote = col_w1.checkbox("Remote", value=False, key="chk_filter_remote")
                chk_hybrid = col_w2.checkbox("Hybrid", value=False, key="chk_filter_hybrid")
                chk_onsite = col_w3.checkbox("On-site", value=False, key="chk_filter_onsite")

            with col_f2:
                min_score = st.slider(
                    "Minimum Similarity Score",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1,
                    key="min_score_filter"
                )
                sort_choice = st.radio(
                    "Sort By",
                    ["Score (high to low)", "Score (low to high)", "Location (A–Z)"],
                    index=0,
                    horizontal=True,
                    key="sort_choice_filter"
                )

        # Apply filtering
        filtered_cards = []
        for card in st.session_state.rank_results:
            card_loc = card.get("location") or ""
            card_score = card.get("similarity_score", 0)

            # Location filter
            if loc_filter.strip() and loc_filter.strip().lower() not in card_loc.lower():
                continue

            # Work arrangement filter
            if not matches_work_arrangement(card_loc, chk_remote, chk_hybrid, chk_onsite):
                continue

            # Minimum similarity filter
            if card_score < min_score:
                continue

            filtered_cards.append(card)

        # Apply sorting (deterministic tie-breaking with jd_id)
        if sort_choice == "Score (high to low)":
            filtered_cards.sort(key=lambda c: (-c.get("similarity_score", 0), str(c.get("jd_id", ""))))
        elif sort_choice == "Score (low to high)":
            filtered_cards.sort(key=lambda c: (c.get("similarity_score", 0), str(c.get("jd_id", ""))))
        elif sort_choice == "Location (A–Z)":
            filtered_cards.sort(key=lambda c: (c.get("location", "").lower(), -c.get("similarity_score", 0), str(c.get("jd_id", ""))))

        # Apply Top-N selection
        if top_n_choice == "Top 5":
            visible_cards = filtered_cards[:5]
        elif top_n_choice == "Top 10":
            visible_cards = filtered_cards[:10]
        else:
            visible_cards = filtered_cards

        st.markdown(f"**Showing {len(visible_cards)} of {len(st.session_state.rank_results)} results**")

        if not visible_cards:
            st.info("No results match your filters.")
        else:
            for card in visible_cards:
                rank_num = card.get("rank", 1)
                sim_score = card.get("similarity_score", 0)
                rank_label = f"#{rank_num}"
                jd_id = card.get("jd_id")

                with st.container():
                    st.markdown(f"### {rank_label} {card.get('title')} - {card.get('company')}")
                    c1, c2, c3 = st.columns([2, 5, 3])

                    with c1:
                        st.metric("Similarity Score", f"{sim_score}%")
                        st.caption(f"Location: {card.get('location')}")
                        card_loc = (card.get("location") or "").lower()
                        if st.session_state.pref_location.strip() and st.session_state.pref_location.strip().lower() in card_loc:
                            st.markdown("`Location match +5`")
                        if st.session_state.pref_arrangement in ["Remote", "Hybrid", "On-site"] and st.session_state.pref_arrangement.lower() in card_loc:
                            st.markdown("`Arrangement match +3`")

                    with c2:
                        st.markdown("**Overview Snippet:**")
                        st.write(card.get("snippet", ""))

                    with c3:
                        st.markdown("**Actions:**")
                        if st.button(f"View Full Gap Analysis", key=f"btn_analyze_{jd_id}"):
                            cached_data = get_cached_analysis(target_res["id"], jd_id)
                            if not cached_data:
                                with st.spinner(f"Running Gemini Tier 2 analysis on {card.get('title')}..."):
                                    try:
                                        analysis_data = api_analyze(target_res["text"], target_res["id"], jd_id)
                                        set_cached_analysis(target_res["id"], jd_id, analysis_data)
                                        st.session_state.selected_jd_id = jd_id
                                        st.success(f"Loaded analysis for {card.get('title')}! Switch to Tab 3.")
                                    except Exception as exc:
                                        st.error(f"Analysis failed for {card.get('title')}: {exc}")
                            else:
                                st.session_state.selected_jd_id = jd_id
                                st.success(f"Loaded analysis for {card.get('title')}! Switch to Tab 3.")

                    st.divider()


# TAB 3: IN-DEPTH GAP ANALYSIS

with tab_detail:
    st.header("Detailed Match and Skill Gap Analysis")

    if not st.session_state.resumes or not (0 <= st.session_state.selected_resume_idx < len(st.session_state.resumes)):
        st.info("Upload a candidate resume and run placement fit in Tab 1.")
    else:
        target_res = st.session_state.resumes[st.session_state.selected_resume_idx]
        active_jd_id = st.session_state.selected_jd_id
        analysis = get_cached_analysis(target_res["id"], active_jd_id) if active_jd_id else None

        if not active_jd_id or not analysis:
            st.info("Select a job description in Tab 2 by clicking 'View Full Gap Analysis' to inspect deep results.")
        else:
            # Check Insufficient Context State
            if not analysis.get("context_sufficient", True):
                st.error("Insufficient Context Detected")
                st.warning(f"Reason: {analysis.get('insufficient_reason', 'Document too short')}")
                st.info("The backend determined that either the resume or job description does not contain sufficient text for statistically reliable skill grounding. Automated generation was safely bypassed without errors.")

                # Feature 3: Download Report for insufficient context
                target_res_name = target_res.get("name", "Candidate")
                jd_title_found = f"Job ({active_jd_id})"
                for j in st.session_state.jds:
                    if j.get("id") == active_jd_id or j.get("backend_id") == active_jd_id:
                        jd_title_found = j.get("title", jd_title_found)
                        break

                res_id_clean = analysis.get("resume_id", target_res["id"])
                report_md = generate_markdown_report(analysis, target_res_name, jd_title_found)
                report_filename = f"gap_analysis_{res_id_clean}_{active_jd_id}.md"

                st.download_button(
                    label="Download Report (.md)",
                    data=report_md,
                    file_name=report_filename,
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"dl_btn_insuff_{active_jd_id}"
                )
            else:
                # Top Row: Gauge & Scores
                score = analysis.get("match_score", 0)
                score_col, summary_col = st.columns([1, 2])

                with score_col:
                    st.plotly_chart(make_gauge_chart(score), use_container_width=True)

                with summary_col:
                    st.subheader("Executive Fit Summary")
                    st.markdown(f"> {analysis.get('summary', 'No summary available.')}")
                    st.caption(f"JD ID: {active_jd_id} | Resume ID: {analysis.get('resume_id')} | Context Sufficient: True")

                    # Feature 3: Download Report
                    target_res_name = target_res.get("name", "Candidate")

                    jd_title_found = f"Job ({active_jd_id})"
                    for j in st.session_state.jds:
                        if j.get("id") == active_jd_id or j.get("backend_id") == active_jd_id:
                            jd_title_found = j.get("title", jd_title_found)
                            break

                    res_id_clean = analysis.get("resume_id", target_res["id"])
                    report_md = generate_markdown_report(analysis, target_res_name, jd_title_found)
                    report_filename = f"gap_analysis_{res_id_clean}_{active_jd_id}.md"

                    st.download_button(
                        label="Download Report (.md)",
                        data=report_md,
                        file_name=report_filename,
                        mime="text/markdown",
                        use_container_width=True,
                        key=f"dl_btn_{active_jd_id}"
                    )

                st.divider()

            # Row 2: Strengths & Weaknesses
            st.subheader("Strengths and Qualification Gaps")
            str_col, weak_col = st.columns(2)

            with str_col:
                st.markdown("#### Key Strengths")
                strengths = analysis.get("strengths", [])
                if strengths:
                    for s in strengths:
                        st.markdown(f"- {s}")
                else:
                    st.write("No specific strengths listed.")

            with weak_col:
                st.markdown("#### Areas for Improvement / Gaps")
                weaknesses = analysis.get("weaknesses", [])
                if weaknesses:
                    for w in weaknesses:
                        st.markdown(f"- {w}")
                else:
                    st.write("No major gaps identified.")

            st.divider()

            # Row 3: Interactive Visualizations
            st.subheader("Visual Alignment Analytics")
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

            # Row 4: Skill Gap Breakdown
            st.subheader("Grounded Skill Gap Breakdown")

            matched_skills = gap_data.get("matched", [])
            miss_essential = gap_data.get("missing_essential", [])
            miss_preferred = gap_data.get("missing_preferred", [])
            extra_skills = gap_data.get("extra", [])

            g_c1, g_c2, g_c3, g_c4 = st.columns(4)

            with g_c1:
                st.markdown(f"**Matched Skills ({len(matched_skills)})**")
                if matched_skills:
                    for m in matched_skills:
                        st.markdown(f"- {m.get('skill_name')}")
                else:
                    st.write("None")

            with g_c2:
                st.markdown(f"**Missing Essential ({len(miss_essential)})**")
                if miss_essential:
                    for s in miss_essential:
                        st.markdown(f"- {s}")
                else:
                    st.write("None")

            with g_c3:
                st.markdown(f"**Missing Preferred ({len(miss_preferred)})**")
                if miss_preferred:
                    for s in miss_preferred:
                        st.markdown(f"- {s}")
                else:
                    st.write("None")

            with g_c4:
                st.markdown(f"**Candidate Extra ({len(extra_skills)})**")
                if extra_skills:
                    for s in extra_skills:
                        st.markdown(f"- {s}")
                else:
                    st.write("None")

            st.divider()

            # Row 5: Actionable Skill Recommendations
            st.subheader("Targeted Skill Recommendations")
            recs = analysis.get("recommendations", [])
            if recs:
                r_cols = st.columns(min(len(recs), 3))
                for i, r in enumerate(recs):
                    col_idx = i % len(r_cols)
                    with r_cols[col_idx]:
                        with st.container(border=True):
                            st.markdown(f"**{r.get('skill', 'Skill')}**")
                            st.write(r.get("reason", ""))
            else:
                st.info("No specific recommendations generated.")

            st.divider()

            # Raw JSON Inspection
            with st.expander("Raw Backend API Response JSON"):
                st.json(analysis)


# TAB 4: MULTI-JD COMPARISON (Feature 4)

with tab_compare:
    st.header("Multi-JD Comparison")
    st.markdown("Select 2 or 3 job descriptions from ranked results to view side-by-side gap analysis, alignment scores, and skill overlap.")

    if not st.session_state.rank_results:
        st.info("No ranking results available. Run ranking in Tab 1 first.")
    else:
        target_res = st.session_state.resumes[st.session_state.selected_resume_idx]
        st.markdown(f"**Target Candidate:** `{target_res['name']}` ({target_res['words']} words)")

        # Prepare JD options from ranked results
        jd_options_map = {}
        for c in st.session_state.rank_results:
            label = f"#{c.get('rank', 1)} {c.get('title')} @ {c.get('company')} ({c.get('jd_id')})"
            jd_options_map[label] = c.get("jd_id")

        all_option_keys = list(jd_options_map.keys())
        default_selected = all_option_keys[:min(2, len(all_option_keys))]

        selected_labels = st.multiselect(
            "Select 2 or 3 Job Descriptions to Compare",
            options=all_option_keys,
            default=default_selected,
            max_selections=3,
            key="compare_jd_multiselect"
        )

        if len(selected_labels) > 3:
            st.warning("Please select at most 3 JDs. Showing only the first 3 selected.")
            selected_labels = selected_labels[:3]

        if len(selected_labels) < 2:
            st.info("Please select at least 2 job descriptions (maximum 3) to view comparison.")
        else:
            selected_ids = [jd_options_map[lbl] for lbl in selected_labels]

            col_btn, _ = st.columns([2, 5])
            with col_btn:
                btn_run_comp = st.button("Compare Selected JDs", type="primary", use_container_width=True, key="btn_run_compare")

            if btn_run_comp:
                # Analyze selected JDs if not already cached
                for jd_id in selected_ids:
                    if not get_cached_analysis(target_res["id"], jd_id):
                        with st.spinner(f"Analyzing {jd_id}..."):
                            try:
                                analysis_res = api_analyze(target_res["text"], target_res["id"], jd_id)
                                set_cached_analysis(target_res["id"], jd_id, analysis_res)
                            except Exception as exc:
                                err_record = {
                                    "resume_id": target_res["id"],
                                    "jd_id": jd_id,
                                    "context_sufficient": False,
                                    "error": str(exc),
                                    "insufficient_reason": f"Analysis failed: {exc}"
                                }
                                set_cached_analysis(target_res["id"], jd_id, err_record)
                                st.error(f"Analysis failed for {jd_id}: {exc}")

            # Check if all selected JDs have been analyzed
            all_analyzed = all(get_cached_analysis(target_res["id"], jid) is not None for jid in selected_ids)
            if not all_analyzed:
                st.info("Click 'Compare Selected JDs' to run gap analysis for unanalyzed roles.")
            else:
                st.divider()

                # Build data maps
                valid_analyses = {}
                failed_jds = {}
                title_map = {}

                for lbl in selected_labels:
                    jid = jd_options_map[lbl]
                    rec = get_cached_analysis(target_res["id"], jid) or {}
                    title_map[jid] = lbl.split("(")[0].strip()

                    if isinstance(rec, dict) and rec.get("context_sufficient", True) and "error" not in rec and "error_code" not in rec and "detail" not in rec:
                        valid_analyses[jid] = rec
                    else:
                        err_msg = rec.get("insufficient_reason") or rec.get("error") or rec.get("detail") or "Analysis failed or insufficient context"
                        failed_jds[jid] = str(err_msg)

                # 1. Summary Table
                st.subheader("Comparison Summary Table")
                rows = [
                    "Match Score",
                    "Matched Skills Count",
                    "Missing Essential Count",
                    "Missing Preferred Count",
                    "Extra Skills Count"
                ]

                table_data = {"Metric": rows}
                for jid in selected_ids:
                    col_name = title_map[jid]
                    if jid in valid_analyses:
                        a = valid_analyses[jid]
                        sg = a.get("skill_gap") or {}
                        table_data[col_name] = [
                            f"{a.get('match_score', 0)}%",
                            str(len(sg.get("matched", []) or [])),
                            str(len(sg.get("missing_essential", []) or [])),
                            str(len(sg.get("missing_preferred", []) or [])),
                            str(len(sg.get("extra", []) or []))
                        ]
                    else:
                        table_data[col_name] = [
                            f"Error / Insufficient: {str(failed_jds.get(jid, 'Failed'))[:30]}",
                            "N/A", "N/A", "N/A", "N/A"
                        ]

                import pandas as pd
                df_summary = pd.DataFrame(table_data)
                st.table(df_summary)

                # 2. Grouped Bar Chart
                if valid_analyses:
                    st.subheader("Requirement Alignment Comparison")
                    fig_grouped = make_multi_jd_grouped_bar(valid_analyses, title_map)
                    st.plotly_chart(fig_grouped, use_container_width=True)

                # 3. Skills Overlap / Side-by-side View
                st.subheader("Side-by-Side Skill Gaps and Overlap")
                cols = st.columns(len(selected_ids))

                for idx, jid in enumerate(selected_ids):
                    with cols[idx]:
                        with st.container(border=True):
                            st.markdown(f"**{title_map[jid]}**")
                            if jid in failed_jds:
                                st.error(f"Analysis failed: {failed_jds[jid]}")
                            else:
                                a = valid_analyses[jid]
                                sg = a.get("skill_gap", {})
                                matched = [m.get("skill_name", str(m)) if isinstance(m, dict) else str(m) for m in sg.get("matched", [])]
                                miss_ess = sg.get("missing_essential", [])
                                miss_pref = sg.get("missing_preferred", [])

                                st.markdown(f"**Matched Skills ({len(matched)}):**")
                                if matched:
                                    for m in matched:
                                        st.markdown(f"- {m}")
                                else:
                                    st.caption("None")

                                st.markdown(f"**Missing Essential ({len(miss_ess)}):**")
                                if miss_ess:
                                    for m in miss_ess:
                                        st.markdown(f"- {m}")
                                else:
                                    st.caption("None")

                                st.markdown(f"**Missing Preferred ({len(miss_pref)}):**")
                                if miss_pref:
                                    for m in miss_pref:
                                        st.markdown(f"- {m}")
                                else:
                                    st.caption("None")

                # Shared Missing Skills Callout
                if len(valid_analyses) >= 2:
                    sets_missing = [set(a.get("skill_gap", {}).get("missing_essential", [])) for a in valid_analyses.values()]
                    common_gaps = set.intersection(*sets_missing) if sets_missing else set()
                    st.divider()
                    if common_gaps:
                        st.info(f"Shared Missing Essential Skills across analyzed roles: {', '.join(sorted(common_gaps))}")
                    else:
                        st.caption("No shared missing essential skills between the analyzed roles.")


# TAB 5: SKILL GAP DASHBOARD (Feature 5)

with tab_dashboard:
    st.header("Skill Gap Dashboard")
    st.markdown("Aggregated analysis across ranked job descriptions highlighting most frequent missing skills and candidate strengths.")

    if not st.session_state.rank_results:
        st.info("No ranking results available. Run ranking in Tab 1 first.")
    else:
        target_res = st.session_state.resumes[st.session_state.selected_resume_idx]
        st.markdown(f"**Target Candidate:** `{target_res['name']}` ({target_res['words']} words)")

        # Determine target JDs from current Top-N selection (Feature 1)
        current_top_n = st.session_state.get("top_n_choice", "Top 5")
        if current_top_n == "Top 5":
            target_cards = st.session_state.rank_results[:5]
        elif current_top_n == "Top 10":
            target_cards = st.session_state.rank_results[:10]
        else:
            target_cards = st.session_state.rank_results

        total_target_count = len(target_cards)
        target_jd_ids = [c["jd_id"] for c in target_cards]

        # Check which JDs are uncompleted
        uncompleted_ids = [jid for jid in target_jd_ids if get_cached_analysis(target_res["id"], jid) is None]

        col_act, _ = st.columns([2, 5])
        with col_act:
            btn_run_dashboard = st.button(
                f"Generate Dashboard ({total_target_count} Roles)",
                type="primary",
                use_container_width=True,
                key="btn_run_dashboard"
            )

        # Run if button clicked or if all are already cached
        if btn_run_dashboard or (len(uncompleted_ids) == 0 and total_target_count > 0):
            if uncompleted_ids:
                p_bar = st.progress(0, text="Analyzing target roles...")
                failed_count = 0
                for idx, jid in enumerate(uncompleted_ids):
                    card_title = next((c.get("title", jid) for c in target_cards if c.get("jd_id") == jid), jid)
                    p_bar.progress(
                        int((idx / len(uncompleted_ids)) * 100),
                        text=f"Analyzing role {idx+1}/{len(uncompleted_ids)}: {card_title}..."
                    )
                    try:
                        analysis_res = api_analyze(target_res["text"], target_res["id"], jid)
                        set_cached_analysis(target_res["id"], jid, analysis_res)
                    except Exception as exc:
                        failed_count += 1
                        err_record = {
                            "resume_id": target_res["id"],
                            "jd_id": jid,
                            "context_sufficient": False,
                            "error": str(exc),
                            "insufficient_reason": f"Provider error: {exc}"
                        }
                        set_cached_analysis(target_res["id"], jid, err_record)
                p_bar.progress(100, text="Completed analyzing roles!")
                if failed_count > 0:
                    st.warning(f"Note: {failed_count} of {len(uncompleted_ids)} roles encountered errors during analysis and were skipped.")

            # Collect analyses for the target JDs
            collected = [
                get_cached_analysis(target_res["id"], jid)
                for jid in target_jd_ids
                if get_cached_analysis(target_res["id"], jid) is not None
            ]
            ess_counts, pref_counts, match_counts, success_count = aggregate_skill_gaps(collected)

            st.divider()

            # Note on analysis count and graceful failures
            if success_count == total_target_count:
                st.caption(f"Based on analysis of {total_target_count} job descriptions. ({success_count} of {total_target_count} JDs analyzed successfully.)")
            else:
                st.warning(f"Based on analysis of {total_target_count} job descriptions. ({success_count} of {total_target_count} JDs analyzed successfully; {total_target_count - success_count} failed or had insufficient context.)")

            if success_count == 0:
                st.error("No job descriptions were successfully analyzed.")
            else:
                # 1. Horizontal Bar Chart (Plotly)
                st.subheader("Frequently Missing Skills")
                chart_fig = make_skill_gap_dashboard_chart(ess_counts, pref_counts)
                st.plotly_chart(chart_fig, use_container_width=True)

                # 2. Table: Skill name | Times Missing (Essential) | Times Missing (Preferred) | % of JDs
                st.subheader("Aggregated Skill Gap Summary Table")
                all_missing_skills = sorted(
                    list(set(list(ess_counts.keys()) + list(pref_counts.keys()))),
                    key=lambda s: (ess_counts.get(s, 0), pref_counts.get(s, 0)),
                    reverse=True
                )

                if all_missing_skills:
                    import pandas as pd
                    table_rows = []
                    for s in all_missing_skills:
                        t_ess = ess_counts.get(s, 0)
                        t_pref = pref_counts.get(s, 0)
                        pct_jds = f"{round(((t_ess + t_pref) / success_count) * 100, 1)}%"
                        table_rows.append({
                            "Skill Name": s,
                            "Times Missing (Essential)": t_ess,
                            "Times Missing (Preferred)": t_pref,
                            "% of JDs": pct_jds
                        })
                    df_gaps = pd.DataFrame(table_rows)
                    st.dataframe(df_gaps, use_container_width=True, hide_index=True)
                else:
                    st.info("No skill gaps found across analyzed JDs.")

                st.divider()

                # 3. Strongest skills section
                st.subheader("Strongest Skills (Most Frequently Matched Across Roles)")
                sorted_matched = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)

                if sorted_matched:
                    top_matched = sorted_matched[:8]
                    cols_m = st.columns(min(len(top_matched), 4))
                    for idx, (skill_name, count) in enumerate(top_matched):
                        col_idx = idx % len(cols_m)
                        with cols_m[col_idx]:
                            pct_m = int(round((count / success_count) * 100))
                            st.metric(
                                label=skill_name,
                                value=f"{count}/{success_count} JDs",
                                delta=f"{pct_m}% matched"
                            )
                else:
                    st.info("No matched skills recorded across analyzed roles.")
        else:
            st.info(f"Click 'Generate Dashboard ({total_target_count} Roles)' to run batch analysis and view aggregated skills.")
