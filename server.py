from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID, uuid4
from typing import TYPE_CHECKING, Optional
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from schemas import ErrorResponse
from chat_schemas import ChatContextRequest, ChatContextResponse, ChatRequest, ChatResponse
from chat_service import ChatService, ChatServiceError

if TYPE_CHECKING:
    from query import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="AI Placement Analyzer & RAG API")

# Enable CORS for browser frontends from any origin or file protocol
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (analyzer is primary Phase 1 service, rag is lazy-loaded for Phase 2)
analyzer = None
_rag: Optional[RAGPipeline] = None
_chat = ChatService()


def get_analyzer():
    global analyzer
    if analyzer is None:
        from analysis_pipeline import AnalysisPipeline
        try:
            analyzer = AnalysisPipeline()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return analyzer


def chat_call(operation, *args):
    try:
        return operation(*args)
    except ChatServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Chat storage is unavailable. Please retry.") from exc


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_FALLBACK_API_KEY")
    return {
        "status": "ok",
        "gemini_configured": bool(gemini_key and gemini_key.strip()),
        "model": os.getenv("RAG_GENERATION_MODEL", "gemini-3.8-flash")
    }


# ── Web UI Endpoints ──

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/ai_ui.html", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/ui.html", methods=["GET", "HEAD"], include_in_schema=False)
def serve_ui():
    for fname in ["ai_ui.html", "ui.html"]:
        ui_path = PROJECT_ROOT / fname
        if ui_path.exists() and ui_path.stat().st_size > 0:
            return FileResponse(str(ui_path))
    raise HTTPException(status_code=404, detail="UI HTML file not found.")


# ── Chat Endpoints ──

@app.post("/chat/documents", response_model=ChatContextResponse)
def prepare_chat_documents(payload: ChatContextRequest):
    return chat_call(_chat.prepare, payload)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    return chat_call(_chat.ask, payload)


@app.delete("/chat/sessions/{session_id}")
def delete_chat_session(session_id: UUID):
    return {"deleted_chunks": chat_call(_chat.delete_session, session_id)}


@app.delete("/chat/documents/{document_id}")
def delete_chat_document(document_id: str, session_id: UUID):
    return {"deleted_chunks": chat_call(_chat.delete_document, session_id, document_id)}


def get_rag() -> RAGPipeline:
    """Lazy-load RAG pipeline so server starts instantly without importing heavy models."""
    global _rag
    if _rag is None:
        from query import RAGPipeline
        _rag = RAGPipeline()
    return _rag


# ── Request Models ──

class JDCreateRequest(BaseModel):
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    jd_text: str = Field(..., description="Full job description text")


class RankRequest(BaseModel):
    resume_text: str = Field(..., description="Candidate resume text")
    resume_id: str = Field(..., description="Candidate identifier")
    top_n: Optional[int] = Field(5, ge=0, description="Number of results to display (0 or None for all); AI reviews at most the top 25")
    preferred_location: Optional[str] = Field(None, description="Preferred city: +5 same city, +4 within 100 km, +3 within 200 km, +2 within 300 km, otherwise 0 (straight-line)")
    preferred_arrangement: Optional[str] = Field(None, description="Preferred work arrangement (Remote, Hybrid, On-site)")


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., description="Candidate resume text")
    resume_id: str = Field(..., description="Candidate identifier")
    jd_id: str = Field(..., description="Target job description identifier")


class ReportRequest(BaseModel):
    analysis: dict
    resume_name: str
    jd_title: str


class ConfigUpdateRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    generation_model: Optional[str] = None
    rerank: Optional[bool] = None


# ── Phase 2 Endpoint ──

@app.post("/query")
def query_rag(query_text: str):
    """General-purpose RAG endpoint reserved for Phase 2."""
    if not query_text.strip():
        raise HTTPException(status_code=422, detail="Please enter a question.")
    try:
        answer = get_rag().run(query_text.strip())
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Library search is unavailable. Check your provider settings and retry.") from exc
    return {
        "answer": answer
    }


# ── Phase 1 Placement Analyzer Endpoints ──

@app.post("/jds", summary="Ingest a new Job Description into store")
def add_job_description(payload: JDCreateRequest):
    """Adds a JD to the store, embeds it once, and returns the record."""
    import jd_index
    result = jd_index.add_jd(
        title=payload.title,
        company=payload.company,
        location=payload.location,
        jd_text=payload.jd_text
    )
    if isinstance(result, ErrorResponse):
        raise HTTPException(status_code=422 if result.error_code == "EMPTY_JD" else 409, detail=result.model_dump())
    return result.model_dump()


@app.get("/jds", summary="List all stored Job Descriptions with full details")
def list_jds():
    import jd_index
    results = []
    for rec in jd_index.get_jd_records():
        results.append({
            "jd_id": rec.jd_id,
            "title": rec.title,
            "company": rec.company,
            "location": rec.location,
            "snippet": rec.snippet,
            "full_text": rec.full_text,
            "words": len(rec.full_text.split())
        })
    return {"jds": results, "count": len(results), "cap": None}


@app.get("/jds/ids")
def get_jd_ids():
    import jd_index
    return {"ids": jd_index.get_jd_ids()}


@app.delete("/jds/{jd_id}")
def delete_job_description(jd_id: str):
    import jd_index
    jd_index.delete_jd(jd_id)
    return {"jd_id": jd_id, "deleted": True}


@app.get("/jds/count", summary="Get stored JD count and cap")
def get_jds_count():
    """Returns the stored JD count; cap is null because storage has no count cap."""
    import jd_index
    return {
        "count": jd_index.count(),
        "cap": None
    }


@app.delete("/jds", summary="Reset/Clear stored Job Descriptions")
def clear_jds():
    """Clears all stored JDs to allow fresh testing."""
    import jd_index
    jd_index.clear_store_for_testing()
    return {
        "message": "All stored job descriptions cleared successfully.",
        "count": jd_index.count(),
        "cap": None
    }


@app.post("/rank", summary="Tier 1: Rank stored JDs against candidate resume")
def rank_jds(payload: RankRequest):
    """Embedding shortlist followed by optional Mercury role-fit reranking."""
    pref_loc = payload.preferred_location.strip() if payload.preferred_location and payload.preferred_location.strip() else None
    pref_arr = payload.preferred_arrangement.strip() if payload.preferred_arrangement and payload.preferred_arrangement.strip() else None
    from ranking import rank_jds as rank_documents
    data = rank_documents(
        resume_text=payload.resume_text,
        resume_id=payload.resume_id,
        top_n=payload.top_n,
        preferred_location=pref_loc,
        preferred_arrangement=pref_arr
    ).model_dump()
    data["rankings"] = data.get("results", [])
    return data


@app.post("/analyze", summary="Tier 2: Detailed gap analysis for chosen JD")
def analyze_candidate_jd(payload: AnalyzeRequest):
    """Executes single Gemini call + deterministic taxonomy grounding and scoring."""
    result = get_analyzer().analyze(
        resume_text=payload.resume_text,
        resume_id=payload.resume_id,
        jd_id=payload.jd_id
    )
    if "error_code" in result:
        raise HTTPException(status_code=404, detail=result)
    return result


@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        from document_parsing import extract_document, validate_doc_text
        content = await file.read()
        if len(content) > 5 * 1024 * 1024:
            raise ValueError("The file exceeds the 5 MB upload limit.")
        filename = file.filename or "uploaded_file"
        text, pages = extract_document(content, filename)
        status, warn = validate_doc_text(text)
        return {
            "name": filename,
            "text": text,
            "pages": pages,
            "words": len(text.split()),
            "status": status,
            "warning": warn
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/load-fixtures/pdfs")
def load_pdf_fixtures():
    import jd_index
    from document_parsing import extract_document, validate_doc_text
    resume_paths = sorted((PROJECT_ROOT / "sample_pdfs/resumes").glob("*.pdf"))
    jd_paths = sorted((PROJECT_ROOT / "sample_pdfs/job_descriptions").glob("*.pdf"))

    if not resume_paths or not jd_paths:
        raise HTTPException(status_code=404, detail="Sample PDFs directory not found.")

    jd_index.clear_store_for_testing()
    resumes = []
    for path in resume_paths[:10]:
        raw_text, pages = extract_document(path.read_bytes(), path.name)
        status, warn = validate_doc_text(raw_text)
        name = path.stem.replace("John_Doe_", "").replace("_", " ")
        resumes.append({
            "id": f"res_{uuid4().hex[:8]}",
            "name": name,
            "text": raw_text,
            "pages": pages,
            "words": len(raw_text.split()),
            "status": status,
            "warning": warn
        })

    jds = []
    for path in jd_paths[:10]:
        raw_text, pages = extract_document(path.read_bytes(), path.name)
        status, warn = validate_doc_text(raw_text)
        title = path.stem.replace("JD_", "").replace("_", " ")
        rec = jd_index.add_jd(
            title=title,
            company="Global Tech Partner",
            location="Remote / Hybrid",
            jd_text=raw_text
        )
        backend_id = rec.jd_id if not isinstance(rec, ErrorResponse) else None
        jds.append({
            "id": backend_id or f"jd_{uuid4().hex[:8]}",
            "title": title,
            "company": "Global Tech Partner",
            "location": "Remote / Hybrid",
            "text": raw_text,
            "pages": pages,
            "words": len(raw_text.split()),
            "status": status,
            "warning": warn,
            "backend_id": backend_id
        })

    return {"resumes": resumes, "jds": jds, "count": len(jds)}


@app.post("/api/load-fixtures/sample")
@app.post("/api/load-fixtures/json", include_in_schema=False)
def load_json_fixtures():
    import json
    import jd_index
    from document_parsing import validate_doc_text
    fixture_path = PROJECT_ROOT / "tests/fixtures/sample_data.json"
    if not fixture_path.exists():
        raise HTTPException(status_code=404, detail="sample_data.json not found.")

    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample_jobs = data.get("jds", [])
    if len(sample_jobs) != 5:
        raise HTTPException(status_code=503, detail="The sample dataset must contain all five jobs.")
    # Add samples without removing the user's jobs. The existing add_jd method
    # reuses identical jobs, so loading the samples again keeps the same IDs.
    resumes = []
    for r in data.get("resumes", []):
        text = r["text"]
        status, warn = validate_doc_text(text)
        resumes.append({
            "id": r["id"],
            "name": r.get("name") or r["id"].removeprefix("res_").replace("_", " ").title(),
            "text": text,
            "pages": [{"text": text, "page": None}],
            "words": len(text.split()),
            "status": status,
            "warning": warn
        })

    jds = []
    for j in sample_jobs:
        text = j["text"]
        status, warn = validate_doc_text(text)
        try:
            rec = jd_index.add_jd(
                title=j.get("title", "Role"),
                company=j.get("company", "Company"),
                location=j.get("location", "Remote"),
                jd_text=text
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Could not save all five sample jobs. Check your Gemini settings and retry; jobs already saved will be reused.") from exc
        if isinstance(rec, ErrorResponse):
            raise HTTPException(status_code=409, detail=rec.message)
        backend_id = rec.jd_id
        jds.append({
            "id": backend_id,
            "title": j.get("title", "Role"),
            "company": j.get("company", "Company"),
            "location": j.get("location", "Remote"),
            "text": text,
            "pages": [{"text": text, "page": None}],
            "words": len(text.split()),
            "status": status,
            "warning": warn,
            "backend_id": backend_id
        })

    saved_ids = set(jd_index.get_jd_ids())
    if not all(job["backend_id"] in saved_ids for job in jds):
        raise HTTPException(status_code=503, detail="Not all five sample jobs were saved. Please retry.")
    return {"resumes": resumes, "jds": jds, "count": len(jds), "stored_count": len(saved_ids)}


@app.post("/api/load-fixtures/india")
def load_india_fixtures():
    """Add 100 synthetic India jobs/resumes without replacing existing data."""
    import json
    import jd_index
    from document_parsing import validate_doc_text
    path = PROJECT_ROOT / "datasets/india_demo_100.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="India demo dataset not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data.get("jds", [])) != 100 or len(data.get("resumes", [])) != 100:
        raise HTTPException(status_code=503, detail="The India dataset must contain 100 jobs and 100 resumes.")
    try:
        records = jd_index.add_jds_batch(data["jds"])
    except Exception as exc:
        raise HTTPException(status_code=503, detail="India dataset could not finish loading. Check Gemini settings and retry; previously saved jobs will be reused.") from exc
    resumes, jobs = [], []
    for item in data["resumes"]:
        status, warning = validate_doc_text(item["text"])
        resumes.append({**item, "pages": [{"text": item["text"], "page": None}],
                        "words": len(item["text"].split()), "status": status, "warning": warning})
    for record in records:
        status, warning = validate_doc_text(record.full_text)
        jobs.append({"id": record.jd_id, "backend_id": record.jd_id, "title": record.title,
                     "company": record.company, "location": record.location, "text": record.full_text,
                     "pages": [{"text": record.full_text, "page": None}], "words": len(record.full_text.split()),
                     "status": status, "warning": warning, "synthetic": True})
    saved_ids = set(jd_index.get_jd_ids())
    if not all(j["backend_id"] in saved_ids for j in jobs):
        raise HTTPException(status_code=503, detail="Some India dataset jobs were not saved. Please retry.")
    return {"resumes": resumes, "jds": jobs, "count": len(jobs), "stored_count": len(saved_ids), "synthetic": True}


@app.get("/api/loaded-demo-resumes")
def get_loaded_demo_resumes():
    """Expose demo resumes when their complete job dataset is already saved.

    This read does not embed or insert anything, and never exposes user uploads.
    It lets a second browser recover the demo profiles loaded by another browser.
    """
    import json
    import jd_index
    saved = {(r.title, r.company, r.location, r.full_text.strip())
             for r in jd_index.get_jd_records()}
    datasets = []
    for name, relative_path, expected in [
        ("sample", "tests/fixtures/sample_data.json", 5),
        ("india", "datasets/india_demo_100.json", 100),
    ]:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        jobs, resumes = data.get("jds", []), data.get("resumes", [])
        if len(jobs) != expected or len(resumes) != expected:
            continue
        if all((j["title"], j["company"], j["location"], j["text"].strip()) in saved for j in jobs):
            datasets.append({"id": name, "resumes": resumes})
    return {"datasets": datasets}


@app.post("/api/report/markdown")
def generate_report_markdown(payload: ReportRequest):
    from report_generator import generate_markdown_report
    report = generate_markdown_report(payload.analysis, payload.resume_name, payload.jd_title)
    return {"markdown": report}


@app.get("/api/config")
def get_system_config():
    import jd_index
    gemini_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GEMINI_FALLBACK_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    return {
        "gemini_configured": bool(gemini_key and gemini_key.strip()),
        "groq_configured": bool(groq_key and groq_key.strip()),
        "openrouter_configured": bool(openrouter_key.strip()),
        "gemini_masked": f"••••••••{gemini_key[-4:]}" if len(gemini_key) > 4 else ("Set" if gemini_key else "Not Set"),
        "groq_masked": f"••••••••{groq_key[-4:]}" if len(groq_key) > 4 else ("Set" if groq_key else "Not Set"),
        "openrouter_masked": f"••••••••{openrouter_key[-4:]}" if len(openrouter_key) > 4 else ("Set" if openrouter_key else "Not Set"),
        "decision_model": os.getenv("OPENROUTER_DECISION_MODEL", "inception/mercury-decide:free"),
        "decision_enabled": os.getenv("RANK_DECISION_ENABLED", "true").lower() == "true",
        "generation_model": os.getenv("RAG_GENERATION_MODEL", "gemini-3.8-flash"),
        "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001"),
        "rerank": os.getenv("RAG_RERANK", "false").lower() == "true",
        "jd_count": jd_index.count(),
        "jd_cap": None
    }


@app.post("/api/config")
def update_system_config(payload: ConfigUpdateRequest):
    global analyzer, _chat, _rag
    import os
    env_file = PROJECT_ROOT / ".env"
    env_lines = []
    if env_file.exists():
        env_lines = env_file.read_text(encoding="utf-8").splitlines()

    env_dict = {}
    for line in env_lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env_dict[k.strip()] = v.strip()

    if payload.gemini_api_key is not None:
        key = payload.gemini_api_key.strip()
        os.environ["GEMINI_API_KEY"] = key
        env_dict["GEMINI_API_KEY"] = key
    if payload.groq_api_key is not None:
        key = payload.groq_api_key.strip()
        os.environ["GROQ_API_KEY"] = key
        env_dict["GROQ_API_KEY"] = key
    if payload.openrouter_api_key is not None:
        key = payload.openrouter_api_key.strip()
        os.environ["OPENROUTER_API_KEY"] = key
        env_dict["OPENROUTER_API_KEY"] = key
    if payload.generation_model:
        os.environ["RAG_GENERATION_MODEL"] = payload.generation_model
        env_dict["RAG_GENERATION_MODEL"] = payload.generation_model
    if payload.rerank is not None:
        val = "true" if payload.rerank else "false"
        os.environ["RAG_RERANK"] = val
        env_dict["RAG_RERANK"] = val
    if "API_BASE_URL" not in env_dict:
        env_dict["API_BASE_URL"] = "http://127.0.0.1:8010"

    new_content = "\n".join(f"{k}={v}" for k, v in env_dict.items()) + "\n"
    env_file.write_text(new_content, encoding="utf-8")

    analyzer = None
    _chat = ChatService()
    _rag = None
    # Jobs and resume ranking also hold provider clients; key edits must take
    # effect everywhere without discarding the persisted job vectors.
    import embedding_manager
    import jd_index
    import ranking
    embedding_manager.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    embedding_manager._DEFAULT_EMBEDDER = None
    jd_index._embedder = None
    ranking._embedder = None
    ranking._embed_resume.cache_clear()
    from decision_ranking import clear_cache
    clear_cache()

    return {"status": "ok", "message": "Configuration updated successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
