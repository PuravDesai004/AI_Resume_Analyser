from __future__ import annotations

from uuid import UUID
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from schemas import ErrorResponse
from chat_schemas import ChatContextRequest, ChatContextResponse, ChatRequest, ChatResponse
from chat_service import ChatService, ChatServiceError

app = FastAPI(title="AI Placement Analyzer & RAG API")

from typing import Optional

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


@app.get("/health")
def health():
    return {"status": "ok"}


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
    top_n: Optional[int] = Field(5, description="Number of top JDs to return (0 or None for all)")
    preferred_location: Optional[str] = Field(None, description="Preferred location filter/boost")
    preferred_arrangement: Optional[str] = Field(None, description="Preferred work arrangement (Remote, Hybrid, On-site)")


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., description="Candidate resume text")
    resume_id: str = Field(..., description="Candidate identifier")
    jd_id: str = Field(..., description="Target job description identifier")


# ── Phase 2 Endpoint (Preserved) ──

@app.post("/query")
def query_rag(query_text: str):
    """General-purpose RAG endpoint reserved for Phase 2."""
    answer = get_rag().run(query_text)
    return {
        "answer": answer
    }


# ── Phase 1 Placement Analyzer Endpoints ──

@app.post("/jds", summary="Ingest a new Job Description into store (max 15)")
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
    """Returns current number of stored JDs and the system cap (15)."""
    import jd_index
    return {
        "count": jd_index.count(),
        "cap": 15
    }


@app.delete("/jds", summary="Reset/Clear stored Job Descriptions")
def clear_jds():
    """Clears all stored JDs to allow fresh testing."""
    import jd_index
    jd_index.clear_store_for_testing()
    return {
        "message": "All stored job descriptions cleared successfully.",
        "count": jd_index.count(),
        "cap": 15
    }


@app.post("/rank", summary="Tier 1: Rank stored JDs against candidate resume")
def rank_jds(payload: RankRequest):
    """Calculates cosine similarity of resume against stored JDs with zero Gemini calls."""
    pref_loc = payload.preferred_location.strip() if payload.preferred_location and payload.preferred_location.strip() else None
    pref_arr = payload.preferred_arrangement.strip() if payload.preferred_arrangement and payload.preferred_arrangement.strip() else None
    # Ranking only needs document vectors; ESCO is required for detailed analysis.
    from ranking import rank_jds as rank_documents
    return rank_documents(
        resume_text=payload.resume_text,
        resume_id=payload.resume_id,
        top_n=payload.top_n,
        preferred_location=pref_loc,
        preferred_arrangement=pref_arr
    ).model_dump()


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
