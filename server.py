from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from query import RAGPipeline
from analysis_pipeline import AnalysisPipeline
import jd_index
from schemas import ErrorResponse

app = FastAPI(title="AI Placement Analyzer & RAG API")

# Initialize services
rag = RAGPipeline()
analyzer = AnalysisPipeline()


# ── Request Models ──

class JDCreateRequest(BaseModel):
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    jd_text: str = Field(..., description="Full job description text")


class RankRequest(BaseModel):
    resume_text: str = Field(..., description="Candidate resume text")
    resume_id: str = Field(..., description="Candidate identifier")


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., description="Candidate resume text")
    resume_id: str = Field(..., description="Candidate identifier")
    jd_id: str = Field(..., description="Target job description identifier")


# ── Phase 2 Endpoint (Preserved) ──

@app.post("/query")
def query_rag(query_text: str):
    """General-purpose RAG endpoint reserved for Phase 2."""
    answer = rag.run(query_text)
    return {
        "answer": answer
    }


# ── Phase 1 Placement Analyzer Endpoints ──

@app.post("/jds", summary="Ingest a new Job Description into store (max 15)")
def add_job_description(payload: JDCreateRequest):
    """Adds a JD to the store, embeds it once, and returns the record."""
    result = analyzer.ingest_jd(
        title=payload.title,
        company=payload.company,
        location=payload.location,
        jd_text=payload.jd_text
    )
    if isinstance(result, ErrorResponse):
        raise HTTPException(status_code=409, detail=result.model_dump())
    return result.model_dump()


@app.get("/jds/count", summary="Get stored JD count and cap")
def get_jds_count():
    """Returns current number of stored JDs and the system cap (15)."""
    return {
        "count": jd_index.count(),
        "cap": 15
    }


@app.delete("/jds", summary="Reset/Clear stored Job Descriptions")
def clear_jds():
    """Clears all stored JDs to allow fresh testing."""
    jd_index.clear_store_for_testing()
    return {
        "message": "All stored job descriptions cleared successfully.",
        "count": jd_index.count(),
        "cap": 15
    }


@app.post("/rank", summary="Tier 1: Rank stored JDs against candidate resume")
def rank_jds(payload: RankRequest):
    """Calculates cosine similarity of resume against stored JDs with zero Gemini calls."""
    return analyzer.rank(
        resume_text=payload.resume_text,
        resume_id=payload.resume_id
    )


@app.post("/analyze", summary="Tier 2: Detailed gap analysis for chosen JD")
def analyze_candidate_jd(payload: AnalyzeRequest):
    """Executes single Gemini call + deterministic taxonomy grounding and scoring."""
    result = analyzer.analyze(
        resume_text=payload.resume_text,
        resume_id=payload.resume_id,
        jd_id=payload.jd_id
    )
    if "error_code" in result:
        raise HTTPException(status_code=404, detail=result)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)