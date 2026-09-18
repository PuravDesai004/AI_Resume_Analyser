import json
from fastapi import FastAPI, Request, HTTPException
from query import RAGPipeline
from analysis_pipeline import AnalysisPipeline

app = FastAPI(title="RAG & Resume-JD Gap Analysis API")

# Initialize services
rag = RAGPipeline()
analyzer = AnalysisPipeline()


@app.post("/query")
def query_rag(query_text: str):
    answer = rag.run(query_text)
    return {
        "answer": answer
    }


@app.post(
    "/analyze",
    summary="Analyze Resume against Job Description",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "resume_text": {
                                "type": "string",
                                "description": "Full resume text. Raw newlines and paragraphs are supported."
                            },
                            "jd_text": {
                                "type": "string",
                                "description": "Full job description text. Raw newlines and paragraphs are supported."
                            }
                        },
                        "required": ["resume_text", "jd_text"]
                    },
                    "example": {
                        "resume_text": "Senior Backend Developer with 5 years experience in Python, PostgreSQL, and Docker.\nBuilt microservices with FastAPI and configured CI/CD with GitHub Actions.",
                        "jd_text": "Looking for a Backend Engineer.\nRequired: Python, PostgreSQL, AWS (EC2/S3).\nPreferred: Docker, Kubernetes, CI/CD experience."
                    }
                }
            }
        }
    }
)
async def analyze_resume_jd(request: Request):
    raw_body = await request.body()
    try:
        # strict=False allows unescaped control characters (newlines/tabs) from pasted text
        data = json.loads(raw_body.decode("utf-8"), strict=False)
    except Exception as err:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid JSON payload: {err}. If pasting multi-line text, ensure valid quotes."
        )

    resume_text = data.get("resume_text", "").strip()
    jd_text = data.get("jd_text", "").strip()

    if not resume_text:
        raise HTTPException(status_code=422, detail="Field 'resume_text' cannot be empty.")
    if not jd_text:
        raise HTTPException(status_code=422, detail="Field 'jd_text' cannot be empty.")

    result = analyzer.run(resume_text, jd_text)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)