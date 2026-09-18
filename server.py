from fastapi import FastAPI
from pydantic import BaseModel
from query import RAGPipeline
from analysis_pipeline import AnalysisPipeline

app = FastAPI(title="RAG & Resume-JD Gap Analysis API")

# Initialize services
rag = RAGPipeline()
analyzer = AnalysisPipeline()


class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str


@app.post("/query")
def query_rag(query_text: str):
    answer = rag.run(query_text)
    return {
        "answer": answer
    }


@app.post("/analyze")
def analyze_resume_jd(request: AnalyzeRequest):
    result = analyzer.run(request.resume_text, request.jd_text)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)