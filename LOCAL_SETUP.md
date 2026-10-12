# Local resume analyser with connected RAG chat

The analyser is cloned from https://github.com/PuravDesai004/AI_Resume_Analyser
(starting commit `944fe52`). Its new chat adapts the RAG engine from
https://github.com/shahmit107/RAG.git, under `backend/pipeline`.

## Environment

Dependencies, including the optional Groq fallback client, are installed in `.venv`
using Python 3.14.2 during development. Configure Gemini/Groq in the ignored `.env`;
keys are not part of source control.

For a fresh installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill `GEMINI_API_KEY` locally; optionally set `GROQ_API_KEY` for the existing fallback.
The example also documents model selection and optional reranking. The tokenizer
may download its encoding cache on first use; enabling `RAG_RERANK=true` may download
the CrossEncoder model. Embeddings and answers use remote providers.

## Run

Open two PowerShell terminals in this repository directory. Ports 8010 and 8510 keep
this app separate from the original RAG application's default ports.

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8010
```

Dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8510
```

Dashboard: http://127.0.0.1:8510

API documentation: http://127.0.0.1:8010/docs

Health check: http://127.0.0.1:8010/health

If changing the backend port, also change `API_BASE_URL` in `.env` and restart the dashboard.

## Use chat

1. Upload or paste a resume and optionally a job description in the first tab.
2. Open **6. Chat with Resume and JD** and choose the documents for the conversation.
3. Ask a question. The first question indexes the context; later questions reuse it.
4. Open Sources beneath an answer to inspect supporting text and real PDF page numbers.
5. Ask a follow-up, switch documents to change scope, or clear the selected conversation.

Sample PDFs can be loaded with the sidebar fixture button. PDF, DOCX and UTF-8 TXT
uploads are supported. Scanned PDFs require OCR before chat can use their text.

Re-uploading a document already loaded in the same list skips the duplicate by
text content. Updated content can still be added under the same filename. Upload
errors, size/cap warnings, and success messages remain visible through dashboard
refreshes until dismissed or replaced by the next upload batch.

Chat and vector ranking work without the ESCO dataset. To generate
new placement analyses, download the English Skills CSV from
https://esco.ec.europa.eu/en/use-esco/download and place it at `data/skills_en.csv`.
Then run the existing analysis workflow; a matching cached result is included in
the selected resume/JD conversation. The backend loads the analysis pipeline lazily.

## Verification

Run the 32 review checks with fake AI calls and isolated test storage. This does not
require provider keys or installing ESCO, and does not reset application data:

```powershell
.\.venv\Scripts\python.exe tests/run_review_checks.py
```

Chat tests use actual Chroma filtering with fake AI calls:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_chat*.py' -v
```

On this Windows machine the restricted execution environment can block test threads
and temporary files. Run these checks in a normal local terminal if that occurs.

The live check uses fictional documents and deletes its session afterward. Full
production authentication and an OCR pipeline are outside this local integration.
