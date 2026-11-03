# AI Resume Analyser

A local FastAPI application for comparing a résumé with job descriptions, identifying skill gaps, and asking questions about a selected match.

The application ranks saved jobs using text similarity and optional AI-assisted role-fit analysis. A report combines ESCO skill matching with provider-generated analysis. The chat uses the selected résumé, job description, and report as context and returns source references. It is a decision-support demo; its scores are not hiring probabilities.

## Results

The bundled India demo contains 100 synthetic résumés and 100 synthetic job descriptions. Count them from the repository root without an API key:

```powershell
python -c "import json; d=json.load(open('datasets/india_demo_100.json', encoding='utf-8')); print('resumes:', len(d['resumes']), 'jobs:', len(d['jds']))"
```

The complete provider workflow was not rerun during the audit because it makes rate-limited API calls. The local `/health` endpoint returned HTTP 200 without an API key.

## How it works

The HTML interface talks to `server.py`. Analysis and ranking use the ESCO vocabulary, embeddings, and configured providers. `chat_service.py` prepares document-scoped context; the `rag_engine/` package performs retrieval and grounded generation. See [LOCAL_SETUP.md](LOCAL_SETUP.md) for the full flow and provider settings.

## Stack

Python 3.12, FastAPI, ChromaDB, ESCO, Gemini and Groq integrations, and an optional OpenRouter reranker. Streamlit is available for the older interface.

## Quickstart

From this folder, create a Python 3.12 environment and install the dependencies. On this computer the bundled Python 3.12 executable is available at the path below; another machine can substitute its own Python 3.12 executable.

```powershell
$Python312 = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $Python312 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010`. The health check is at `http://127.0.0.1:8010/health`. Starting the server does not by itself validate the remote analysis, embedding, or chat providers. Configure a provider in the existing `.env` file or in the app before using those features. Do not overwrite an existing `.env` file. Provider calls can incur charges or rate limits.

## Project structure

- `server.py` and `ai_ui.html`: API and browser interface.
- `analysis_pipeline.py`, `ranking.py`, and `skill_comparison.py`: analysis and ranking.
- `chat_service.py` and `rag_engine/`: document-scoped chat and retrieval.
- `datasets/` and `scripts/`: synthetic examples and their generator.
- `tests/`: regression and interface tests.

## Tests

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pytest -q
```

The full suite could not be collected on the audited Windows machine because application control blocked a `grpc` DLL imported by ChromaDB. A run excluding `tests/test_all.py` exercised the remaining suite; provider interactions used existing test doubles. UTF-8 mode avoids a Windows text-decoding failure in the synthetic dataset test.

## Data and limitations

The bundled example résumés and job descriptions are synthetic. `scripts/generate_india_demo.py` generates the India demo dataset. The ESCO skills vocabulary is third-party reference data, not synthetic candidate information. Do not enter real résumé or job-description content unless you intend to send it to the configured external provider.

AI-generated analysis can be incomplete or incorrect. The application does not establish hiring outcomes, and scores should be reviewed alongside the underlying evidence.

## Attribution and license

The connected RAG implementation is adapted from [shahmit107/RAG](https://github.com/shahmit107/RAG) at the commit recorded in [rag_engine/README.md](rag_engine/README.md). No license file is supplied for this repository; confirm the applicable rights and data terms before redistribution.
