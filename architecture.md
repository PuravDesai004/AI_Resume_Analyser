# Resume × JD Analysis Pipeline — Full Architecture Document

> **Purpose of this document:** Complete reference for the entire project — every file, its responsibilities, its boundaries, its interactions, and how it fits into the overall pipeline. Written so an external reviewer can understand the full system without reading the code.

---

## 1. Project Overview

### What This Project Does

This project takes two text inputs — a **parsed resume** and a **parsed job description (JD)** — and produces a structured, human-readable analysis showing how well the candidate's skills match the job requirements. The output includes matched skills, missing skills (split by essential vs. preferred), extra skills the candidate has, a numerical match score, a narrative summary, and actionable recommendations.

### Core Design Principles

1. **Single Gemini Call:** The entire pipeline makes exactly **one** AI call (Google Gemini) — at the very end, for generating the human-readable summary and recommendations. All skill extraction, matching, and comparison is done **programmatically** with no AI involvement.

2. **ESCO Grounding:** Every skill mention is verified against the [ESCO taxonomy](https://esco.ec.europa.eu/) (European Skills, Competences, Qualifications, and Occupations) — a standardized dataset of ~13,890 skills. This prevents hallucination by ensuring the pipeline only works with verified, standardized skill names.

3. **Deterministic Core:** Stages 1–4 (extraction, grounding, comparison, sufficiency check) are fully deterministic and produce the same output for the same input every time. Only Stage 5 (generation) involves an AI model.

4. **Two-Phase Roadmap:**
   - **Phase 1 (current):** Resume × JD skill gap analysis pipeline
   - **Phase 2 (future):** Resume Q&A chatbot — users can ask conversational questions about their resume, powered by the existing RAG infrastructure

### Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10–3.12 | Runtime |
| Google Gemini (`google-genai`) | Single AI call for analysis generation (model: `gemini-3.6-flash`) |
| ESCO taxonomy CSV | Standardized skills reference dataset (~13,890 skills) |
| `rapidfuzz` | Fuzzy string matching for skill grounding |
| `pydantic` | Data validation and schema enforcement |
| FastAPI + Uvicorn | HTTP API server |
| `python-dotenv` | Environment variable management (`.env` file for `GEMINI_API_KEY`) |

---

## 2. The Five-Stage Pipeline

```
Input: resume_text + jd_text
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 1: EXTRACTION (programmatic) │  skill_extraction.py
│  Parse text → detect sections →     │
│  generate n-gram phrases            │
│  AI: ❌ None                        │
└──────────────┬──────────────────────┘
               │ ExtractionResult
               ▼
┌─────────────────────────────────────┐
│  Stage 2: GROUNDING (ESCO lookup)   │  skill_grounding.py + esco_loader.py
│  Match n-grams against ESCO →       │
│  exact / fuzzy / discard            │
│  AI: ❌ None                        │
└──────────────┬──────────────────────┘
               │ GroundingResult
               ▼
┌─────────────────────────────────────┐
│  Stage 3: COMPARISON (deterministic)│  skill_comparison.py
│  Set operations on grounded skills  │
│  → matched / missing / extra        │
│  AI: ❌ None                        │
└──────────────┬──────────────────────┘
               │ ComparisonResult
               ▼
┌─────────────────────────────────────┐
│  Stage 4: SUFFICIENCY CHECK (gate)  │  sufficiency_check.py
│  Enough verified skills to proceed? │
│  AI: ❌ None                        │
└──────────┬───────────┬──────────────┘
           │ PASS      │ FAIL
           ▼           ▼
┌──────────────────┐  ┌──────────────────────┐
│ Stage 5: GENERATE│  │ Return "insufficient │
│ Build prompt →   │  │ data" response       │
│ ONE Gemini call  │  │ (no AI call made)    │
│ AI: ✅ Yes (1x)  │  └──────────────────────┘
└────────┬─────────┘
         │
         ▼
   AnalysisOutput (JSON)
```

---

## 3. File-by-File Reference

### Legend

- **Does:** What the file is responsible for
- **Does NOT:** Explicit boundaries — what this file intentionally avoids
- **Purpose:** Why it exists as a separate file
- **Calls Gemini:** Whether this file makes any AI API calls
- **Imports from:** Which project files it depends on
- **Imported by:** Which project files use it

---

## 3A. New Files (Phase 1 — Analysis Pipeline)

---

### `schemas.py`

**Does:**
- Defines all Pydantic `BaseModel` classes used for data validation across the pipeline
- Defines 7 models: `RawSkill`, `GroundedSkill`, `ExtractionResult`, `GroundingResult`, `ComparisonResult`, `SufficiencyResult`, `AnalysisOutput`
- Enforces type constraints using `Literal` types (e.g., `match_type` must be one of `"exact"`, `"close"`, `"unverified"`)
- Provides `.model_dump()` serialization for JSON output
- Acts as the single source of truth for the data contract between pipeline stages

**Does NOT:**
- Contain any logic, processing, or data transformation
- Make any API calls
- Import any project files — it has zero internal dependencies
- Define any functions beyond the Pydantic model classes

**Purpose:** Ensures every inter-stage handoff is typed, validated, and documented. If the shape of data changes, it changes in one place. Prevents drift between what one stage outputs and the next stage expects.

**Calls Gemini:** No

**Imports from:** Nothing (only `pydantic`)

**Imported by:** Every other new file (`skill_extraction.py`, `skill_grounding.py`, `skill_comparison.py`, `sufficiency_check.py`, `analysis_prompt_builder.py`, `analysis_pipeline.py`)

**Key models:**

| Model | Stage | Fields |
|---|---|---|
| `RawSkill` | Step 1 output | `skill_name`, `context`, `priority` |
| `GroundedSkill` | Step 2 output | `original_name`, `standardized_name`, `esco_id`, `match_type`, `match_score`, `context`, `priority` |
| `ExtractionResult` | Step 1 container | `resume_skills: list[RawSkill]`, `jd_skills: list[RawSkill]` |
| `GroundingResult` | Step 2 container | `resume_skills: list[GroundedSkill]`, `jd_skills: list[GroundedSkill]` |
| `ComparisonResult` | Step 3 output | `matched_skills`, `missing_essential`, `missing_preferred`, `extra_skills`, `unverified_resume`, `unverified_jd` |
| `SufficiencyResult` | Step 4 output | `passed`, `reason`, `resume_grounded_count`, `jd_grounded_count` |
| `AnalysisOutput` | Final output | `summary`, `match_score`, `matched_skills`, `missing_essential`, `missing_preferred`, `extra_skills`, `unverified_skills`, `recommendations`, `generation_ran`, `sufficiency_detail` |

---

### `esco_loader.py`

**Does:**
- Reads the ESCO skills taxonomy CSV file (`data/skills_en.csv`) at startup
- Parses the `preferredLabel` and `altLabels` columns from the CSV
- Builds an in-memory lookup dictionary mapping every normalized label (lowercased, stripped) to its ESCO concept URI and preferred label name
- Builds a flat list of all normalized labels for fuzzy matching
- Provides an `ESCOIndex` class with `exact_match()` and `fuzzy_match()` methods
- Uses `rapidfuzz.process.extractOne` for fuzzy matching with a configurable threshold

**Does NOT:**
- Make any API calls — the ESCO data is loaded from a local CSV file
- Download the ESCO dataset — the user must place `skills_en.csv` in `data/` manually
- Modify or write to the CSV file
- Perform any skill extraction or comparison — it only provides the lookup structure
- Handle any resume or JD text directly

**Purpose:** The ESCO taxonomy is the foundation of the grounding system. By loading it once at startup and providing fast lookups, it avoids repeated file I/O and enables the grounding step to check thousands of n-grams efficiently.

**Calls Gemini:** No

**Imports from:** Nothing (only `csv`, `rapidfuzz`)

**Imported by:** `skill_grounding.py`, `analysis_pipeline.py`

**Key class: `ESCOIndex`**

| Method | Input | Output | Description |
|---|---|---|---|
| `__init__(csv_path)` | Path to CSV | — | Loads CSV, builds lookup structures |
| `exact_match(phrase)` | Normalized string | `dict \| None` | O(1) dict lookup |
| `fuzzy_match(phrase, threshold=85)` | String, int | `dict \| None` | rapidfuzz best match above threshold |
| `get_all_labels()` | — | `list[str]` | All normalized labels |

---

### `skill_extraction.py`

**Does:**
- Parses resume text to detect section headers (e.g., "Work Experience", "Skills", "Projects") using regex patterns
- Parses JD text to detect priority sections (e.g., "Requirements" → essential, "Nice to Have" → preferred) using keyword matching
- Generates n-gram phrases (1-word through 4-word combinations) from the text
- Classifies JD requirements as `"essential"`, `"preferred"`, or `"unspecified"` based on section headings and line-level keyword signals
- Tags each extracted phrase with its source section (for resume) or priority level (for JD)
- Returns `ExtractionResult` containing `list[RawSkill]` for both resume and JD

**Does NOT:**
- Make any AI/Gemini calls — extraction is fully programmatic
- Determine whether an n-gram is actually a skill — that's the grounding step's job
- Match against ESCO — it only generates candidate phrases
- Filter or deduplicate across resume and JD — it processes them independently
- Handle PDF parsing or file I/O — it receives pre-parsed text strings

**Purpose:** Separates the text-parsing concern from the skill-matching concern. This file turns free-form text into a list of candidate phrases with metadata (section, priority). The grounding step then filters to only real skills.

**Calls Gemini:** No

**Imports from:** `schemas.py` (`RawSkill`, `ExtractionResult`)

**Imported by:** `analysis_pipeline.py`

**Key functions:**

| Function | Input | Output |
|---|---|---|
| `detect_resume_sections(text)` | Raw resume text | `list[tuple[str, str]]` — (section_name, content) |
| `detect_jd_sections(text)` | Raw JD text | `list[tuple[str, str]]` — (priority, content) |
| `detect_jd_priority(text)` | A line or section header | `"essential" \| "preferred" \| "unspecified"` |
| `generate_ngrams(text)` | Any text | `list[str]` — 1-to-4 word phrases |
| `extract_resume_skills(resume_text)` | Resume text | `list[RawSkill]` |
| `extract_jd_skills(jd_text)` | JD text | `list[RawSkill]` |
| `extract_skills(resume_text, jd_text)` | Both texts | `ExtractionResult` |

**Key constants:**

| Constant | Value | Purpose |
|---|---|---|
| `RESUME_SECTION_PATTERNS` | List of (regex, label) tuples | Identifies resume section headers |
| `ESSENTIAL_SIGNALS` | List of keyword strings | Signals that a JD requirement is essential |
| `PREFERRED_SIGNALS` | List of keyword strings | Signals that a JD requirement is preferred |
| `STOP_WORDS` | Set of common filler words | Skipped when generating n-grams |
| `MAX_NGRAM` | `4` | Maximum n-gram length (captures "Amazon Web Services") |

---

### `skill_grounding.py`

**Does:**
- Takes each raw n-gram phrase from extraction and attempts to match it against the ESCO taxonomy
- Performs a three-step matching cascade per phrase: exact match → fuzzy match (threshold=85) → discard
- Filters out non-matching n-grams (the majority of n-grams won't match — this is expected)
- Deduplicates matched skills by `standardized_name` (keeps the first/strongest context)
- Preserves the `context` (section name) and `priority` (essential/preferred) from the input `RawSkill` into the output `GroundedSkill`
- Returns `GroundingResult` with clean, ESCO-verified skill lists for both resume and JD

**Does NOT:**
- Make any AI/Gemini calls
- Parse or read text — it only processes pre-extracted `RawSkill` objects
- Decide whether a skill is essential or preferred — that was decided in extraction
- Compare resume skills against JD skills — that's the comparison step's job
- Keep unmatched n-grams — unlike v1, non-matching phrases are silently discarded because n-gram extraction produces mostly noise

**Purpose:** This is the "truth filter" of the pipeline. It transforms noisy n-gram candidates into verified, standardized skills. Everything downstream operates on ESCO-verified data only.

**Calls Gemini:** No

**Imports from:** `schemas.py` (`RawSkill`, `GroundedSkill`, `ExtractionResult`, `GroundingResult`), `esco_loader.py` (`ESCOIndex`)

**Imported by:** `analysis_pipeline.py`

**Key functions:**

| Function | Input | Output |
|---|---|---|
| `ground_skill(raw, esco_index)` | `RawSkill`, `ESCOIndex` | `GroundedSkill \| None` |
| `ground_skills(extraction, esco_index)` | `ExtractionResult`, `ESCOIndex` | `GroundingResult` |

**Fuzzy match threshold:** `85` (out of 100). Higher than typical because n-gram extraction is noisy — a higher threshold reduces false positives without missing legitimate variations like "React.js" → "React".

---

### `skill_comparison.py`

**Does:**
- Compares the two grounded skill lists (resume vs. JD) using set operations on `standardized_name`
- Categorizes every skill into exactly one of: matched, missing-essential, missing-preferred, extra, or unverified
- Splits missing skills into two groups based on `priority` from the JD extraction
- Returns a `ComparisonResult` with all five categories populated

**Does NOT:**
- Make any AI/Gemini calls
- Assign scores, weights, or rankings to skills — it only categorizes
- Generate any text, summaries, or recommendations
- Access the ESCO taxonomy — it operates on already-grounded data
- Modify the input data — it's read-only

**Purpose:** Isolated, deterministic comparison logic. Given the same two grounded lists, it always produces the same `ComparisonResult`. This makes it independently testable with crafted test data, no API keys needed.

**Calls Gemini:** No

**Imports from:** `schemas.py` (`GroundingResult`, `ComparisonResult`, `GroundedSkill`)

**Imported by:** `analysis_pipeline.py`

**Key function:**

| Function | Input | Output |
|---|---|---|
| `compare_skills(grounding)` | `GroundingResult` | `ComparisonResult` |

**Matching key:** `standardized_name.lower()` — safe because grounding already standardized to ESCO preferred labels.

---

### `sufficiency_check.py`

**Does:**
- Inspects the grounding and comparison results to decide if there's enough verified data for generation
- Applies a simple, tunable rule: at least `MIN_GROUNDED_SKILLS` (default: 3) verified skills from BOTH the resume and the JD
- If the check fails, builds a complete `AnalysisOutput` with `generation_ran=False` and a descriptive reason — the caller still gets partial data
- If the check passes, returns a simple pass signal

**Does NOT:**
- Make any AI/Gemini calls
- Generate summaries, recommendations, or scores
- Modify any data — it only reads and decides
- Consider unverified skills in the count — only exact and close matches count

**Purpose:** The hallucination prevention gate. If there isn't enough grounded data, the pipeline stops before ever calling Gemini. This is the core credibility feature — the system never generates analysis from insufficient evidence. Kept as a small, separate file so it can be tested directly with edge cases.

**Calls Gemini:** No

**Imports from:** `schemas.py` (`ComparisonResult`, `GroundingResult`, `SufficiencyResult`, `AnalysisOutput`)

**Imported by:** `analysis_pipeline.py`

**Key functions:**

| Function | Input | Output |
|---|---|---|
| `check_sufficiency(comparison, grounding, min_grounded=3)` | `ComparisonResult`, `GroundingResult`, int | `SufficiencyResult` |
| `build_insufficient_response(sufficiency, comparison)` | `SufficiencyResult`, `ComparisonResult` | `AnalysisOutput` |

**Key constant:** `MIN_GROUNDED_SKILLS = 3`

---

### `analysis_prompt_builder.py`

**Does:**
- Builds the system prompt and user prompt for the single Gemini generation call
- Formats the `ComparisonResult` into five labeled sections: matched skills, missing essential, missing preferred, extra skills, unverified skills
- Includes a pre-computed statistics block (counts of each category) to help the model calibrate the match score
- Defines the `GENERATION_RESPONSE_SCHEMA` — the JSON schema enforced on Gemini's output via `response_schema`
- Defines the `ANALYSIS_SYSTEM_PROMPT` — the full system prompt with 6 rules governing how the model should analyze the data
- Provides `get_generation_config()` which returns the complete `GenerateContentConfig` (system instruction, response MIME type, response schema, temperature)

**Does NOT:**
- Call Gemini — it only builds the prompt and config; the actual API call happens in `analysis_pipeline.py`
- Access the ESCO taxonomy or any raw text
- Modify the `ComparisonResult` data — it only reads and formats
- Produce the final `AnalysisOutput` — it produces the prompt; the orchestrator assembles the final output

**Purpose:** Separates prompt engineering from pipeline orchestration. The prompt, schema, and system instructions can be iterated independently without touching the pipeline logic. This is the domain-specific sibling to the existing `generation.py` (which builds prompts for general RAG Q&A).

**Calls Gemini:** No (builds the prompt — does not send it)

**Imports from:** `schemas.py` (`ComparisonResult`), `google.genai` (for `GenerateContentConfig` type)

**Imported by:** `analysis_pipeline.py`

**Key functions:**

| Function | Input | Output |
|---|---|---|
| `build_analysis_prompt(comparison)` | `ComparisonResult` | `str` (prompt text) |
| `get_generation_config()` | — | `genai.types.GenerateContentConfig` |

**Key constants:**

| Constant | Purpose |
|---|---|
| `ANALYSIS_SYSTEM_PROMPT` | Full system prompt: 6 rules governing model behavior, scoring logic, recommendation format, unverified skill handling |
| `GENERATION_RESPONSE_SCHEMA` | JSON schema with 3 fields: `summary` (string), `match_score` (number), `recommendations` (array of strings) |

**Gemini config details:**

| Setting | Value | Rationale |
|---|---|---|
| `temperature` | `0.3` | Needs coherent prose, but must stay grounded |
| `response_mime_type` | `"application/json"` | Forces JSON output |
| `response_schema` | `GENERATION_RESPONSE_SCHEMA` | Enforces exact output shape |
| `system_instruction` | `ANALYSIS_SYSTEM_PROMPT` | Separates role instructions from data |

---

### `analysis_pipeline.py`

**Does:**
- Orchestrates the entire five-stage pipeline in order
- Initializes shared resources once at startup: ESCO index, Gemini client
- Calls each stage in sequence: extract → ground → compare → sufficiency check → generate
- Makes **exactly one** Gemini API call (in Stage 5, only if sufficiency check passes)
- Merges AI-generated fields (`summary`, `match_score`, `recommendations`) with deterministic comparison data into a complete `AnalysisOutput`
- Returns `.model_dump()` — a plain dict that FastAPI can serialize directly
- Includes `if __name__ == "__main__":` block with test resume/JD data for standalone testing

**Does NOT:**
- Implement any stage logic itself — it delegates to the appropriate module
- Make more than one Gemini call per pipeline run
- Let Gemini see or modify the deterministic skill lists — the AI only generates summary, score, and recommendations
- Handle HTTP concerns — that's `server.py`'s job
- Parse PDFs or files — it receives pre-parsed text strings

**Purpose:** The single entry point that the next developer in the team pipeline calls. Whether invoked via the API endpoint or directly in code, the interface is the same: `pipeline.run(resume_text, jd_text) → dict`. This is the domain-specific sibling to the existing `query.py` (which orchestrates general RAG Q&A).

**Calls Gemini:** Yes — exactly **1 call** per pipeline run (Stage 5 only, only if sufficiency passes)

**Imports from:** All new files: `schemas.py`, `esco_loader.py`, `skill_extraction.py`, `skill_grounding.py`, `skill_comparison.py`, `sufficiency_check.py`, `analysis_prompt_builder.py`. Also: `google.genai`, `dotenv`, `json`.

**Imported by:** `server.py`

**Key class: `AnalysisPipeline`**

| Method | Input | Output |
|---|---|---|
| `__init__()` | — | Loads ESCO index, initializes Gemini client |
| `run(resume_text, jd_text)` | Two strings | `dict` (serialized `AnalysisOutput`) |

---

## 3B. Existing Files (Phase 1: Modified)

---

### `server.py`

**Current state:** Serves a single `POST /query` endpoint for general RAG Q&A.

**Phase 1 modification:** Add a new `POST /analyze` endpoint alongside the existing `/query` endpoint. The existing code is **not modified** — the new endpoint is purely additive.

**New additions:**
- Import `AnalysisPipeline` from `analysis_pipeline.py`
- Import `BaseModel` from `pydantic`
- Define `AnalyzeRequest(BaseModel)` with `resume_text: str` and `jd_text: str`
- Instantiate `analyzer = AnalysisPipeline()` at module level
- Add `@app.post("/analyze")` endpoint

**Endpoints after modification:**

| Endpoint | Purpose | Phase |
|---|---|---|
| `POST /query?query_text=...` | General RAG Q&A (existing) | Phase 2 |
| `POST /analyze` (JSON body) | Resume × JD analysis (new) | Phase 1 |

---

### `requirements.txt`

**Current state:** Lists 12 dependencies for the general RAG pipeline.

**Phase 1 modification:** Add one line: `rapidfuzz`. All existing dependencies are kept.

---

## 3C. Existing Files (Untouched — Reserved for Phase 2)

These files belong to the general-purpose RAG pipeline. They are **not modified, not deleted, and not used** by the Phase 1 analysis pipeline. They will be reused in Phase 2 when the Resume Q&A Chatbot feature is built.

---

### `chunking.py`

**What it does:** Loads PDF and TXT files from a directory, splits them into token-measured chunks using `RecursiveCharacterTextSplitter` with `tiktoken` (`cl100k_base` encoding). Chunk size: 300 tokens, overlap: 50 tokens. Supports markdown table extraction and image text extraction from PDFs via `PyMuPDFLoader`.

**What it does NOT do (in Phase 1):** Nothing — this file is not imported or called by any Phase 1 code.

**Phase 2 role:** Will chunk the user's resume into sections for retrieval-based Q&A.

**Imports:** `langchain_community.document_loaders`, `langchain_text_splitters`, `tiktoken`

---

### `embedding_manager.py`

**What it does:** Wraps the Google GenAI embedding API (`gemini-embedding-001`). Provides `embed_chunk(text)` for single texts and `embed_chunks_batch(chunk_list)` for batch embedding. Uses `GEMINI_API_KEY` from `.env`.

**What it does NOT do (in Phase 1):** Nothing — not imported or called.

**Phase 2 role:** Will embed resume chunks for vector similarity search.

**Imports:** `google.genai`, `dotenv`

---

### `chroma_db.py`

**What it does:** Manages a persistent ChromaDB vector database at `./my_local_db`. Provides `add_chunks_to_collection(chunks, embeddings)` to store document chunks with metadata. Creates a `"my_documents"` collection on import.

**What it does NOT do (in Phase 1):** Nothing — not imported or called.

**Phase 2 role:** Will store per-session resume chunks for retrieval.

**Imports:** `chromadb`

---

### `retrieval.py`

**What it does:** Implements a hybrid retrieval system: vector search via ChromaDB, keyword search via BM25 (`rank_bm25`), Reciprocal Rank Fusion to merge results, and neural reranking via a CrossEncoder (`sentence-transformers`). Contains 6 functions: `query_collection`, `get_all_chunks_from_db`, `build_bm25_index`, `bm25_search`, `merge_rrf`, `rerank`.

**What it does NOT do (in Phase 1):** Nothing — not imported or called.

**Phase 2 role:** Full hybrid search for resume Q&A (dense + BM25 + RRF + reranking).

**Imports:** `chroma_db.collection`, `rank_bm25`

---

### `generation.py`

**What it does:** Builds grounded Q&A prompts from retrieved chunks. Formats each chunk with source, page number, and content. Enforces structured JSON output (`answer` + `citations`), strict anti-hallucination rules, and a fallback message ("I don't know based on the provided context.").

**What it does NOT do (in Phase 1):** Nothing — not imported or called. The Phase 1 equivalent is `analysis_prompt_builder.py`.

**Phase 2 role:** Will build grounded Q&A prompts for resume-related questions.

**Imports:** None (pure string formatting)

---

### `query.py`

**What it does:** The general RAG pipeline orchestrator. `RAGPipeline` class loads resources at startup (embedder, cross-encoder, BM25 index, Gemini client) and executes the full retrieval → reranking → generation pipeline via `run(query_text)`. Uses `gemini-3.6-flash`.

**What it does NOT do (in Phase 1):** Nothing — not imported or called. The Phase 1 equivalent is `analysis_pipeline.py`.

**Phase 2 role:** Will orchestrate resume Q&A queries.

**Imports:** `google.genai`, `generation`, `embedding_manager`, `sentence_transformers`, `retrieval`

---

### `main.py`

**What it does:** Document ingestion entry point. Loads documents from `./Pdfs`, chunks them, embeds them via `EmbeddingManager`, and stores them in ChromaDB via `add_chunks_to_collection`. Validates `GEMINI_API_KEY` on startup.

**What it does NOT do (in Phase 1):** Nothing — not imported or called.

**Phase 2 role:** Will ingest and index the user's resume.

**Imports:** `chunking`, `embedding_manager`, `chroma_db`, `dotenv`

---

### `print_db.py`

**What it does:** Debug utility. Connects to ChromaDB at `./my_local_db`, fetches all records (excluding embeddings), and prints ID, document snippet, and metadata for each.

**What it does NOT do (in Phase 1):** Nothing — not imported or called.

**Phase 2 role:** Debug/inspect indexed resume data.

**Imports:** `chromadb`

---

## 4. Data Flow — Complete Example

**Resume input:**
```
PROFESSIONAL EXPERIENCE
Senior Software Engineer at TechCorp (2020-2024)
- Built microservices using Python, FastAPI, and PostgreSQL
- Deployed on AWS using EC2, S3, and Lambda

SKILLS
Python, JavaScript, React.js, Docker, Git
```

**JD input:**
```
Requirements:
- Python (required)
- React (must have)
- AWS (required)

Nice to Have:
- Kubernetes
- GraphQL
```

### Stage-by-stage:

| Stage | What Happens | Output (simplified) |
|---|---|---|
| **1. Extraction** | Resume: detect "Professional Experience" and "Skills" sections. Generate n-grams: `["Python", "FastAPI", "PostgreSQL", "AWS", "EC2", "React.js", "Docker", "Git", ...]` (hundreds of n-grams). JD: detect "Requirements" (essential) and "Nice to Have" (preferred) sections. Generate n-grams with priority tags. | `ExtractionResult` with ~100+ resume candidates and ~30+ JD candidates |
| **2. Grounding** | Each n-gram checked against ESCO. `"Python"` → exact match. `"React.js"` → fuzzy match to "React" (score 92). `"Built microservices"` → no match → discarded. Most n-grams get discarded. | `GroundingResult`: Resume: 7 verified skills. JD: 5 verified skills. |
| **3. Comparison** | Set operations: `{Python, React} ∩ both` → matched. `{AWS} ∈ JD essential, ∉ resume` → missing essential. `{Kubernetes, GraphQL} ∈ JD preferred, ∉ resume` → missing preferred. `{FastAPI, PostgreSQL, Docker, Git} ∈ resume, ∉ JD` → extra. | `ComparisonResult` with 5 categories |
| **4. Sufficiency** | Resume verified: 7 ≥ 3 ✓. JD verified: 5 ≥ 3 ✓. → PASS | `SufficiencyResult(passed=True)` |
| **5. Generation** | Prompt built with all 5 sections + stats. Gemini called once. Returns summary, score (72), and 5 recommendations. | `AnalysisOutput` with all fields |

---

## 5. Inter-File Dependency Map

```
schemas.py ◄──────────────────────────────────────────────────┐
    │                                                          │
    ├── skill_extraction.py ──────────────────┐                │
    │                                         │                │
    ├── esco_loader.py ──► skill_grounding.py ┤                │
    │                                         │                │
    ├── skill_comparison.py ──────────────────┤                │
    │                                         │                │
    ├── sufficiency_check.py ─────────────────┤                │
    │                                         │                │
    └── analysis_prompt_builder.py ───────────┤                │
                                              │                │
                                              ▼                │
                                    analysis_pipeline.py ──────┘
                                              │
                                              ▼
                                          server.py
```

**No circular dependencies.** Every arrow points downward. `schemas.py` is at the root, `server.py` is at the leaf.

---

## 6. Gemini Integration Details

### Where Gemini is Called

**Exactly one location:** `analysis_pipeline.py` → `AnalysisPipeline.run()` → Stage 5

```python
response = self.client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,               # Built by analysis_prompt_builder.py
    config=gen_config,             # Built by analysis_prompt_builder.py
)
```

### What Gemini Receives

- **System instruction:** The `ANALYSIS_SYSTEM_PROMPT` (6 rules about grounding, scoring, formatting)
- **User content:** A formatted prompt containing all 5 comparison sections + statistics
- **Response schema:** `GENERATION_RESPONSE_SCHEMA` enforcing `{summary, match_score, recommendations}`

### What Gemini Returns

Only 3 fields — the AI never touches the verified skill lists:

| Field | Type | Description |
|---|---|---|
| `summary` | `string` | 3-5 sentence narrative assessment |
| `match_score` | `number` | 0-100 alignment score |
| `recommendations` | `string[]` | 3-7 actionable recommendations |

### What Gemini Does NOT Touch

All deterministic fields are populated by `analysis_pipeline.py` from the `ComparisonResult`:
- `matched_skills`
- `missing_essential`
- `missing_preferred`
- `extra_skills`
- `unverified_skills`
- `generation_ran`
- `sufficiency_detail`

---

## 7. ESCO Taxonomy Details

### What is ESCO

The European Skills, Competences, Qualifications, and Occupations taxonomy — a standardized dataset of ~13,890 skills and knowledge concepts maintained by the European Commission.

### How We Use It

- **Downloaded as CSV** from the [ESCO portal](https://esco.ec.europa.eu/en/use-esco/download)
- **Stored locally** at `data/skills_en.csv`
- **Loaded once** at startup by `esco_loader.py`
- **Key columns used:** `conceptUri` (unique ID), `preferredLabel` (canonical name), `altLabels` (synonyms separated by `\n`)
- **Both preferred labels AND alt labels** are indexed for matching

### Matching Strategy

```
Input phrase (e.g., "React.js")
       │
       ▼
 Exact match against all labels?
       │
  YES ─┤─► match_type="exact", score=100
       │
  NO ──┤
       ▼
 Fuzzy match (rapidfuzz, threshold=85)?
       │
  YES ─┤─► match_type="close", score=85-99
       │
  NO ──┤─► Discard (not a recognized skill)
```

---

## 8. Folder Structure — Complete

```
rag project/
│
├── .env                         (GEMINI_API_KEY=...)
├── .gitignore
├── requirements.txt             (MODIFIED — added rapidfuzz)
│
├── data/
│   └── skills_en.csv            (ESCO dataset — user-provided)
│
├── ── Phase 1: Analysis Pipeline ──────────────────
├── schemas.py                   (Pydantic models — 7 schemas)
├── esco_loader.py               (ESCO CSV loader + ESCOIndex)
├── skill_extraction.py          (Programmatic n-gram extraction)
├── skill_grounding.py           (ESCO matching — exact/fuzzy)
├── skill_comparison.py          (Set-based comparison)
├── sufficiency_check.py         (Min-skills gate)
├── analysis_prompt_builder.py   (Gemini prompt + schema + config)
├── analysis_pipeline.py         (Orchestrator — 1 Gemini call)
│
├── ── Phase 2: General RAG (untouched) ────────────
├── chunking.py                  (PDF/TXT loading + token chunking)
├── embedding_manager.py         (Gemini embeddings wrapper)
├── chroma_db.py                 (ChromaDB persistent storage)
├── retrieval.py                 (Hybrid: vector + BM25 + RRF + reranking)
├── generation.py                (General RAG prompt builder)
├── query.py                     (General RAG orchestrator)
├── main.py                      (Document ingestion script)
├── print_db.py                  (ChromaDB debug utility)
│
└── server.py                    (FastAPI — /query + /analyze endpoints)
```

---

## 9. Phase 2 Roadmap: Resume Q&A Chatbot

> After Phase 1 is complete and tested, Phase 2 adds a conversational Q&A feature where users can ask questions about their resume.

**How existing files will be reused:**

| Step | Files Used | What Happens |
|---|---|---|
| 1. Resume ingestion | `chunking.py` → `embedding_manager.py` → `chroma_db.py` | Resume is chunked, embedded, and stored in ChromaDB |
| 2. User asks a question | `retrieval.py` (vector + BM25 + reranking) | Relevant resume chunks are retrieved |
| 3. Answer generation | `generation.py` → `query.py` → Gemini | Grounded answer with citations is produced |
| 4. API endpoint | `server.py` → `POST /chat` | New endpoint alongside `/analyze` and `/query` |

**Phase 2 is intentionally deferred** — the analysis pipeline must be complete and validated first.
