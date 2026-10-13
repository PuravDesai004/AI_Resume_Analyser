# Local resume analyser with connected RAG chat

The analyser is cloned from https://github.com/PuravDesai004/AI_Resume_Analyser
(starting commit `944fe52`). Its new chat adapts the RAG engine from
https://github.com/shahmit107/RAG.git, under `backend/pipeline`.

## Install and run

Python 3.12 is the tested local runtime. From the repository directory on macOS or Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8010
```

Before starting the server, fill `GEMINI_API_KEY` in `.env`, or set it through
Settings after opening the app. The key stays in the ignored `.env` file.
`GROQ_API_KEY` is an optional chat fallback. `OPENROUTER_API_KEY` enables Mercury
role-fit reranking; set it in `.env` or the OpenRouter field in Settings. Model settings are in `.env.example`.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8010
```

Open http://127.0.0.1:8010. The backend serves the HTML interface; one server is
sufficient. The optional older Streamlit dashboard runs separately with
`streamlit run app.py --server.port 8510` from the activated virtual environment.

Embeddings, analysis and chat require configured remote AI providers. The tokenizer
may download its encoding cache on first use; enabling `RAG_RERANK=true` may download
the CrossEncoder model.

## App flow

1. Add your resume in **Resumes** and job descriptions in **Jobs**.
2. Open **Placement fit** and choose your resume on the setup screen.
3. Run placement fit to rank all saved jobs, with optional location/work preferences.
   Matching opens a separate results screen; the setup controls are hidden.
4. Click **Run gap analysis** on a ranked job to open its report inside Placement fit.
5. Click **Ask AI about this match** in the report. The assistant uses this exact
   resume, job and completed analysis, displayed as read-only context labels.
6. Use **Back to matching jobs** to explore another role, or **Back to setup**
   to change the resume/preferences. The ranking remains available until the resume changes.

Resume text, original page references and your selected resume are saved in this
browser's local storage and restored after refresh. Jobs persist in the backend.
Analysis and conversation state are per-page sessions; after refresh run the
analysis again before chatting. Changing the selected resume or opening another
match closes the prior chat and prevents late answers from appearing in the new
conversation. Reset removes the saved browser resumes, backend jobs and current
chat data. Local browser storage is specific to the URL and browser profile.

Result cards show **Role fit /100**, with Strong/Moderate/Weak in the same score area, as the primary score when reranking succeeds.
This is a normalized model rating, not percentage requirement coverage or a hiring
probability. On fallback, the primary score is explicitly labelled similarity.
Location and work-arrangement bonuses appear separately, alongside the resulting
ranking total in points; that total can exceed 100. The minimum-score filter uses
the displayed base score, without bonuses. The gap
report's skill coverage score uses 80% essential and 20% preferred requirement
coverage. Experience, education and responsibility percentages are AI estimates
shown separately; they do not enter that computed skill score.

The HTML interface includes provider settings, Markdown report downloads, Top 5 /
Top 10 / Top 25 / All rankings, location and work arrangement preferences, result filters
and sorting, two- or three-role comparisons, a demo dataset selector (the original five examples or 100 India jobs/resumes), and job-store
refresh. The resume selector is on the Placement fit setup screen. Resumes and Jobs
handle uploads and document management. Placement fit has separate setup, ranked
results and gap-report screens; returning from a gap report keeps
the ranked results. Chat includes a completed analysis
for the selected pair, preserves real PDF page references, and clears indexed
contexts when documents or conversations are removed. Provider failures are
displayed instead of simulated ranking or analysis results.

Chat is available only after a successful gap analysis for the selected pair.
Follow-up questions reuse the report and indexed evidence. Failed or mismatched
analyses cannot unlock chat, and chat does not silently run another analysis.
The four suggested questions stay available after every answer and are disabled
while a request is running. Project ideas, interview preparation and learning plans
may use general knowledge, clearly labelled as suggestions. Sources establish the
skills/role basis; proposed projects are never presented as completed resume work.
Learning timelines include assumptions and estimates, while candidate facts, job
requirements and computed scores remain tied to the selected documents.

Resume/JD questions are available in the AI Assistant. The sample button adds all five bundled jobs and sample resumes, keeps your loaded
resume and existing jobs, and reuses previously saved samples instead of duplicating
them. Success is shown only after verifying every sample job in the backend. Resetting
clears stored jobs and the current chat's indexed data after confirmation.

Settings save provider keys locally in `.env`. Empty key fields keep existing
keys; use the removal checkboxes to clear them. Model changes apply to chat and placement analysis, with the existing model
fallbacks retained. Chat reranking applies to the resume/JD chat pipeline.

API documentation: http://127.0.0.1:8010/docs

Health check: http://127.0.0.1:8010/health

If changing the backend port, also change `API_BASE_URL` in `.env` and restart the dashboard.

## Bounded role-fit reranking

Job storage has no artificial count cap; API `cap` / `jd_cap` fields are null.
Placement fit compares embeddings against all stored jobs and applies location/work
preferences. The **top 25** embedding matches form the fixed maximum Mercury pool,
regardless of how many results the user chooses to display. Top 5/10/25 shows that
many results after reranking; **All jobs** displays the reviewed shortlist first,
then the rest in original similarity order. Jobs outside the pool are explicitly
labelled “similarity · outside AI shortlist” and never receive invented role-fit
scores. A positive role-fit minimum filters to AI-reviewed jobs only.

[OpenRouter System One](https://openrouter.ai/docs/guides/community/typesafe-sdk)
receives at most 25 complete resume/job pairs per ranking run, divided into batches
of up to ten questions. The batch builder also enforces the 60 KB request budget;
no document is silently truncated. Results are cached by pair. No extra analysis
screen is required, and embeddings still choose the candidate pool.

Set `OPENROUTER_API_KEY` in the ignored `.env` or Settings. The default model is
`OPENROUTER_DECISION_MODEL=inception/mercury-decide:free`; provider access and free
rate limits still apply. Resume and shortlisted job text are sent to OpenRouter
when enabled. To retain only the original ranking, set `RANK_DECISION_ENABLED=false`
and restart the server.

The typed rubric is Weak / Moderate / Strong, based on evidenced skills, relevant
work and explicit minimum qualifications. It treats “A or B” as alternatives and
ignores personal attributes and location, which code handles separately. The
probability-weighted rubric index is normalized to 0–100 internally; results sort
by that model fit plus the preference bonus, with original boosted similarity
breaking ties. The model's confidence and fit are not hiring probabilities, and
this is a career guidance tool. **Highest similarity** in the existing Sort by
menu lets you inspect the original semantic ordering without another API call.

The combined Role fit score/category appears only for validated assessments used for reranking.
Missing keys, disabled reranking, provider errors, incomplete/malformed answers,
an individual document pair that exceeds the 60 KB request budget, or any confidence below
`OPENROUTER_DECISION_MIN_CONFIDENCE=0.5` preserve the whole baseline order. The
results explain the fallback; uncertain assessments keep the similarity display. Calls use a
3-second connection / 20-second read timeout and a 20-second failure cooldown.

Successful document-pair assessments are cached in memory for 30 minutes (128
entries). Resume/JD content, model, key and rubric changes invalidate reuse;
Settings edits clear the cache. Restarting also clears it. Gemini still handles
individual gap reports and report-scoped chat; Mercury changes job ordering only.

## Nearby location hierarchy

The rules are defined in `location_preferences.py`, with no location API calls or
additional keys. The `geonamescache==3.0.2` dependency bundles GeoNames city data
locally; the app uses its cities dataset with a minimum population of 15,000.
[Package documentation](https://pypi.org/project/geonamescache/) and
[GeoNames data attribution](https://www.geonames.org/about.html).

| Location relative to the preferred city | Bonus |
|---|---:|
| Same resolved city | +5 |
| Different city, up to 100 km | +4 |
| Over 100 km, up to 200 km | +3 |
| Over 200 km, up to 300 km | +2 |
| Over 300 km | 0 |

Cutoffs are inclusive and use unrounded straight-line distance between city
coordinates, not driving distance or a commute estimate. Every job is checked
against these fixed bands; the nearest available job never gets a bonus merely
because it is the nearest. Mercury fit stays unchanged: the final ranking value
is **base fit + location bonus + work-arrangement bonus**. The same preference
points also support the earlier embedding shortlist, without adding them twice
to the final value.

For a Mumbai preference, the bundled coordinates give Mumbai +5, Navi Mumbai
(14.6 km) +4, Pune (119.5 km) +3, Ratnagiri (235.8 km) +2, and Ahmedabad
(440.6 km) +0. Results show the distance, bonus and reason separately.

Names normalize capitalization, whitespace, accents and known GeoNames aliases,
e.g. Bombay resolves to Mumbai and Poona to Pune. Use `City, Country` or
`City, State, Country` to disambiguate. US state abbreviations are recognized.
Bare names resolve to a dominant city only when its population is at least ten
times that of the next candidate. Unknown/ambiguous cities, unsupported smaller
places, multi-city alternatives and remote-only locations receive no guessed
location bonus; the result explains why. Country-only preferences do not resolve
to arbitrary city coordinates.

## Use chat

1. Upload or paste a resume in **Resumes**, and optionally add job descriptions in **Jobs**.
2. Run placement fit, open a job’s gap report, then click **Ask AI about this match**.
3. Ask a question. The first question indexes the context; later questions reuse it.
4. Open Sources beneath an answer to inspect supporting text and real PDF page numbers.
5. Ask a follow-up or clear the conversation. Open a different job’s gap analysis to change scope.

The five sample roles use Indian cities to demonstrate the location hierarchy.
Choose **Mumbai** as your preferred location and **Any** work arrangement:
Python Backend → Mumbai (+5), React Frontend → Navi Mumbai (+4), AI/ML → Pune (+3),
DevOps → Ratnagiri (+2), and Full Stack → Ahmedabad (+0). Their qualifications
remain unchanged; Mercury fit and the location bonus are separate signals.

The **Jobs** tab has one **Load dataset** button with a dataset selector. The
**India dataset** contains 100 additional synthetic jobs and 100 resumes, using
the same JSON structure and concise style as the five originals. See
[datasets/README.md](datasets/README.md) and `datasets/india_demo_100.json`.
Loading keeps personal records, embeds new jobs in batches of 20, and reuses
existing records on repeat loads or after a partial provider failure. Resumes
persist in browser storage; the JSON also preserves a reproducible copy. PDF, DOCX and UTF-8 TXT
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

Run the complete review suite with fake AI calls and isolated test storage. It uses
the bundled ESCO CSV, requires no provider keys, and does not reset application data:

```bash
.venv/bin/python tests/run_review_checks.py
```

On Windows, use `.\.venv\Scripts\python.exe` instead of `.venv/bin/python`.

HTML integration checks (Node.js, no provider calls or application-data changes):

```sh
node tests/test_html_chat_context.cjs
node tests/test_html_chat_gaps.cjs
node tests/test_html_features.cjs
node tests/test_html_placement_flow.cjs
node tests/test_html_chat_suggestions.cjs
node tests/test_html_mercury.cjs
node tests/test_html_india_dataset.cjs
```

Chat tests use actual Chroma filtering with fake AI calls:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_chat*.py' -v
```

Live checks use fictional sample documents. This is a local single-user app;
production authentication and an OCR pipeline are outside this project setup.
