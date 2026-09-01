# Job Copilot 💼

AI-powered job search automation that runs **100% locally — zero API costs**.
Paste a job posting, get an honest match score against your real resume,
a tailored cover letter, and a tracked application pipeline.

Built with Python, Ollama (llama3.1:8b), ChromaDB, FastAPI, and Pydantic.

## Why

Job searching means reading dozens of postings and guessing "am I a fit?"
Job Copilot answers that with evidence: it retrieves the most relevant parts
of your resume for each posting (RAG), scores the match against a rubric,
tells you exactly which keywords you're missing, and logs everything to a
local tracker.

## How it works

## Features

- **RAG matching** — resume is chunked, embedded, and stored in ChromaDB;
  each posting retrieves only the most relevant profile chunks
- **Structured outputs** — the model must return JSON matching a Pydantic
  schema (score 0-100, skills, missing keywords); invalid output is retried,
  then fails loudly
- **Rubric-based scoring** — explicit score bands in the prompt to fight
  both flattery and over-harshness (both observed and fixed via evals)
- **Eval harness** — honesty, fit, and consistency test cases with expected
  score ranges; used to catch a real scoring bug during development
- **Cover letters** — generated only from experience that actually appears
  in the resume (grounded generation, no invented skills)
- **Application tracker** — SQLite log of every analyzed posting with score,
  status, and date
- **REST API** — FastAPI service with auto-generated docs at /docs

## Quickstart

Requires [Ollama](https://ollama.com) running locally.

```bash
git clone https://github.com/Yasminenaser1/job-copilot.git
cd job-copilot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 1. Put your resume in profile/resume.md, then index it
python ingest.py

# 2. Score a posting from the command line
python match.py jobs/example-role.md

# 3. Or run the API
uvicorn api:app --reload
# open http://127.0.0.1:8000/docs
```

## Run the evals

```bash
python run_evals.py
```

## Lessons learned

- Same input, different score: LLM scoring drifted between runs until
  temperature was pinned to 0 — and the eval suite is what made the drift
  visible and measurable.
- "Grade harshly" overcorrected: the strong-fit test case dropped to 42%.
  Replacing vibes with an explicit rubric fixed it.
- Confident garbage: the API happily scored placeholder text at 85% —
  length checks catch short garbage, not plausible garbage. Semantic input
  validation is the planned fix.

## Roadmap

- [ ] LLM input validation ("is this actually a job posting?")
- [ ] Expanded eval suite (license/degree detection, not-a-posting case)
- [ ] Docker deployment
