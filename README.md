# Job Copilot 💼

AI-powered job search automation that runs **100% locally — zero API costs**.
Paste a job posting, get an honest match score against your real resume,
a tailored cover letter, and a tracked application pipeline.

Built with Python, Ollama (llama3.1:8b), ChromaDB, FastAPI, and Pydantic.

![Job Copilot](screenshots/app.png)

## Why

Job searching means reading dozens of postings and guessing "am I a fit?"
Job Copilot answers that with evidence: it retrieves the most relevant parts
of your resume for each posting (RAG), scores the match against a rubric,
tells you exactly which keywords you're missing, and logs everything to a
local tracker.

## How it works

    job posting --> embed (nomic-embed-text)
                        |
                        v
                ChromaDB similarity search --> top resume chunks
                        |
                        v
          llama3.1:8b + scoring rubric (temperature=0)
                        |
                        v
          JSON validated against Pydantic schema (with retries)
                        |
                        v
       match report --> SQLite tracker --> FastAPI endpoints

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
# open http://127.0.0.1:8000       <- web UI
# open http://127.0.0.1:8000/docs  <- API docs
```

## Run the evals

```bash
python run_evals.py       # v1 matcher: scoring, consistency, input validation
python scout_evals.py     # Scout agent: fit judgment, keyword bait, seniority traps
python insights_evals.py  # Insights agent: real themes found, invented ones dropped
python ask_evals.py       # Ask: answers from the data, refusals for everything else
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

- [x] LLM input validation ("is this actually a job posting?")
- [ ] Expanded eval suite (license/degree detection, not-a-posting case)
- [ ] Docker deployment

## v2: Agent Edition

Job Copilot now hunts on its own. A scheduled pipeline fetches postings from permitted
job feeds, and a crew of local agents (CrewAI + Ollama) decides what deserves attention:

    RemoteOK feed
        |
        v
    cheap filter (title keywords, word-boundary matched)
        |
        v
    Scout agent -- judges genuine fit: rejects keyword bait,
        |          non-engineering roles, and senior-level traps
        v
    matcher (RAG + rubric scoring, from v1)
        |
        v
    65+ score? -> Letter agent drafts a grounded cover letter to drafts/
        |
        v
    tracker (status: "scouted") -> visible in the UI's Scouted view

Run once: `python scout_run.py` - or on a schedule: `python scheduler.py` (visible,
killable loop; nothing runs hidden in the background).

GUARDRAIL BY DESIGN: the system never applies, sends, or touches any account.
Agents prepare; the human decides.

### v2 lessons learned

- Feed tags lie: a Handyperson posting arrived tagged `golang`. Filtering moved to
  role titles only.
- Substring matching is a trap: the keyword "ai" matched "mAIntenance" and "mAIl
  carrier" until word-boundary regex fixed it.
- Keywords are not judgment: the v1 matcher scored a "Course Writer, UX/AI" posting
  85% because it mentions AI constantly. The Scout agent rejects it with the correct
  reason - which is exactly why the agent layer exists.
- Installing CrewAI silently upgraded chromadb and broke the existing vector store
  (Rust panic on open). Fixed by rebuilding the index - a classic dependency-collision
  lesson.

### Scout evals

`python scout_evals.py` - 3 cases: junior-fit approval, keyword-bait rejection,
seniority-trap rejection. Current: 3/3.

## Insights: what your gaps add up to

One `missing_keywords` row tells you nothing. Forty of them are a pattern. The Insights
agent reads every scored posting and clusters the gaps into themes - "application
security", say - each one carrying the postings that back it.

It is scoped to skill gaps on purpose. It says nothing about outcomes: with a handful
of rows and no rejections logged there is no funnel to analyze, and a model asked about
one anyway will invent it.

Grounded the same way Ask is: every theme is checked against real row ids before it is
returned, and a theme the rows don't support is dropped rather than softened. A single
posting is not a pattern, so one-off gaps are dropped too.

`python insights_evals.py` - 7 cases, 5 of them model-free guard tests. Current: 7/7.

## Top Picks: what to act on next

The pipeline lists everything. Top Picks answers the narrower question - what should I
do today? - with the best still-open postings and whatever cover letter the scout has
already drafted, attached and ready to read.

A pick is deliberately narrow: not already applied to or rejected, and scoring at least
65 - the same floor the scout uses before it bothers drafting a letter. Calling a 40%
match a "top pick" would make the view flattery rather than triage.

No model call and no network: it is a pure read over `tracker.db` and the `drafts/`
folder, so it opens instantly and still works with ollama down.

## Ask: questions in plain English

The Insights view answers one question the code picked. **Ask** answers the one you
type - "what skill keeps costing me points?", "how many have I applied to?" - against
your own tracker, on the same local model. Still zero API cost.

An open question box is riskier than a fixed report, because an open prompt invites
the model to answer from the internet's idea of job searching instead of from your
rows. Four things keep it honest:

1. **A scope gate first.** One cheap yes/no call decides whether the question can be
   answered from the columns the tracker actually has. Culture, pay, company size,
   the future - refused before the answering call ever runs.
2. **A hard-coded refusal list.** Salary, visas, recruiter contacts and "what are my
   chances" never reach the model at all: there is no column behind them.
3. **Python does the arithmetic.** Counts, averages and status totals are computed in
   code and handed to the model as facts, so it cannot invent a funnel it doesn't have.
4. **Citations are verified.** Every answer names the posting ids behind it, checked
   against real rows. An answer citing postings that don't exist is dropped whole.

    your question --> length + scope guards (no model call)
                            |
                            v
                    scope gate: answerable from these columns? --> no: refuse
                            |
                            v
              FACTS (counted in Python) + PIPELINE rows --> llama3.1:8b
                            |
                            v
                  citations checked against real ids --> answer, or refusal

Read-only and stateless: it never writes to the tracker, and nothing about what you
asked is stored. `python ask_agent.py "what am I missing most?"` from the CLI, or the
Ask tab in the UI.

`python ask_evals.py` - 13 cases, 9 of them model-free guard tests. Current: 13/13.

### Ask lessons learned

- Told to answer *and* to police itself in one call, the model does the fun half: it
  rated a company's "engineering culture" from a row that holds a match score and a
  date. Splitting the scope decision into its own call fixed it - one job per call.
- "Be blunt" produced answers like `0` and `penetration testing methodologies`. Blunt
  is not the same as terse; the prompt now asks for complete sentences.
- The scope gate refused "what am I missing most often?" - a question the tracker
  plainly covers. No column stores a *frequency*, so the gate ruled the whole class of
  aggregates unanswerable, never mind that `_facts()` counts them in Python before the
  model sees anything. A guard that only understands columns will refuse real questions.
- That bug survived a green eval suite. The guard cases assert on `out_of_scope()`, the
  regex list - but the refusal came from `in_scope()`, the model call, which had no
  coverage at all. Test the layer that made the decision, not the one next to it.
- A bare `Not Found` in the UI turned out to be a *stale server*: uvicorn had been
  started without `--reload` before the route existed, so the page was newer than the
  API. `/health` now reports a version the page checks on load, because "your server is
  older than your frontend" is not something a 404 will ever tell you.
