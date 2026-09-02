"""Job Copilot API - match, cover letters, and application tracking."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from match import analyze, MatchReport, validate_posting
from cover_letter import write_cover_letter
from tracker import log_application, update_status, list_applications
from gaps import keyword_rows
from insights_agent import find_gap_themes
from picks import top_picks, MIN_PICK_SCORE, MAX_PICKS

app = FastAPI(title="Job Copilot", version="0.4.0")

class MatchRequest(BaseModel):
    posting: str

class AnalyzeRequest(BaseModel):
    company: str
    role: str
    posting: str

class AnalyzeResponse(BaseModel):
    application_id: int
    report: MatchReport

class CoverLetterRequest(BaseModel):
    company: str
    posting: str

class StatusUpdate(BaseModel):
    status: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/match", response_model=MatchReport)
def match(request: MatchRequest):
    return _safe_analyze(request.posting)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_and_log(request: AnalyzeRequest):
    report = _safe_analyze(request.posting)
    app_id = log_application(request.company, request.role, report)
    return AnalyzeResponse(application_id=app_id, report=report)

@app.post("/cover-letter")
def cover_letter(request: CoverLetterRequest):
    return {"cover_letter": write_cover_letter(request.posting, request.company)}

@app.get("/insights")
def insights():
    """Read-only skill-gap themes across every scored posting. Never writes."""
    rows = keyword_rows()
    try:
        themes = find_gap_themes(rows)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Insights agent unavailable: {e}")
    return {
        "postings_analyzed": len(rows),
        "themes": [t.model_dump() for t in themes],
    }

@app.get("/top-picks")
def picks(limit: int = MAX_PICKS):
    """The best still-open postings, with any letter already on disk attached.

    Pure local read - no model call, so this view works with ollama down.
    """
    limit = max(1, min(limit, 20))
    return {
        "min_score": MIN_PICK_SCORE,
        "picks": top_picks(limit),
    }

@app.get("/applications")
def applications():
    return list_applications()

@app.patch("/applications/{app_id}")
def set_status(app_id: int, update: StatusUpdate):
    update_status(app_id, update.status)
    return {"id": app_id, "status": update.status}

def _safe_analyze(posting: str) -> MatchReport:
    if len(posting.strip()) < 50:
        raise HTTPException(status_code=400, detail="Posting too short to analyze")
    check = validate_posting(posting)
    if not check.is_job_posting:
        raise HTTPException(status_code=400, detail=f"That doesn't look like a job posting: {check.reason}")
    try:
        return analyze(posting)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

# ---- serve the frontend ----
from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("frontend/index.html")
