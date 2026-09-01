"""Job Copilot API - match, cover letters, and application tracking."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from match import analyze, MatchReport
from cover_letter import write_cover_letter
from tracker import log_application, update_status, list_applications

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
    try:
        return analyze(posting)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

# ---- serve the frontend ----
from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("frontend/index.html")
