"""Job Copilot API - serves the matcher over HTTP."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from match import analyze, MatchReport   # reuse the logic you already built

app = FastAPI(title="Job Copilot", version="0.2.0")

# What a request must contain: just the posting text
class MatchRequest(BaseModel):
    posting: str

# Simple "is the server alive?" check - every real service has one
@app.get("/health")
def health():
    return {"status": "ok"}

# The main endpoint. response_model=MatchReport means FastAPI validates
# the output AND documents it automatically in /docs
@app.post("/match", response_model=MatchReport)
def match(request: MatchRequest):
    if len(request.posting.strip()) < 50:
        raise HTTPException(status_code=400, detail="Posting too short to analyze")
    try:
        return analyze(request.posting)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
