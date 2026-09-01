"""Match a job posting against your profile -> validated structured report."""
import sys
from pathlib import Path
import chromadb
import ollama
from pydantic import BaseModel, Field, ValidationError

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"
TOP_K = 5
MAX_RETRIES = 2

class MatchReport(BaseModel):
    match_score: int = Field(ge=0, le=100)
    matching_skills: list[str]
    missing_keywords: list[str]
    projects_to_emphasize: list[str]
    one_line_verdict: str

def embed(text: str) -> list[float]:
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]

def retrieve_profile(posting: str) -> str:
    collection = chromadb.PersistentClient(path="db").get_collection("profile")
    results = collection.query(query_embeddings=[embed(posting)], n_results=TOP_K)
    return "\n---\n".join(results["documents"][0])

def build_prompt(profile: str, posting: str) -> str:
    return f"""You are a strict, honest career coach. Compare the CANDIDATE PROFILE to the JOB POSTING.
Use this rubric for match_score:
- 80-100: candidate has the core required skills and relevant hands-on projects or experience
- 60-79: has most core skills, missing some secondary ones
- 40-59: has some overlapping skills but lacks key requirements
- 0-39: different field, or missing most core requirements
Do not penalize for years of experience if the posting welcomes early-career candidates.

CANDIDATE PROFILE:
{profile}

JOB POSTING:
{posting}

Respond with ONLY a JSON object with exactly these keys:
"match_score" (integer 0-100), "matching_skills" (list of strings),
"missing_keywords" (list of strings), "projects_to_emphasize" (list of strings),
"one_line_verdict" (string)."""

def analyze(posting: str) -> MatchReport:
    profile = retrieve_profile(posting)
    prompt = build_prompt(profile, posting)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        try:
            return MatchReport.model_validate_json(response["message"]["content"])
        except ValidationError as e:
            last_error = e
            print(f"⚠️  Attempt {attempt}: model output failed validation, retrying...")
    raise RuntimeError(f"Model never produced valid output:\n{last_error}")

def print_report(report: MatchReport, job_name: str):
    print(f"\n💼 {job_name}")
    print(f"📊 Match score: {report.match_score}%")
    print(f"✅ Matching skills: {', '.join(report.matching_skills)}")
    print(f"❌ Missing keywords: {', '.join(report.missing_keywords)}")
    print(f"⭐ Emphasize: {', '.join(report.projects_to_emphasize)}")
    print(f"💬 {report.one_line_verdict}")

if __name__ == "__main__":
    job_file = Path(sys.argv[1] if len(sys.argv) > 1 else "jobs/example-role.md")
    print_report(analyze(job_file.read_text()), job_file.stem)
