"""Match a job posting against your profile -> structured report."""
import json
import sys
from pathlib import Path
import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"
TOP_K = 5

def embed(text: str) -> list[float]:
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]

def retrieve_profile(posting: str) -> str:
    collection = chromadb.PersistentClient(path="db").get_collection("profile")
    results = collection.query(query_embeddings=[embed(posting)], n_results=TOP_K)
    return "\n---\n".join(results["documents"][0])

def analyze(posting: str) -> dict:
    profile = retrieve_profile(posting)
    prompt = f"""You are a career coach. Compare the CANDIDATE PROFILE to the JOB POSTING.

CANDIDATE PROFILE:
{profile}

JOB POSTING:
{posting}

Respond with ONLY a JSON object with exactly these keys:
- "match_score": integer 0-100
- "matching_skills": list of skills the candidate has that the job wants
- "missing_keywords": list of important skills/terms in the posting the candidate lacks
- "projects_to_emphasize": list of the candidate's projects most relevant to this role
- "one_line_verdict": a single honest sentence"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return json.loads(response["message"]["content"])

def print_report(report: dict, job_name: str):
    print(f"\n💼 {job_name}")
    print(f"📊 Match score: {report.get('match_score')}%")
    print(f"✅ Matching skills: {', '.join(report.get('matching_skills', []))}")
    print(f"❌ Missing keywords: {', '.join(report.get('missing_keywords', []))}")
    print(f"⭐ Emphasize: {', '.join(report.get('projects_to_emphasize', []))}")
    print(f"💬 {report.get('one_line_verdict')}")

if __name__ == "__main__":
    job_file = Path(sys.argv[1] if len(sys.argv) > 1 else "jobs/example-role.md")
    report = analyze(job_file.read_text())
    print_report(report, job_file.stem)
