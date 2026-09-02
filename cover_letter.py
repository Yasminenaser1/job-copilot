"""Generate a cover letter grounded in YOUR real profile for a given posting."""
import sys
from pathlib import Path
import ollama
from match import retrieve_profile, CHAT_MODEL

def write_cover_letter(posting: str, company: str = "the company") -> str:
    profile = retrieve_profile(posting)   # same RAG retrieval as the matcher
    prompt = f"""Write a concise, professional cover letter (max 250 words) for this job.

RULES:
- Use ONLY experience and skills that appear in the CANDIDATE PROFILE. Never invent anything.
- Mention 2-3 specific things from the profile that match the posting.
- Confident but not arrogant. No cliches like "I am writing to express my interest".
- Address it to the hiring team at {company}.
- Output ONLY the letter itself. No preamble, no introduction like "Here is a cover letter", no commentary before or after.

CANDIDATE PROFILE:
{profile}

JOB POSTING:
{posting}"""
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.4},   # a little creativity, but not wild
    )
    return response["message"]["content"]

if __name__ == "__main__":
    job_file = Path(sys.argv[1] if len(sys.argv) > 1 else "jobs/example-role.md")
    company = sys.argv[2] if len(sys.argv) > 2 else "the company"
    print(write_cover_letter(job_file.read_text(), company))
