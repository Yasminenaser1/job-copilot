"""Week 2: the Letter agent - pre-drafts cover letters for high-scoring scouted jobs."""
from pathlib import Path
import re
from crewai import Agent, Task, Crew, LLM
from match import retrieve_profile

llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.4,   # letters want a little life; scoring wants none
)

writer = Agent(
    role="Cover Letter Writer",
    goal=(
        "Draft a concise, professional cover letter (max 250 words) grounded ONLY in the "
        "candidate profile provided. Never invent experience, skills, or credentials."
    ),
    backstory=(
        "You are a career writer who despises generic letters and hallucinated claims. "
        "Every sentence you write can be traced to something in the candidate's real profile. "
        "You open with substance, never with 'I am writing to express my interest'."
    ),
    llm=llm,
    verbose=False,   # letters don't need the monologue
)

def draft_letter(role: str, company: str, posting: str) -> str:
    profile = retrieve_profile(posting)   # same RAG retrieval the matcher uses
    task = Task(
        description=(
            f"Draft a cover letter for this job.\n\n"
            f"ROLE: {role}\nCOMPANY: {company}\n\n"
            f"CANDIDATE PROFILE (the ONLY permitted source of claims):\n{profile}\n\n"
            f"POSTING:\n{posting[:3000]}"
        ),
        expected_output="Only the letter text itself. No preamble, no commentary.",
        agent=writer,
    )
    return str(Crew(agents=[writer], tasks=[task]).kickoff())

def save_draft(app_id: int, company: str, role: str, letter: str) -> Path:
    Path("drafts").mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{company}-{role}".lower()).strip("-")[:60]
    path = Path("drafts") / f"{app_id:03d}-{slug}.md"
    path.write_text(f"# {role} @ {company}\n\n{letter}\n")
    return path

if __name__ == "__main__":
    letter = draft_letter(
        "Junior AI Automation Engineer", "TestCo",
        "Build LLM automation pipelines in Python. RAG, FastAPI, evals. Early-career welcome.",
    )
    print(letter)
