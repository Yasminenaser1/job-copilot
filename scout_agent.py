"""Week 2: the Scout agent - judgment layer for whether a posting is worth Yasmine's time."""
from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel

llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0,
)

class ScoutVerdict(BaseModel):
    worth_pursuing: bool
    reason: str

scout = Agent(
    role="Job Scout",
    goal=(
        "Decide if a job posting is genuinely worth an early-career AI automation "
        "engineer's time. Approve: junior/mid AI, LLM, automation, Python backend roles. "
        "Reject: senior/principal/staff roles, non-engineering roles (course writers, "
        "designers, teachers, sales), and jobs whose core is a field she is not in "
        "(security, hardware, ML research)."
    ),
    backstory=(
        "You are a pragmatic recruiter who has read thousands of postings. You know that "
        "keywords lie: a 'Course Writer, UX/AI' posting mentions AI constantly but is a "
        "writing job, not an engineering job. You judge the ROLE, not the buzzwords."
    ),
    llm=llm,
    verbose=True,
)

def judge_posting(role: str, company: str, posting: str) -> ScoutVerdict:
    task = Task(
        description=(
            f"Judge this posting for an early-career AI automation engineer.\n\n"
            f"ROLE: {role}\nCOMPANY: {company}\n\nPOSTING:\n{posting[:3000]}"
        ),
        expected_output=(
            'A JSON object: {"worth_pursuing": true or false, "reason": "one short sentence"}'
        ),
        agent=scout,
        output_pydantic=ScoutVerdict,
    )
    crew = Crew(agents=[scout], tasks=[task])
    result = crew.kickoff()
    return result.pydantic

if __name__ == "__main__":
    # smoke test: one obvious yes, one obvious no
    yes = judge_posting(
        "Junior AI Automation Engineer", "TestCo",
        "Build LLM automation pipelines in Python. RAG, FastAPI. Early-career welcome.",
    )
    print(f"\n✅ expected YES → {yes.worth_pursuing}: {yes.reason}")

    no = judge_posting(
        "Course Writer, UX/UI and AI", "TestCo",
        "Write engaging course content about AI and design. Strong writing skills required.",
    )
    print(f"\n❌ expected NO → {no.worth_pursuing}: {no.reason}")
