"""Eval suite for the Scout agent: does its judgment hold on known cases?"""
from scout_agent import judge_posting

SCOUT_CASES = [
    (
        "junior_ai_role_yes",
        "Junior AI Automation Engineer", "GrowthCo",
        "Build and maintain LLM-powered automation workflows in Python. RAG pipelines, "
        "FastAPI services, prompt engineering. 0-2 years experience, strong portfolio welcome.",
        True,
    ),
    (
        "keyword_bait_no",
        "AI Content Marketing Manager", "BuzzCorp",
        "Lead our AI-focused content strategy! Write about AI trends, manage our AI newsletter, "
        "coordinate with AI influencers. Passion for AI required. No coding involved.",
        False,
    ),
    (
        "seniority_trap_no",
        "Staff Machine Learning Engineer", "DeepScale",
        "Design distributed training infrastructure for foundation models. 8+ years ML experience, "
        "PhD preferred, deep expertise in CUDA and large-scale systems required.",
        False,
    ),
]

def run_scout_evals() -> tuple[int, int]:
    passed = 0
    print("🤖 Scout eval cases...")
    for name, role, company, posting, expected in SCOUT_CASES:
        verdict = judge_posting(role, company, posting)
        ok = verdict.worth_pursuing == expected
        print(f"{'✅ PASS' if ok else '❌ FAIL'}  {name}"
              + ("" if ok else f"\n       ↳ expected {expected}, got {verdict.worth_pursuing} ({verdict.reason})"))
        passed += ok
    return passed, len(SCOUT_CASES)

if __name__ == "__main__":
    p, t = run_scout_evals()
    print(f"\n📊 {p}/{t} passed")
