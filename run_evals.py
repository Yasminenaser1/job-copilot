"""Eval harness: run every test case through the matcher and report pass/fail."""
import json
from match import analyze

CONSISTENCY_TOLERANCE = 10   # two runs of the same posting must be within this many points

def _expect(terms: list[str], actual: list[str], field: str, failures: list[str]):
    """Case-insensitive substring check that every term shows up in one field."""
    found = " ".join(actual).lower()
    for term in terms:
        if term.lower() not in found:
            failures.append(f"expected '{term}' in {field}, got {actual}")

def run_case(case: dict) -> tuple[bool, list[str]]:
    """Returns (passed, list of failure reasons)."""
    failures = []

    # Run twice - once for the checks, once to measure consistency
    r1 = analyze(case["posting"])
    r2 = analyze(case["posting"])

    # Check 1: score in expected range
    if not (case["min_score"] <= r1.match_score <= case["max_score"]):
        failures.append(f"score {r1.match_score} outside [{case['min_score']}, {case['max_score']}]")

    # Check 2: the posting's requirements land in the right bucket.
    #   must_match -> what the resume genuinely covers  (matching_skills)
    #   must_miss  -> what it doesn't                   (missing_keywords)
    # A wrong-fit case belongs in must_miss. Asserting "BSN degree" against
    # matching_skills tested nothing: an empty matching_skills is the correct
    # answer for a nursing role, so the case could only ever fail.
    _expect(case.get("must_match", []), r1.matching_skills, "matching_skills", failures)
    _expect(case.get("must_miss", []), r1.missing_keywords, "missing_keywords", failures)

    # Check 3: consistency between two runs
    drift = abs(r1.match_score - r2.match_score)
    if drift > CONSISTENCY_TOLERANCE:
        failures.append(f"inconsistent: {r1.match_score} vs {r2.match_score} (drift {drift})")

    return (len(failures) == 0, failures)

def run_all():
    cases = json.load(open("evals/cases.json"))
    passed = 0
    print(f"🧪 Running {len(cases)} eval cases...\n")
    for case in cases:
        ok, failures = run_case(case)
        print(f"{'✅ PASS' if ok else '❌ FAIL'}  {case['name']}")
        for f in failures:
            print(f"       ↳ {f}")
        passed += ok
    v_passed, v_total = run_validator_cases()
    total, total_passed = len(cases) + v_total, passed + v_passed
    print(f"\n📊 {total_passed}/{total} passed ({100 * total_passed // total}%)")




# ---- validator evals: inputs that must be accepted/rejected ----
from match import validate_posting

VALIDATOR_CASES = [
    ("readme_text", "Job Copilot is a full-stack AI application with RAG matching, AI cover letters, and an application tracker built with FastAPI and Ollama.", False),
    ("random_text", "The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet, consectetur adipiscing elit sed do eiusmod tempor.", False),
    ("real_posting", "Software Engineer at DataCorp. Requirements: 2+ years Python, SQL, REST API experience. Responsibilities include building data pipelines and maintaining internal services.", True),
]

def run_validator_cases() -> tuple[int, int]:
    passed = 0
    print("\n🛡️  Validator cases...")
    for name, text, expected in VALIDATOR_CASES:
        result = validate_posting(text)
        ok = result.is_job_posting == expected
        print(f"{'✅ PASS' if ok else '❌ FAIL'}  {name}" + ("" if ok else f"\n       ↳ expected {expected}, got {result.is_job_posting} ({result.reason})"))
        passed += ok
    return passed, len(VALIDATOR_CASES)


if __name__ == "__main__":
    run_all()
