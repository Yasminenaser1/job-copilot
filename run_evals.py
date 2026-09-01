"""Eval harness: run every test case through the matcher and report pass/fail."""
import json
from match import analyze

CONSISTENCY_TOLERANCE = 10   # two runs of the same posting must be within this many points

def run_case(case: dict) -> tuple[bool, list[str]]:
    """Returns (passed, list of failure reasons)."""
    failures = []

    # Run twice - once for the checks, once to measure consistency
    r1 = analyze(case["posting"])
    r2 = analyze(case["posting"])

    # Check 1: score in expected range
    if not (case["min_score"] <= r1.match_score <= case["max_score"]):
        failures.append(f"score {r1.match_score} outside [{case['min_score']}, {case['max_score']}]")

    # Check 2: required skills detected (case-insensitive substring match)
    found = " ".join(r1.matching_skills).lower()
    for skill in case["must_match"]:
        if skill.lower() not in found:
            failures.append(f"expected '{skill}' in matching_skills, got {r1.matching_skills}")

    # Check 3: consistency between two runs
    drift = abs(r1.match_score - r2.match_score)
    if drift > CONSISTENCY_TOLERANCE:
        failures.append(f"inconsistent: {r1.match_score} vs {r2.match_score} (drift {drift})")

    return (len(failures) == 0, failures)

def main():
    cases = json.load(open("evals/cases.json"))
    passed = 0
    print(f"🧪 Running {len(cases)} eval cases...\n")

    for case in cases:
        ok, failures = run_case(case)
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"{status}  {case['name']}")
        for f in failures:
            print(f"       ↳ {f}")
        passed += ok

    print(f"\n📊 {passed}/{len(cases)} passed ({100 * passed // len(cases)}%)")

if __name__ == "__main__":
    main()
