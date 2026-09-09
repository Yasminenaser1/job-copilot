"""Eval harness: run every test case through the matcher and report pass/fail.

Three groups now, because v2 of match.py has three things that can break:
  - case evals    : does a posting land in the right score range, with the right
                    requirements in the right half of the partition
  - partition evals: the structural guarantees the split was built to buy - row
                    #7's failure, and monotonicity of the score
  - validator evals: inputs that must be accepted or rejected before scoring
"""
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

def _check(name, ok, detail=""):
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  {name}" + ("" if ok else f"\n       ↳ {detail}"))
    return bool(ok)

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
    p_passed, p_total = run_partition_cases()
    v_passed, v_total = run_validator_cases()
    total = len(cases) + p_total + v_total
    total_passed = passed + p_passed + v_passed
    print(f"\n📊 {total_passed}/{total} passed ({100 * total_passed // total}%)")



# ---- partition evals: the two guarantees the v2 split was built to buy ----
#
# These are regressions, not quality checks. Each one encodes a failure that
# actually shipped, or a property the scoring formula claims in its docstring.

# Row #7 (Benzinga, AI Engineer Data APIs) scored 80 with these four in
# missing_keywords: "RAG matching over resume embeddings", "vector databases
# (ChromaDB)", "RAG", "embeddings". All four are verbatim strings from
# profile/resume.md, and the first is a project blurb that cannot appear in any
# posting - the model was reading requirements out of the candidate profile.
#
# The posting below asks for exactly the things resume.md's skills line names, so
# every one of them must land in matching_skills and none in missing_keywords.
BENZINGA_SHAPED = """AI Engineer, Data APIs at Benzinga.

We are hiring an AI Engineer to build and ship the data APIs behind our financial
news products.

Requirements:
- Strong Python
- Experience building RAG pipelines
- Experience with embeddings
- Experience with vector databases such as ChromaDB
- FastAPI or a comparable Python web framework
- SQL
Early-career candidates are welcome to apply."""

RESUME_TERMS = ["rag", "embeddings", "chromadb", "fastapi", "python", "sql"]

# Monotonicity: the same posting, plus five requirements the resume does not meet.
# score_report() claims an unmet requirement can only grow the denominator, so the
# score must not rise. Asserted on the real model, not on the formula alone,
# because the formula is only monotone if the judgements stay stable.
MONOTONICITY_BASE = """Backend Engineer at DataCorp.

Requirements:
- Python
- SQL
- Docker
- Git"""

MONOTONICITY_PLUS_UNMET = MONOTONICITY_BASE + """
- Kubernetes and Helm in production
- Terraform and infrastructure as code
- Kafka and streaming data pipelines
- Go or Rust for performance-critical services
- German language, C1 or better"""

def run_partition_cases() -> tuple[int, int]:
    passed = 0
    print("\n🧩 Partition cases...")

    r = analyze(BENZINGA_SHAPED)

    # The #7 regression itself.
    leaked = [k for k in r.missing_keywords
              if any(term in k.lower() for term in RESUME_TERMS)]
    passed += _check("resume_terms_never_land_in_missing", not leaked, f"leaked: {leaked}")

    # ...and the other half of it: they have to be somewhere, so they must be in
    # matching_skills. Without this, an empty report would pass the check above.
    found = " ".join(r.matching_skills).lower()
    absent = [t for t in RESUME_TERMS if t not in found]
    passed += _check("resume_terms_land_in_matching", not absent,
                     f"missing from matching_skills: {absent} (got {r.matching_skills})")

    # The partition property, stated directly: the two lists are disjoint, and
    # together they are the scored requirements. Pure set arithmetic over one report.
    scored = {j.text for j in r.requirements if j.weight > 0}
    both = set(r.matching_skills) & set(r.missing_keywords)
    passed += _check("halves_are_disjoint", not both, f"in both halves: {sorted(both)}")
    passed += _check("halves_cover_every_scored_requirement",
                     set(r.matching_skills) | set(r.missing_keywords) == scored,
                     f"partition {sorted(set(r.matching_skills) | set(r.missing_keywords))} "
                     f"!= scored {sorted(scored)}")

    base = analyze(MONOTONICITY_BASE)
    more = analyze(MONOTONICITY_PLUS_UNMET)
    passed += _check("unmet_requirements_never_raise_the_score",
                     more.match_score <= base.match_score,
                     f"{base.match_score} -> {more.match_score} after adding 5 unmet requirements")

    return passed, 5


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
