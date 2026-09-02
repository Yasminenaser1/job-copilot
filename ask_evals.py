"""Eval suite for the Ask agent: does it answer from the data, and refuse everything else?

An open question box fails in a specific way - it answers. The cases that matter most
here are the ones where the correct output is a refusal.

  - guard cases : scope gate and grounding filter, no model, no flake
  - model cases : real questions against a fixed synthetic pipeline
"""
from ask_agent import ask, out_of_scope, _ground, _facts, Answer

def _row(app_id, company, role, score, status, keywords, date="2026-09-02"):
    return {"id": app_id, "company": company, "role": role, "match_score": score,
            "status": status, "missing_keywords": keywords, "analyzed_on": date}

PIPELINE = [
    _row(1, "QuilrAI", "AI Solutions Engineer", 80, "analyzed", "OWASP, red teaming, application security"),
    _row(2, "SecureCo", "AI Engineer", 78, "analyzed", "cybersecurity, OWASP, penetration testing"),
    _row(3, "ThreatLab", "Automation Engineer", 71, "scouted", "AI security, red teaming, threat modeling"),
    _row(4, "Benzinga", "AI Engineer Data APIs", 55, "scouted", "Cloud Computing, API Design"),
]

def _check(name, ok, detail=""):
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  {name}" + ("" if ok else f"\n       ↳ {detail}"))
    return bool(ok)

def run_guard_cases() -> tuple[int, int]:
    passed = 0
    print("🛡️  Ask guard cases...")

    passed += _check("blocks_salary_question", out_of_scope("which of these pays the most?") is not None)
    passed += _check("blocks_prediction_question", out_of_scope("will I get the QuilrAI job?") is not None)
    passed += _check("allows_real_question", out_of_scope("what skill am I missing most?") is None)

    a = ask("hi", PIPELINE)
    passed += _check("rejects_empty_question", not a.answerable, a.answer)

    a = ask("why " * 200, PIPELINE)
    passed += _check("rejects_overlong_question", not a.answerable, a.answer)

    a = ask("what is my best match?", [])
    passed += _check("refuses_with_no_data", not a.answerable, a.answer)

    invented = Answer(answerable=True, answer="Google was your strongest match.", evidence_ids=[99])
    passed += _check("drops_invented_posting_ids", not _ground(invented, PIPELINE).answerable)

    mixed = Answer(answerable=True, answer="Security keeps coming up.", evidence_ids=[1, 2, 99])
    grounded = _ground(mixed, PIPELINE)
    passed += _check("strips_invented_ids_keeps_real",
                     grounded.answerable and grounded.evidence_ids == [1, 2],
                     f"got {grounded.evidence_ids}")

    facts = _facts(PIPELINE)
    passed += _check("states_absent_statuses_as_zero", "'applied': 0" in facts, facts)

    return passed, 9

def run_model_cases() -> tuple[int, int]:
    passed = 0
    print("\n🔍 Ask model cases...")

    a = ask("What skill keeps costing me points?", PIPELINE)
    ok = a.answerable and any(t in a.answer.lower() for t in ("security", "owasp", "red team", "penetration"))
    passed += _check("answers_from_the_data", ok, f"got: {a.answer}")

    # No row has status 'applied', so the only honest answer is zero.
    a = ask("How many jobs have I applied to?", PIPELINE)
    ok = any(t in a.answer.lower() for t in ("0", "zero", "none", "no ", "haven't", "not applied"))
    passed += _check("does_not_invent_a_funnel", ok, f"got: {a.answer}")

    a = ask("Which company has the best engineering culture?", PIPELINE)
    passed += _check("refuses_what_tracker_cannot_know", not a.answerable, f"got: {a.answer}")

    return passed, 3

if __name__ == "__main__":
    gp, gt = run_guard_cases()
    mp, mt = run_model_cases()
    print(f"\n📊 {gp + mp}/{gt + mt} passed")
