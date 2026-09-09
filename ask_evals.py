"""Eval suite for the Ask agent: does it answer from the data, and refuse everything else?

An open question box fails in a specific way - it answers. The cases that matter most
here are the ones where the correct output is a refusal.

  - guard cases : scope gate and grounding filter, no model, no flake
  - model cases : real questions against a fixed synthetic pipeline
"""
from ask_agent import (ask, in_scope, out_of_scope, _ground, _facts, _verdict,
                       _checked_span, _scope_refusal, Answer)

def _row(app_id, company, role, score, status, keywords: list[str], date="2026-09-02"):
    return {"id": app_id, "company": company, "role": role, "match_score": score,
            "status": status, "missing_keywords": keywords, "analyzed_on": date}

# missing_keywords is a list, because that is what tracker.list_applications() hands
# back after decode_keywords(). It used to be a comma-joined string here, which made
# _facts() iterate the string one character at a time and report "the skill 's'" as
# the top gap - the fixture was testing a row shape the app never produces.
PIPELINE = [
    _row(1, "QuilrAI", "AI Solutions Engineer", 80, "analyzed",
         ["OWASP", "red teaming", "application security"]),
    _row(2, "SecureCo", "AI Engineer", 78, "analyzed",
         ["cybersecurity", "OWASP", "penetration testing"]),
    _row(3, "ThreatLab", "Automation Engineer", 71, "scouted",
         ["AI security", "red teaming", "threat modeling"]),
    _row(4, "Benzinga", "AI Engineer Data APIs", 55, "scouted",
         ["Cloud Computing", "API Design"]),
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

    # The span rule, exercised without a model. These are the cases that decide
    # whether a refusal is evidence or prose.
    q = "Which roles is best for me?"
    passed += _check("verifies_a_real_span", _checked_span("best for me", q) == "best for me")
    passed += _check("rejects_a_fabricated_span", _checked_span("company culture", q) is None)
    passed += _check("rejects_a_too_short_span", _checked_span("me", q) is None)

    # A refusal the gate could not quote is not a refusal. This is the exact shape of
    # the reported bug: answerable=false with a reason found nowhere in the question.
    v = _verdict(False, "a company's culture, size, or products", q)
    passed += _check("fails_open_on_fabricated_refusal", v.answerable and v.span == "", str(v))

    v = _verdict(False, "engineering culture", "What is QuilrAI's engineering culture like?")
    passed += _check("keeps_refusal_that_quotes_the_question",
                     not v.answerable and v.span == "engineering culture", str(v))

    # B: the user-facing sentence is ours, keyed off the span, never model prose.
    msg = _scope_refusal("What is QuilrAI's engineering culture like?", "engineering culture")
    passed += _check("refusal_text_quotes_span_and_uses_our_sentence",
                     '"engineering culture"' in msg and "logs postings, not employers" in msg, msg)

    return passed, 15

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

# Twelve wordings of one question - "which of my rows has the highest match_score".
# Every one is answerable from a column the tracker has. A single phrasing used to
# stand here, which is exactly why the bug shipped: the gate never sees the rows, so
# it can only judge wording, and "best match" passed while "best for me" refused with
# an invented complaint about company culture. A phrasing-sensitivity bug cannot be
# caught by one phrasing. Add wordings here; do not thin them out.
PARAPHRASES = [
    "What is my best match so far?",
    "Which roles is best for me?",
    "Which role is best for me?",
    "Which roles are best for me?",
    "What are my best roles?",
    "Which posting has the highest match score?",
    "Which role fits me best?",
    "Which jobs suit me best?",
    "Rank my roles from best to worst.",
    "Which roles should I focus on?",
    "What's my top role?",
    "Which of these roles is the strongest match for me?",
]

# Genuinely outside the columns: answering needs facts about a real employer. The
# gate must still refuse these - it earns its keep here. Without it the answering
# model reads a missing-keywords list as the company's tech stack.
OUT_OF_REACH = [
    "What is QuilrAI's engineering culture like?",
    "Which of these companies is the biggest?",
    "What tech stack does Benzinga use?",
    "How much funding has SecureCo raised?",
]

def run_paraphrase_cases() -> tuple[int, int]:
    """The gate, on wording alone. One call per question, no answering call."""
    print("\n🔀 Ask gate paraphrase cases...")
    refused = [q for q in PARAPHRASES if not in_scope(q).answerable]
    ok = _check("gate_allows_every_wording_of_an_answerable_question", not refused,
                f"refused {len(refused)}/{len(PARAPHRASES)}: {refused}")

    allowed, unquoted = [], []
    for q in OUT_OF_REACH:
        scope = in_scope(q)
        if scope.answerable:
            allowed.append(q)
        # The fabrication assertion: a refusal must quote words that are really there.
        elif scope.span.lower() not in " ".join(q.lower().split()):
            unquoted.append((q, scope.span))
    ok2 = _check("gate_still_refuses_what_the_columns_cannot_reach", not allowed, f"allowed: {allowed}")
    ok3 = _check("every_refusal_quotes_the_question", not unquoted, f"fabricated spans: {unquoted}")
    return ok + ok2 + ok3, 3

if __name__ == "__main__":
    gp, gt = run_guard_cases()
    mp, mt = run_model_cases()
    pp, pt = run_paraphrase_cases()
    print(f"\n📊 {gp + mp + pp}/{gt + mt + pt} passed")
