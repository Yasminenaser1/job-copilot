"""Eval suite for the Insights agent: does it find real themes and refuse fake ones?

Two kinds of case:
  - model cases   : run the agent on fixed synthetic pipelines
  - guard cases   : test the grounding filter directly, no model, no flake
"""
from insights_agent import find_gap_themes, _ground, SkillTheme, MIN_POSTINGS_PER_THEME

# (id, company, role, [missing keywords])
SECURITY_PIPELINE = [
    (1, "QuilrAI", "AI Solutions Engineer", ["OWASP", "red teaming", "application security"]),
    (2, "SecureCo", "AI Engineer", ["cybersecurity", "OWASP", "penetration testing methodologies"]),
    (3, "ThreatLab", "Automation Engineer", ["AI security", "red teaming", "threat modeling"]),
    (4, "Benzinga", "AI Engineer Data APIs", ["Cloud Computing", "API Design"]),
]

SCATTERED_PIPELINE = [
    (1, "Acme", "AI Automation Engineer", ["Playwright"]),
    (2, "Motion", "Software Integration Engineer", ["onsite commissioning activities"]),
    (3, "DesignCo", "Design Lead", ["public speaking"]),
    (4, "EduCorp", "Course Writer", ["video production teams"]),
]

def _mentions(themes, *terms) -> bool:
    blob = " ".join(t.theme.lower() + " " + " ".join(t.keywords).lower() for t in themes)
    return any(term in blob for term in terms)

def run_model_cases() -> tuple[int, int]:
    passed = 0
    print("🔍 Insights model cases...")

    themes = find_gap_themes(SECURITY_PIPELINE)
    ok = bool(themes) and _mentions(themes, "security", "owasp", "red team", "penetration")
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  security_theme_surfaces"
          + ("" if ok else f"\n       ↳ expected a security theme, got {[t.theme for t in themes]}"))
    passed += ok

    themes = find_gap_themes(SCATTERED_PIPELINE)
    ok = themes == []
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  no_pattern_invented"
          + ("" if ok else f"\n       ↳ expected no themes from unrelated gaps, got {[t.theme for t in themes]}"))
    passed += ok

    return passed, 2

def run_guard_cases() -> tuple[int, int]:
    """The grounding filter is the safety net. Test it without the model."""
    passed = 0
    print("\n🛡️  Insights guard cases...")

    hallucinated = SkillTheme(theme="Kubernetes", keywords=["Kubernetes"],
                              evidence_ids=[99, 100], why_it_matters="invented")
    ok = _ground([hallucinated], SECURITY_PIPELINE) == []
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  drops_invented_posting_ids")
    passed += ok

    one_off = SkillTheme(theme="Threat Modeling", keywords=["threat modeling"],
                         evidence_ids=[1, 3], why_it_matters="skill appears in only one posting")
    ok = _ground([one_off], SECURITY_PIPELINE) == []
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  drops_single_posting_theme")
    passed += ok

    generic = SkillTheme(theme="Leadership and Communication", keywords=["AI"],
                         evidence_ids=[1, 2], why_it_matters="rests on a generic term")
    ok = _ground([generic], SECURITY_PIPELINE) == []
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  drops_generic_keyword_theme")
    passed += ok

    invented_kw = SkillTheme(theme="Security", keywords=["OWASP", "Kubernetes"],
                             evidence_ids=[1, 2], why_it_matters="one keyword is invented")
    kept = _ground([invented_kw], SECURITY_PIPELINE)
    ok = len(kept) == 1 and kept[0].keywords == ["OWASP"]
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  strips_invented_keywords"
          + ("" if ok else f"\n       ↳ got {kept[0].keywords if kept else 'theme dropped'}"))
    passed += ok

    empty = find_gap_themes([SECURITY_PIPELINE[0]])
    ok = empty == []
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  no_themes_below_minimum_data")
    passed += ok

    return passed, 5

if __name__ == "__main__":
    gp, gt = run_guard_cases()
    mp, mt = run_model_cases()
    print(f"\n📊 {gp + mp}/{gt + mt} passed")
