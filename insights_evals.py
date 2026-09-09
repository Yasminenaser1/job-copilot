"""Eval suite for the Insights agent: does it find real themes and refuse fake ones?

Three kinds of case:
  - guard cases   : test the grounding gates directly, no model, no flake
  - real cases    : the same gates against real tracker rows, messy ones included
  - model cases   : run the whole agent on fixed pipelines

The synthetic pipelines below are clean: short keyword lists, one gap per string,
no duplicates. They passed while the real 23-row table produced an "AI and Machine
Learning" theme containing a JVM backend role, because nothing here looked like
the real data. REAL_PIPELINE fixes that - it is copied verbatim from tracker.db and
keeps the mess: row #15 lists LangGraph twice, row #13 lists "LangChain",
"LangGraph" and "orchestration frameworks like LangChain/LangGraph" as three
separate entries, row #12 hides three clouds inside one parenthesis, and #25 is the
Camunda backend role that the old union-of-ids rule kept filing under AI.
"""
from insights_agent import (find_gap_themes, _ground, recurring_terms, ungrouped_terms,
                            SkillTheme, NON_DISCRIMINATIVE_TERMS)

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

# Verbatim from tracker.db (ids preserved). Nine of the fifteen rows that logged
# gaps - the ones that carry the bug: two security roles, three AI/LLM roles, a
# DevOps role, a browser-automation role, a JVM backend role and a web-stack role.
REAL_PIPELINE = [
    (3, 'ByteHire', 'Automation engineer', [
        'Playwright', 'browser automation', 'RDP', 'virtual environments',
        'scheduled workflows', 'credential management', 'access controls', 'retries',
        'monitoring', 'structured logging', 'recovery', 'alerting', 'MFA',
        'sensitive healthcare information',
    ]),
    (5, 'Motion Recruitment', 'AI Software Engineer', [
        'Customer Collaboration', 'Solve Complex Problems', 'Feedback Loop',
        'Cybersecurity', 'OWASP', 'red teaming', 'penetration testing methodologies',
    ]),
    (6, 'QuilrAI', 'AI Solutions Engineer', [
        'cybersecurity', 'application security', 'AI security', 'OWASP', 'red teaming',
        'penetration testing methodologies',
    ]),
    (7, 'Benzinga', 'AI Engineer Data APIs', [
        'RAG matching over resume embeddings', 'vector databases (ChromaDB)', 'RAG',
        'embeddings',
    ]),
    (12, 'Machine Learning Reply', 'DevOps/ML Engineer (m/f/d)', [
        'DevOps', 'MLOps', 'cloud technologies (AWS, Azure or GCP)', 'Kubernetes', 'Java',
        'Scala', 'Apache Spark', 'Apache Kafka', 'Apache Airflow', 'Dagster',
    ]),
    (13, 'Machine Learning Reply', 'Agentic AI Software Engineer', [
        'agentic AI', 'generative AI applications', 'AWS', 'GCP', 'Azure',
        'cloud platforms', 'autonomous agents', 'scalable cloud architectures',
        'MLOps / LLMOps pipelines', 'CI/CD workflows', 'testing', 'monitoring',
        'observability', 'performance optimization', 'LangChain', 'LangGraph',
        'orchestration frameworks', 'vector databases', 'RAG pipelines',
        'orchestration frameworks like LangChain/LangGraph',
    ]),
    (15, 'adorsys GmbH', 'AI Engineer (m/w/d)', [
        'Multi-Agent-Systeme', 'autonomer Workflows', 'Konzeption von RAG-Architekturen',
        'hybride Retrieval-Strategien', 'Document-Processing-Pipelines',
        'Fine-tuning und Optimierung von LLMs', 'multimodale Modelle', 'Agent-Stacks',
        'MCP', 'Prompt-Engineering', 'Pipelines für Preprocessing', 'Entity Resolution',
        'Data Harmonization', 'Data-Engineering', 'Integration von LLM-Anwendungen',
        'Evaluation', 'Monitoring und Performance-Optimierung', 'cross-funktionalen Teams',
        'Agent Engineering', 'LangGraph', 'AutoGen', 'CrewAI', 'LangChain', 'LangGraph',
        'LLM-APIs', 'Claude', 'OpenAI', 'Gemini', 'SFT', 'RLHF', 'Production-ML',
        'Deployment', 'Monitoring', 'Performance', 'Wissensgraphen', 'Semantic Web',
        'Graph-ML', 'Cloud-AI-Plattformen', 'MLOps', 'AWS Bedrock', 'MLflow', 'Kubernetes',
        'Branchenkenntnisse', 'Steuern', 'Recht', 'Finanzen',
    ]),
    (25, 'Camunda', 'Software Engineer, Backend - Core/API & Process Automation', [
        'event-driven systems', 'event sourcing', 'distributed systems', 'consistency',
        'ordering', 'failure basics', 'API design instincts', 'REST',
        'backward compatibility', 'product thinking', 'workflow automation',
        'process orchestration', 'BPMN', 'Elasticsearch/OpenSearch', 'relational/Postgres',
        'observability', 'logging', 'metrics', 'tracing', 'JVM knowledge',
    ]),
    (26, 'ThriveCart', 'Software Engineer', [
        'TypeScript', 'Node.js', 'PHP', 'Laravel', 'MariaDB', 'Redis', 'AWS', 'Stripe',
        'PayPal', 'AI coding tools',
    ]),
]

# Postings that must never end up under the same theme as each other: the security
# pair and the Camunda backend role are the exact pairing the old code produced.
SECURITY_IDS = {5, 6}
CAMUNDA_ID = 25

def _check(name, ok, detail=""):
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  {name}" + ("" if ok else f"\n       ↳ {detail}"))
    return bool(ok)

def _mentions(themes, *terms) -> bool:
    blob = " ".join(t.theme.lower() + " " + " ".join(t.keywords).lower() for t in themes)
    return any(term in blob for term in terms)

def _one(theme, keywords, rows):
    """Ground a single hand-built theme and return it, or None if it was rejected."""
    kept = _ground([SkillTheme(theme=theme, keywords=keywords, evidence_ids=[99],
                               why_it_matters="hand-built")], rows)
    return kept[0] if kept else None

def run_guard_cases() -> tuple[int, int]:
    """The grounding filter is the safety net. Test it without the model."""
    passed = 0
    print("\n🛡️  Insights guard cases...")

    hallucinated = SkillTheme(theme="Kubernetes", keywords=["Kubernetes"],
                              evidence_ids=[99, 100], why_it_matters="invented")
    passed += _check("drops_invented_posting_ids", _ground([hallucinated], SECURITY_PIPELINE) == [])

    one_off = SkillTheme(theme="Threat Modeling", keywords=["threat modeling"],
                         evidence_ids=[1, 3], why_it_matters="skill appears in only one posting")
    passed += _check("drops_single_posting_theme", _ground([one_off], SECURITY_PIPELINE) == [])

    generic = SkillTheme(theme="Leadership and Communication", keywords=["AI"],
                         evidence_ids=[1, 2], why_it_matters="rests on a generic term")
    passed += _check("drops_generic_keyword_theme", _ground([generic], SECURITY_PIPELINE) == [])

    invented_kw = SkillTheme(theme="Security", keywords=["OWASP", "Kubernetes"],
                             evidence_ids=[1, 2], why_it_matters="one keyword is invented")
    kept = _ground([invented_kw], SECURITY_PIPELINE)
    ok = len(kept) == 1 and kept[0].keywords == ["OWASP"]
    passed += _check("strips_invented_keywords", ok,
                     f"got {kept[0].keywords if kept else 'theme dropped'}")

    passed += _check("no_themes_below_minimum_data", find_gap_themes([SECURITY_PIPELINE[0]]) == [])

    return passed, 5

def run_real_cases() -> tuple[int, int]:
    """The same gates, against the rows that actually broke it."""
    passed = 0
    print("\n🧾 Insights real-row cases...")

    terms = recurring_terms(REAL_PIPELINE)
    raw_count = sum(len(k) for _, _, _, k in REAL_PIPELINE)

    # Row #15 says LangGraph twice and row #13 says LangChain in two separate
    # entries. Both are still one posting each, so both terms sit at exactly two.
    ok = terms.get("langgraph") == {13, 15} and terms.get("langchain") == {13, 15}
    passed += _check("counts_postings_not_keyword_strings", ok,
                     f"langgraph={terms.get('langgraph')} langchain={terms.get('langchain')}")

    # "cloud technologies (AWS, Azure or GCP)" is three gaps in one string.
    ok = terms.get("azure") == {12, 13} and terms.get("gcp") == {12, 13}
    passed += _check("unpacks_parenthesised_lists", ok,
                     f"azure={terms.get('azure')} gcp={terms.get('gcp')}")

    # The model must be handed a shortlist, not the raw log.
    ok = len(terms) <= 25 and raw_count > 100
    passed += _check("shortlist_is_short", ok, f"{len(terms)} terms from {raw_count} strings")

    ok = not (set(terms) & NON_DISCRIMINATIVE_TERMS)
    passed += _check("bridge_terms_never_reach_the_model", ok,
                     f"leaked {sorted(set(terms) & NON_DISCRIMINATIVE_TERMS)}")

    # The reported failure, reproduced: security terms and ops terms share no
    # posting, so this grouping is incoherent and the label cannot be true of both.
    bug = _one("AI and Machine Learning",
               ["AI", "MLOps", "monitoring", "observability", "OWASP", "red teaming"],
               REAL_PIPELINE)
    passed += _check("rejects_cross_domain_grouping", bug is None,
                     f"kept it with evidence {bug.evidence_ids if bug else None}")

    # Evidence is the postings that support the theme, not every id any keyword
    # touches. #26 names AWS inside a TypeScript/PHP/Redis stack - one term, not a
    # cloud-platform gap.
    cloud = _one("Cloud platforms", ["AWS", "GCP", "Azure", "Kubernetes"], REAL_PIPELINE)
    ok = cloud is not None and cloud.evidence_ids == [12, 13, 15]
    passed += _check("evidence_is_not_the_union_of_keywords", ok,
                     f"got {cloud.evidence_ids if cloud else 'theme dropped'}, expected [12, 13, 15]")

    # ...but the gates must not prune an honest theme. monitoring/observability are
    # deliberately unblocked, and Camunda genuinely belongs here.
    ops = _one("Operational maturity", ["monitoring", "observability"], REAL_PIPELINE)
    ok = ops is not None and CAMUNDA_ID in ops.evidence_ids and len(ops.evidence_ids) >= 3
    passed += _check("keeps_camunda_in_the_ops_theme", ok,
                     f"got {ops.evidence_ids if ops else 'theme dropped'}")

    # A term the model leaves out must still reach the user. kubernetes is in #12
    # and #15, so it cleared every gate; being unnamed is not the same as being a
    # one-off, and until ungrouped_terms() there was no output path that could
    # tell those two apart. The omission is hand-built rather than provoked out of
    # the model, so this case cannot flake.
    cloud = [SkillTheme(theme="Cloud platforms", keywords=["AWS", "GCP", "Azure"],
                        why_it_matters="hand-built, deliberately omits kubernetes")]
    left = ungrouped_terms(cloud, REAL_PIPELINE)
    passed += _check("omitted_term_still_reported", left.get("kubernetes") == {12, 15},
                     f"kubernetes={left.get('kubernetes')}, expected {{12, 15}}")

    # The join runs through _norm, so the model's casing must not strand a term it
    # did group - "AWS" in a theme has to cancel "aws" on the shortlist.
    ok = not ({"aws", "gcp", "azure"} & set(left))
    passed += _check("grouped_terms_are_not_relisted", ok,
                     f"relisted {sorted({'aws', 'gcp', 'azure'} & set(left))}")

    # Zero themes is not evidence of zero patterns. This is the state the old
    # "the gaps so far are one-offs" line described, wrongly, as no pattern.
    ok = ungrouped_terms([], REAL_PIPELINE) == recurring_terms(REAL_PIPELINE)
    passed += _check("no_themes_still_reports_whole_shortlist", ok,
                     f"got {sorted(ungrouped_terms([], REAL_PIPELINE))}")

    # Nothing recurs -> nothing to also-report, and the one-offs line is honest.
    ok = ungrouped_terms([], SCATTERED_PIPELINE) == {}
    passed += _check("empty_shortlist_reports_nothing", ok,
                     f"got {sorted(ungrouped_terms([], SCATTERED_PIPELINE))}")

    return passed, 11

def run_model_cases() -> tuple[int, int]:
    passed = 0
    print("\n🔍 Insights model cases...")

    themes = find_gap_themes(SECURITY_PIPELINE)
    passed += _check("security_theme_surfaces",
                     bool(themes) and _mentions(themes, "security", "owasp", "red team", "penetration"),
                     f"expected a security theme, got {[t.theme for t in themes]}")

    themes = find_gap_themes(SCATTERED_PIPELINE)
    passed += _check("no_pattern_invented", themes == [],
                     f"expected no themes from unrelated gaps, got {[t.theme for t in themes]}")

    themes = find_gap_themes(REAL_PIPELINE)
    passed += _check("real_rows_produce_themes", bool(themes),
                     "expected at least one theme from nine real postings")

    bad = [t.theme for t in themes
           if set(t.evidence_ids) & SECURITY_IDS and CAMUNDA_ID in t.evidence_ids]
    passed += _check("no_theme_spans_security_and_backend", not bad,
                     f"these themes cite both a security posting and #{CAMUNDA_ID}: {bad}")

    return passed, 4

if __name__ == "__main__":
    gp, gt = run_guard_cases()
    rp, rt = run_real_cases()
    mp, mt = run_model_cases()
    print(f"\n📊 {gp + rp + mp}/{gt + rt + mt} passed")
