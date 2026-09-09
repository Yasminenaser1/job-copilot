"""Eval suite for the Archaeology agent: does a hypothesis have to earn its evidence?

Three kinds of case:
  - guard cases : the boundary, grounding and headline logic directly, no model, no flake
  - real cases  : the same logic against the postings actually stored in tracker.db
  - model cases : the whole agent on real rows, which is where the definitions are tested

The synthetic posting below is small and tidy, and a tidy fixture is exactly what
missed all of this the first time: bare hypothesis names and an ungrounded headline
both look fine until a real 7,000-character posting arrives with a marketing blurb
bolted to the front. So the real cases read the three stored descriptions instead of
embedding them - three postings run to 23KB and the text already lives in tracker.db,
which this module (like the agent) only ever SELECTs from.

What each stored row is here to catch:
  #24 RoomPriceGenie - every finding it produced quoted the company blurb, and one
      cited "Your job isn't to build raw ML models" as evidence of unclear scope
  #25 Camunda        - the headline announced an understaffed team with unclear
      responsibilities while zero such findings survived; it also says, at 58% depth,
      "This role is an existing vacancy", which no reading ever quoted
  #26 ThriveCart     - the control: a posting whose real signal sits just below the
      boundary, so an over-eager blurb rule would show up here first
"""
from typing import get_args

from archaeology_agent import (read_posting, _ground, _boundary, _normalize,
                               _compose_overall, _confidence, _verified,
                               Finding, HYPOTHESES, NOTHING_FOUND,
                               MAX_CONFIDENCE, MAX_PHRASE_CHARS)
from tracker import list_applications

# A posting with the shape the real ones have: a company blurb, a role section, and
# a marker phrase ("the opportunity") sitting mid-sentence inside the blurb where a
# naive boundary search would trip over it.
BLURB_POSTING = (
    "About Northwind. Founded in 2014, Northwind helps regional grocers forecast "
    "demand. We are proud to be recognised as a Best Place to Work 2026 and we have "
    "doubled our customer base every year since launch. Our investors backed us "
    "because they saw the opportunity to reshape an entire category. You will make "
    "an impact from day one and we would love you to be part of that story. "
    "Your Role You will be the first data engineer on the platform team. "
    "You will own the ingestion pipeline end to end, run it in production, carry the "
    "pager for it, and answer support tickets from grocers when a forecast looks "
    "wrong. You will also help the sales team demo the product. "
    "This role is an existing vacancy and you will take over from the engineer who "
    "built the current pipeline. "
    "Requirements Five years of Python. Comfort with Airflow and dbt."
)

HEAD_PHRASE = "we have doubled our customer base every year since launch"
HEAD_BOILERPLATE = "You will make an impact from day one"
BODY_PHRASE = "You will own the ingestion pipeline end to end"
BODY_PHRASE_2 = "You will also help the sales team demo the product"
BODY_PHRASE_3 = "This role is an existing vacancy"

# No section heading anywhere, so the boundary must fail open rather than fence
# off text it cannot locate.
UNMARKED_POSTING = (
    "Northwind is hiring. We build forecasting tools for grocers and we have been "
    "doing it since 2014. The person we want is comfortable owning a pipeline, "
    "carrying a pager, and talking to customers when a number looks wrong."
)

STORED_IDS = (24, 25, 26)

# Verbatim from the stored rows. Each one is a quote a previous reading produced or
# should have produced, kept here so the real cases assert on fixed text rather than
# on whatever the model says today.
BLURB_QUOTE_24 = "We invite you to join us on this journey!"
PRECISION_QUOTE_24 = "Your job isn't to build raw ML models or write throwaway demos."
LONG_QUOTE_25 = ("Build and evolve features in the event-sourced engine and its C8 "
                 "REST/gRPC/MCP API (command/event, CQRS), with guidance on the "
                 "harder design calls.")
BACKFILL_QUOTE_25 = "This role is an existing vacancy"

# Known failures, recorded rather than hidden. See _check(xfail=...).
XFAILED = []

def _check(name, ok, detail="", xfail=""):
    """xfail is a reason string: this case is known to fail and does not fail the
    suite. It is still run, and an unexpected pass prints XPASS so the marker gets
    removed once the failure stops happening rather than masking a later one."""
    if xfail:
        XFAILED.append(f"{name} - {xfail}")
        print(f"{'❗ XPASS' if ok else '⚠️  XFAIL'}  {name}\n       ↳ {xfail}"
              + ("" if ok else f"\n       ↳ {detail}"))
        return True
    print(f"{'✅ PASS' if ok else '❌ FAIL'}  {name}" + ("" if ok else f"\n       ↳ {detail}"))
    return bool(ok)

def _finding(hypothesis, phrases, confidence=0.8):
    """A hand-built finding. confidence defaults high on purpose: every case that
    checks a number is also checking that the model's own number was discarded."""
    return Finding(hypothesis=hypothesis, evidence_phrases=list(phrases),
                   implication="hand-built", confidence=confidence)

def _stored() -> dict[int, str]:
    return {a["id"]: a["description"] for a in list_applications()
            if a.get("description") and a["id"] in STORED_IDS}

def _hypotheses(findings) -> list[str]:
    return [f.hypothesis for f in findings]

def run_guard_cases() -> tuple[int, int]:
    """The gates are the safety net. Test them without the model."""
    results = []
    print("\n🛡️  Archaeology guard cases...")

    # --- the hypotheses themselves ---
    literal = set(get_args(Finding.model_fields["hypothesis"].annotation))
    results.append(_check("hypotheses_match_the_literal", set(HYPOTHESES) == literal,
                          f"dict={sorted(HYPOTHESES)} literal={sorted(literal)}"))

    incomplete = [name for name, h in HYPOTHESES.items()
                  if not (h.means and h.looks_like and h.not_this and h.summary)]
    results.append(_check("every_hypothesis_is_fully_defined", not incomplete,
                          f"incomplete: {incomplete}"))

    # _compose_overall reads summary and nothing else, so a summary that starts with
    # a capital or ends in a full stop would render mid-sentence.
    malformed = [name for name, h in HYPOTHESES.items()
                 if h.summary[:1].isupper() or h.summary.endswith(".")]
    results.append(_check("summaries_compose_as_clauses", not malformed,
                          f"not clause-shaped: {malformed}"))

    # --- the boundary ---
    hay = _normalize(BLURB_POSTING)
    boundary = _boundary(hay)
    results.append(_check("boundary_ignores_midsentence_marker",
                          0 < boundary <= hay.find("your role"),
                          f"boundary at {boundary}, 'your role' at {hay.find('your role')}"))
    results.append(_check("boundary_puts_blurb_above_and_role_below",
                          hay.find(_normalize(HEAD_PHRASE)) < boundary
                          <= hay.find(_normalize(BODY_PHRASE)),
                          f"head={hay.find(_normalize(HEAD_PHRASE))} boundary={boundary} "
                          f"body={hay.find(_normalize(BODY_PHRASE))}"))
    results.append(_check("boundary_fails_open_without_markers",
                          _boundary(_normalize(UNMARKED_POSTING)) == 0,
                          f"got {_boundary(_normalize(UNMARKED_POSTING))}"))

    # A marker past MAX_BOUNDARY_FRACTION is a word in a sentence, not a heading.
    late = "x. " * 400 + "Responsibilities: own the pipeline."
    results.append(_check("boundary_ignores_late_marker", _boundary(_normalize(late)) == 0,
                          f"got {_boundary(_normalize(late))}"))

    # --- grounding ---
    invented = _finding("backfill", ["We are winding down the platform team"])
    results.append(_check("drops_invented_phrase",
                          _ground([invented], BLURB_POSTING) == []))

    results.append(_check("drops_boilerplate_only_finding",
                          _ground([_finding("growth_hire", [HEAD_BOILERPLATE])],
                                  BLURB_POSTING) == []))

    over_long = "x" * (MAX_PHRASE_CHARS + 1)
    results.append(_check("drops_overlong_phrase",
                          _ground([_finding("backfill", [over_long])], BLURB_POSTING) == []))

    # The reported failure, in miniature: a hypothesis resting on the blurb alone.
    results.append(_check("drops_header_only_finding",
                          _ground([_finding("growth_hire", [HEAD_PHRASE])],
                                  BLURB_POSTING) == []))

    # ...and the softer half of that rule: the blurb phrase stays as corroboration
    # once a body phrase has established the finding.
    kept = _ground([_finding("understaffed_team", [HEAD_PHRASE, BODY_PHRASE])], BLURB_POSTING)
    ok = len(kept) == 1 and kept[0].evidence_phrases == [HEAD_PHRASE, BODY_PHRASE]
    results.append(_check("keeps_header_phrase_as_corroboration", ok,
                          f"got {kept[0].evidence_phrases if kept else 'finding dropped'}"))

    # --- confidence ---
    results.append(_check("confidence_ladder_climbs_with_evidence",
                          (_confidence(1), _confidence(2), _confidence(3), _confidence(9))
                          == (0.4, 0.6, MAX_CONFIDENCE, MAX_CONFIDENCE),
                          f"got {[_confidence(n) for n in (1, 2, 3, 9)]}"))

    # The docstring used to claim this and line 125 used to clamp instead: a finding
    # submitted at 0.8 on one phrase came out at 0.6, and one submitted at 0.4 came
    # out at 0.4 untouched.
    kept = _ground([_finding("understaffed_team", [BODY_PHRASE], confidence=0.8)], BLURB_POSTING)
    results.append(_check("confidence_ignores_the_models_number",
                          len(kept) == 1 and kept[0].confidence == 0.4,
                          f"got {kept[0].confidence if kept else 'finding dropped'}"))

    kept = _ground([_finding("understaffed_team", [BODY_PHRASE, BODY_PHRASE_2],
                             confidence=0.1)], BLURB_POSTING)
    results.append(_check("confidence_rises_with_a_second_body_phrase",
                          len(kept) == 1 and kept[0].confidence == 0.6,
                          f"got {kept[0].confidence if kept else 'finding dropped'}"))

    # A blurb quote can corroborate a finding but cannot establish one, so it must
    # not raise the number either.
    kept = _ground([_finding("understaffed_team", [BODY_PHRASE, HEAD_PHRASE])], BLURB_POSTING)
    results.append(_check("header_phrase_does_not_raise_confidence",
                          len(kept) == 1 and kept[0].confidence == 0.4,
                          f"got {kept[0].confidence if kept else 'finding dropped'}"))

    # --- exclusivity ---
    pair = [_finding("growth_hire", [BODY_PHRASE]),
            _finding("understaffed_team", [BODY_PHRASE])]
    kept = _ground(pair, BLURB_POSTING)
    results.append(_check("one_line_cannot_support_two_hypotheses", len(kept) == 1,
                          f"kept {_hypotheses(kept)}"))

    # A finding dropped after exclusivity strips its body phrase must not walk off
    # with the header phrase it never got to use. backfill outranks both (two body
    # phrases) and takes BODY_PHRASE; growth_hire is left header-only and dies;
    # unclear_scope must still find HEAD_PHRASE free to corroborate with.
    trio = [_finding("backfill", [BODY_PHRASE, BODY_PHRASE_2]),
            _finding("growth_hire", [BODY_PHRASE, HEAD_PHRASE]),
            _finding("unclear_scope", [BODY_PHRASE_3, HEAD_PHRASE])]
    kept = _ground(trio, BLURB_POSTING)
    unclear = next((f for f in kept if f.hypothesis == "unclear_scope"), None)
    ok = (_hypotheses(kept) == ["backfill", "unclear_scope"]
          and unclear is not None and HEAD_PHRASE in unclear.evidence_phrases)
    results.append(_check("dropped_finding_releases_its_evidence", ok,
                          f"kept {_hypotheses(kept)}, unclear_scope has "
                          f"{unclear.evidence_phrases if unclear else None}"))

    # Strongest-first is now a fact about the evidence, not the model's self-rating:
    # the weakly-rated finding with two body phrases must outrank the confident one.
    ranked = _ground([_finding("growth_hire", [BODY_PHRASE], confidence=0.9),
                      _finding("backfill", [BODY_PHRASE_2, BODY_PHRASE_3], confidence=0.1)],
                     BLURB_POSTING)
    results.append(_check("ranking_follows_evidence_not_confidence",
                          _hypotheses(ranked)[:1] == ["backfill"], f"got {_hypotheses(ranked)}"))

    # --- the headline ---
    results.append(_check("overall_is_constant_when_nothing_survives",
                          _compose_overall([]) == NOTHING_FOUND))

    kept = _ground([_finding("backfill", [BODY_PHRASE_3]),
                    _finding("understaffed_team", [BODY_PHRASE, BODY_PHRASE_2])],
                   BLURB_POSTING)
    overall = _compose_overall(kept)
    named = {name for name, h in HYPOTHESES.items() if h.summary in overall}
    results.append(_check("overall_names_exactly_the_kept_hypotheses",
                          named == set(_hypotheses(kept)),
                          f"overall names {sorted(named)}, kept {_hypotheses(kept)}"))

    # The reported failure: a headline asserting hypotheses that did not survive.
    # growth_hire is header-only here, so it must not reach the sentence.
    kept = _ground([_finding("growth_hire", [HEAD_PHRASE]),
                    _finding("backfill", [BODY_PHRASE_3])], BLURB_POSTING)
    overall = _compose_overall(kept)
    results.append(_check("overall_cannot_outrun_the_evidence",
                          HYPOTHESES["growth_hire"].summary not in overall
                          and HYPOTHESES["backfill"].summary in overall,
                          f"got {overall!r}"))

    dup = [_finding("backfill", [BODY_PHRASE_3]), _finding("backfill", [BODY_PHRASE])]
    overall = _compose_overall(_ground(dup, BLURB_POSTING))
    results.append(_check("overall_does_not_repeat_a_hypothesis",
                          overall.count(HYPOTHESES["backfill"].summary) == 1, f"got {overall!r}"))

    # --- the short-posting path (no model call) ---
    reading = read_posting("too short to read")
    results.append(_check("short_posting_is_not_read",
                          reading.findings == [] and "Not enough" in reading.overall,
                          f"got {reading.overall!r}"))

    return sum(results), len(results)

def run_real_cases() -> tuple[int, int]:
    """The same gates, against the postings that actually broke them. No model."""
    results = []
    print("\n🧾 Archaeology real-row cases...")

    stored = _stored()
    missing = [i for i in STORED_IDS if i not in stored]
    if missing:
        print(f"       ↳ skipped: no stored description for {missing}. "
              "Run scout_run.py to re-ingest.")
        return 0, 0

    # Every stored posting has a locatable blurb boundary. If this fails the rule
    # has silently stopped applying - it fails open, so nothing else would complain.
    unlocated = [i for i, d in stored.items() if _boundary(_normalize(d)) == 0]
    results.append(_check("boundary_located_in_every_stored_posting", not unlocated,
                          f"failed open on {unlocated}"))

    hay24 = _normalize(stored[24])
    b24 = _boundary(hay24)
    results.append(_check("blurb_boundary_precedes_the_role_section",
                          hay24.find(_normalize(BLURB_QUOTE_24)) < b24
                          <= hay24.find(_normalize(PRECISION_QUOTE_24)),
                          f"blurb quote at {hay24.find(_normalize(BLURB_QUOTE_24))}, "
                          f"boundary at {b24}"))

    # The exact finding #24 produced: growth_hire on a line from the company blurb.
    results.append(_check("marketing_blurb_cannot_carry_a_finding",
                          _ground([_finding("growth_hire", [BLURB_QUOTE_24])], stored[24]) == [],
                          "a blurb-only finding survived on #24"))

    # ...while the same blurb line still corroborates a finding with body evidence.
    kept = _ground([_finding("unclear_scope", [BLURB_QUOTE_24, PRECISION_QUOTE_24])], stored[24])
    results.append(_check("blurb_quote_survives_as_corroboration",
                          len(kept) == 1 and BLURB_QUOTE_24 in kept[0].evidence_phrases,
                          f"got {kept[0].evidence_phrases if kept else 'finding dropped'}"))

    # 147 chars. Under the old 120-char cap this was discarded for length, which is
    # what deleted #25's only body-level finding and left the headline unsupported.
    verified = _verified(_finding("unclear_scope", [LONG_QUOTE_25]),
                         _normalize(stored[25]), _boundary(_normalize(stored[25])))
    results.append(_check("long_real_quote_is_no_longer_dropped_for_length",
                          len(verified) == 1 and verified[0][2],
                          f"{len(LONG_QUOTE_25)} chars, verified={verified}"))

    # The backfill signal no reading ever quoted. It has to be reachable: real,
    # below the boundary, and not boilerplate - otherwise the definitions in the
    # prompt have nothing to succeed against.
    hay25 = _normalize(stored[25])
    kept = _ground([_finding("backfill", [BACKFILL_QUOTE_25])], stored[25])
    results.append(_check("existing_vacancy_line_is_quotable_body_text",
                          len(kept) == 1 and kept[0].confidence == 0.4,
                          f"phrase at {hay25.find(_normalize(BACKFILL_QUOTE_25))}, "
                          f"boundary at {_boundary(hay25)}, kept={_hypotheses(kept)}"))

    # #26 is the control: its real signal sits close under the boundary, so an
    # over-eager blurb rule would swallow it.
    kept = _ground([_finding("unclear_scope",
                             ["This is not a hand-off role"])], stored[26])
    results.append(_check("control_posting_keeps_its_body_signal", len(kept) == 1,
                          "the blurb rule reached below the boundary on #26"))

    return sum(results), len(results)

def run_model_cases() -> tuple[int, int]:
    """The whole agent on real rows. This is where the hypothesis definitions are
    on trial - everything above holds whatever the model says."""
    results = []
    print("\n🏺 Archaeology model cases...")

    stored = _stored()
    missing = [i for i in STORED_IDS if i not in stored]
    if missing:
        print(f"       ↳ skipped: no stored description for {missing}.")
        return 0, 0

    readings = {i: read_posting(stored[i]) for i in STORED_IDS}

    # The invariant the headline exists to hold, end to end on every stored row.
    broken = []
    for app_id, reading in readings.items():
        named = {name for name, h in HYPOTHESES.items() if h.summary in reading.overall}
        if named != set(_hypotheses(reading.findings)):
            broken.append((app_id, sorted(named), _hypotheses(reading.findings)))
    results.append(_check("overall_matches_findings_on_every_stored_posting", not broken,
                          f"{broken}"))

    # No finding anywhere rests on the blurb alone.
    blurb_only = []
    for app_id, reading in readings.items():
        boundary = _boundary(_normalize(stored[app_id]))
        hay = _normalize(stored[app_id])
        for f in reading.findings:
            if not any(hay.rfind(_normalize(p)) >= boundary for p in f.evidence_phrases):
                blurb_only.append((app_id, f.hypothesis))
    results.append(_check("no_finding_rests_on_the_blurb_alone", not blurb_only, f"{blurb_only}"))

    # Confidence never exceeds what the surviving evidence buys.
    mis_scored = [(i, f.hypothesis, f.confidence) for i, r in readings.items()
                  for f in r.findings if f.confidence > MAX_CONFIDENCE or f.confidence <= 0]
    results.append(_check("confidence_stays_inside_the_ladder", not mis_scored, f"{mis_scored}"))

    # The reported #24 failure: a sentence whose whole purpose is to narrow the role,
    # cited as evidence that the role is not defined.
    cited = [f.hypothesis for f in readings[24].findings
             if any(_normalize(PRECISION_QUOTE_24) in _normalize(p) for p in f.evidence_phrases)]
    results.append(_check("precision_is_not_read_as_unclear_scope",
                          "unclear_scope" not in cited,
                          f"'{PRECISION_QUOTE_24}' cited for {cited}"))

    # The same inversion on #25, where the quoted lines are the most precisely
    # specified duties in the posting.
    results.append(_check("precise_duties_are_not_read_as_unclear_scope",
                          "unclear_scope" not in _hypotheses(readings[25].findings),
                          f"#25 findings: {_hypotheses(readings[25].findings)}"))

    # The definitions' headline test, and the one case here that does not hold.
    # #25 says "This role is an existing vacancy" in plain words, and the real-row
    # case above proves that line is quotable body text - so the gates are not what
    # is dropping it. llama3.1:8b simply does not reliably surface backfill on this
    # posting: it reads the duties, reports understaffed_team, and never revisits a
    # lone administrative line sitting between the nice-to-haves and the hashtags.
    # Rewording the hypothesis to say that one such line is enough, wherever it
    # sits, did not change the answer. This stays failing on purpose. Making it pass
    # would mean matching the phrase in Python, and a hypothesis the code hands to
    # the model is not a hypothesis the model found - the suite would then be
    # asserting on our own keyword rule and would report nothing about whether the
    # definitions work. Expect it to pass on a larger model; if it XPASSes here,
    # drop the marker.
    results.append(_check("backfill_is_found_when_the_posting_states_it",
                          "backfill" in _hypotheses(readings[25].findings),
                          f"#25 findings: {_hypotheses(readings[25].findings)}",
                          xfail="llama3.1:8b does not reliably surface backfill on #25"))

    return sum(results), len(results)

if __name__ == "__main__":
    gp, gt = run_guard_cases()
    rp, rt = run_real_cases()
    mp, mt = run_model_cases()
    print(f"\n📊 {gp + rp + mp}/{gt + rt + mt} passed")
    for known in XFAILED:
        print(f"   known failure: {known}")
