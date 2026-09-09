"""Week 4: the Archaeology agent - reads what a posting accidentally says about
the team behind it.

Scope (v1): one posting, read in isolation. It reports hypotheses about the
organizational reality behind the text - is this a backfill, a compliance
posting for a role already promised, a team that is understaffed - and cites the
exact phrases that led it there.

Deliberately out of scope: repost detection, cross-company patterns, and any
correlation with outcomes. Those need posting history and rejection data that
tracker.db does not carry yet, and a model asked about them will invent them.

Division of labour, as in insights_agent: the model proposes a hypothesis and
quotes the posting for it. Everything checkable is decided here afterwards -
which quotes are real, which part of the posting they came from, how confident
the finding is allowed to be, and what the headline says. The model's surviving
contribution is the hypothesis, the quotes, and one sentence of implication.

Safety properties this module is built to hold:
  - read-only: it SELECTs through tracker.list_applications() and never writes
  - local-only: same ollama endpoint the rest of the app already uses
  - grounded: every evidence phrase is checked against the real posting text
    before it is returned, so the agent cannot quote a line that was never there
  - about the team, not the company: a hypothesis has to rest on at least one
    phrase from the part of the posting that describes THIS role. The blurb at
    the top is written once and pasted into every req the company will ever
    publish, so it cannot be evidence about one team
  - never certain: confidence is computed from surviving evidence and capped,
    because none of these hypotheses can be verified - the company never tells
    you why the posting existed
  - the headline cannot outrun the evidence: `overall` is composed here from the
    findings that survived grounding, so it can never name a hypothesis that the
    reader will not find quoted below it
"""
import os
import re
import sys
from typing import Literal, NamedTuple

from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel

from tracker import list_applications

MAX_FINDINGS = 4
MAX_CONFIDENCE = 0.8          # unverifiable by construction, so never certain
MIN_DESCRIPTION_LEN = 400     # below this there is not enough text to read

# A cap on a single quote, not a filter on evidence. At 120 it was doing the
# latter: five of the eight phrases dropped across the three stored postings were
# real, verbatim quotes thrown out for length alone, and on the Camunda row it
# silently deleted an entire finding whose two quotes ran to 147 and 225 chars.
# 300 still catches a model that pastes a whole section back at us.
MAX_PHRASE_CHARS = 300

# Confidence by surviving body-phrase count. One phrase can suggest, two can
# establish; nothing here can ever be verified, so the ladder stops at
# MAX_CONFIDENCE. See _confidence().
CONFIDENCE_BY_PHRASES = {1: 0.4, 2: 0.6}

# Where the company blurb ends and the role begins. Matched only in heading
# position (see _is_heading) - "the opportunity" occurs mid-sentence in the
# RoomPriceGenie blurb, and treating that as a section break would put the
# boundary inside the very text it exists to fence off.
SECTION_MARKERS = (
    "your role", "about the role", "about this role", "the role:", "role overview",
    "job overview", "job description", "position summary", "your mission",
    "the opportunity", "what you'll do", "what you will do", "what you'll be doing",
    "what you will be doing", "what you'll own", "who you are", "responsibilities",
    "key responsibilities", "what we're looking for", "what we are looking for",
)

# A marker found this late is not a section break, it is a word in a sentence.
# Past it the function fails open and treats the whole posting as body text -
# never the other way round, because failing closed would drop every finding.
MAX_BOUNDARY_FRACTION = 0.6

# Too generic to be evidence: these fragments turn up in nearly every posting, so
# quoting one says nothing about this team and would support any hypothesis at all.
BOILERPLATE = {"growing fast", "fast-paced", "top talent", "wear many hats",
               "join our team", "hit the ground running", "dynamic environment",
               "self-starter", "team player", "cutting edge", "world-class",
               "competitive salary", "make an impact", "exciting opportunity",
               "rockstar", "passionate"}

NOTHING_FOUND = "Nothing this posting can be quoted on - most postings are unremarkable."

class Hypothesis(NamedTuple):
    """What one hypothesis means, and what does and does not count as evidence.

    All four fields reach the model. Until they did, the six hypotheses arrived as
    bare snake_case names with their definitions sitting in comments the model
    never saw, so it matched on tone: three of the six were never proposed at all
    across every stored posting, including on a req that says in plain words
    "This role is an existing vacancy", and `unclear_scope` was repeatedly
    supported by sentences that were the posting being precise. `not_this` exists
    because naming the near-miss is what stops that - a definition tells the model
    what to look for, only a counter-example tells it what to stop looking at.

    `summary` is our own sentence for this hypothesis, and it is the only prose
    that reaches `overall`. See _compose_overall().
    """
    means: str
    looks_like: str
    not_this: str
    summary: str

HYPOTHESES = {
    "backfill": Hypothesis(
        means="someone left and they need a replacement",
        looks_like="the posting treats the role as one that already exists - an "
                   "existing vacancy, a backfill, replacing or taking over from "
                   "someone, a handover, or duties described as work already being "
                   "done that this person will continue",
        not_this="a company hiring many people at once. Expansion is not replacement",
        summary="it reads like a replacement for someone who left",
    ),
    "compliance_posting": Hypothesis(
        means="the role is already promised to a particular person and the posting "
              "exists because a process requires the job to be advertised",
        looks_like="requirements so exact they describe one individual, an unusually "
                   "short or already-closing window, a statement that internal "
                   "candidates are preferred or that a candidate has been identified",
        not_this="a long or demanding requirements list. Demanding is not the same as "
                 "written around one named person",
        summary="it reads like a formality for a role already spoken for",
    ),
    "likely_ghost": Hypothesis(
        means="there is no real budget or urgency behind it and it may never be filled",
        looks_like="no named team, no concrete duties, no start date, an evergreen or "
                   "always-open framing, talent-pool or pipeline language, or a req "
                   "that describes the company at length and the job barely at all",
        not_this="an enthusiastic or marketing-heavy posting. A company that writes "
                 "breathlessly about itself is not thereby short of budget",
        summary="there is no real urgency behind it",
    ),
    "growth_hire": Hypothesis(
        means="genuinely new headcount on a team that is functioning",
        looks_like="the posting says this team or function is new or expanding, calls "
                   "the headcount additional, or describes work that did not exist "
                   "before and is now someone's job",
        not_this="the company describing its own growth, funding, awards or customer "
                 "numbers. That is the blurb talking about the company, and it says "
                 "nothing about whether THIS team gained a head",
        summary="it reads like new headcount on a team that is functioning",
    ),
    "understaffed_team": Hypothesis(
        means="one hire is expected to cover work that is really several jobs",
        looks_like="duties spanning roles normally held by different people - building "
                   "and running and supporting and selling - sole ownership of a whole "
                   "system, on-call named alongside full feature delivery, or the "
                   "person being the first or only one doing something",
        not_this="a long list of duties that all belong to one discipline. A thorough "
                 "description of one job is not a description of two",
        summary="one hire is expected to cover several jobs",
    ),
    "unclear_scope": Hypothesis(
        means="they have not decided what this role is",
        looks_like="vague verbs with no object, contradictory seniority signals, a "
                   "title that does not match the duties, outcomes named with no work "
                   "named, or the posting hedging about what the person will do",
        not_this="a long list of specific, concrete duties - precision about many "
                 "tasks is understaffed_team at most. A sentence that rules work OUT "
                 "(\"your job isn't to build X\") or fixes a level or a stack is the "
                 "posting being CLEAR. Never quote a precise sentence as unclear scope",
        summary="they have not settled what this role is",
    ),
}

# Fixed tie-break order for contested evidence, and the guard that keeps the
# Literal on Finding.hypothesis and the dict above from drifting apart.
HYPOTHESIS_ORDER = {name: i for i, name in enumerate(HYPOTHESES)}

COUNT_WORDS = {1: "One thing", 2: "Two things", 3: "Three things", 4: "Four things"}

llm = LLM(
    model="ollama/llama3.1:8b",
    base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    temperature=0,   # inference over evidence, not creativity
)

class Finding(BaseModel):
    hypothesis: Literal[
        "backfill", "compliance_posting", "likely_ghost",
        "growth_hire", "understaffed_team", "unclear_scope",
    ]
    evidence_phrases: list[str]
    implication: str
    # Recomputed in _ground() from the phrases that survive. The model is told not
    # to supply this; the default exists so a model that supplies it anyway still
    # parses, exactly as with SkillTheme.evidence_ids in insights_agent.
    confidence: float = 0.0

class PostingReading(BaseModel):
    findings: list[Finding]
    # Composed in read_posting() from the grounded findings. Same deal as
    # Finding.confidence: the model is told not to write one, and anything it
    # writes anyway is overwritten rather than trusted.
    overall: str = ""

archaeologist = Agent(
    role="Job posting archaeologist",
    goal=("Infer what a posting reveals about the team that wrote it, using only "
          "phrases that actually appear in it."),
    backstory=("You have read thousands of job postings and know that the wording "
               "leaks things the company did not mean to say. You also know the "
               "difference between a posting that is vague and a posting that is "
               "demanding, and between a company boasting about itself and a team "
               "giving something away. You never guess beyond the text in front "
               "of you."),
    llm=llm,
    verbose=False,
)

def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase - scraped postings have ragged spacing,
    and a phrase that differs only by a newline is still the same phrase."""
    return re.sub(r"\s+", " ", text).strip().lower()

# --- where the blurb ends ---------------------------------------------------

def _is_heading(haystack: str, start: int, end: int) -> bool:
    """Is this marker a section heading, or just those words inside a sentence?

    Two ways to qualify, because scraped postings arrive as one blob with the
    newlines already gone and the heading's own capitalisation lowercased away:
    the marker follows a sentence end, or it is immediately followed by a colon.
    "Job Overview:" in the ThriveCart row qualifies on the colon (it trails a
    close-paren); "Your Role" in the RoomPriceGenie row qualifies on the full
    stop before it - and "the opportunity to build high-value internal AI
    systems", in that same row's blurb, qualifies on neither.
    """
    if start == 0:
        return True
    before = haystack[max(0, start - 2):start].strip()
    return (before[-1:] in {".", "!", "?", ":", ")", ";"}) or haystack[end:end + 1] == ":"

def _boundary(haystack: str) -> int:
    """Offset in the normalized text where the company blurb ends.

    Returns 0 when no marker is found in heading position within the first
    MAX_BOUNDARY_FRACTION of the text. 0 means "everything is body text", so an
    undetectable boundary costs nothing - the corroboration rule in _ground()
    simply stops biting. Failing the other way would drop every finding on any
    posting whose section headings we do not recognise.
    """
    limit = len(haystack) * MAX_BOUNDARY_FRACTION
    found = []
    for marker in SECTION_MARKERS:
        for match in re.finditer(re.escape(marker), haystack):
            if match.start() > limit:
                break
            if _is_heading(haystack, match.start(), match.end()):
                found.append(match.start())
                break
    return min(found, default=0)

# --- grounding --------------------------------------------------------------

def _confidence(body_phrases: int) -> float:
    """Confidence from surviving evidence, never from what the model claimed.

    The model's own number used only to be clamped, which meant anything it
    reported below the ceiling passed straight through to the UI - a finding that
    went in at 0.4 came out at 0.4, rating evidence the model had chosen before
    grounding took half of it away.

    Only body phrases count. A blurb quote can corroborate a finding but cannot
    establish one, and a phrase that cannot establish a finding has no business
    raising its confidence either.
    """
    return CONFIDENCE_BY_PHRASES.get(body_phrases, MAX_CONFIDENCE)

def _verified(finding: Finding, haystack: str, boundary: int) -> list[tuple[str, str, bool]]:
    """(phrase, normalized, is_body) for every phrase that clears the fixed gates.

    Fixed meaning: everything except exclusivity, which depends on what other
    findings have already taken and so cannot be judged one finding at a time.
    Splitting it out this way is what lets _strength() rank findings by evidence
    they actually have, before any of it has been claimed.

    A phrase counts as body text if it occurs below the boundary ANYWHERE - rfind,
    not find - so a sentence the blurb and the role section both use is credited
    to the role section.
    """
    checked = []
    for phrase in finding.evidence_phrases:
        if not phrase or len(phrase) > MAX_PHRASE_CHARS:
            continue
        normalized = _normalize(phrase)
        if not normalized or normalized not in haystack:
            continue
        if any(fragment in normalized for fragment in BOILERPLATE):
            continue
        checked.append((phrase, normalized, haystack.rfind(normalized) >= boundary))
    return checked

def _strength(item: tuple[Finding, list[tuple[str, str, bool]]]) -> tuple[int, int, int]:
    """Strongest first: most body evidence, then most evidence, then declared order.

    This used to be the model's own confidence, which was incoherent in two
    directions at once - the number was not trusted enough to reach the user, but
    was trusted to decide which of two contradictory hypotheses owned a contested
    line. On the Camunda row that is precisely what happened: growth_hire (0.8)
    and understaffed_team (0.4) submitted the same two quotes, and the model's
    unverified self-rating settled it. Ranking on verified evidence, with a fixed
    declaration order to break ties, makes the outcome a fact about the posting.
    """
    finding, phrases = item
    body = sum(1 for _, _, is_body in phrases if is_body)
    return (-body, -len(phrases), HYPOTHESIS_ORDER.get(finding.hypothesis, len(HYPOTHESES)))

def _ground(findings: list[Finding], description: str) -> list[Finding]:
    """Keep only what the posting actually supports.

    Every evidence phrase must appear verbatim in the description and must say
    something a posting would not say by default. A finding whose phrases were all
    invented, or all boilerplate, is dropped entirely - not softened, dropped,
    because a hypothesis with no evidence is just the model's prior.

    A finding must also keep at least one phrase from below the blurb boundary.
    Header phrases survive alongside it as corroboration - the softer rule, chosen
    over excluding the blurb outright because three stored postings are too thin a
    corpus to justify suppressing a header phrase that might be the real signal.
    What it stops is a hypothesis resting on the blurb ALONE, which was every
    finding on the RoomPriceGenie row.

    Evidence is also exclusive: findings are read strongest-first, and a phrase one
    finding has claimed cannot support a later one. Two contradictory hypotheses
    resting on the same line means the model is guessing, not reading. A finding
    that is then dropped releases nothing, because it never claimed anything - its
    phrases are only marked taken once the finding is known to be kept.

    Confidence is recomputed here from the phrases that survive.
    """
    haystack = _normalize(description)
    boundary = _boundary(haystack)
    ranked = sorted(((f, _verified(f, haystack, boundary)) for f in findings), key=_strength)

    claimed: set[str] = set()
    kept: list[Finding] = []
    for finding, phrases in ranked:
        real, seen, body = [], set(), 0
        for phrase, normalized, is_body in phrases:
            if normalized in claimed or normalized in seen:
                continue
            seen.add(normalized)
            real.append(phrase)
            body += is_body
        if not body:
            continue          # invented, boilerplate, taken, or blurb-only
        claimed |= seen
        finding.evidence_phrases = real
        finding.confidence = _confidence(body)
        kept.append(finding)
    return kept[:MAX_FINDINGS]

# --- the headline -----------------------------------------------------------

def _compose_overall(findings: list[Finding]) -> str:
    """Write the headline from the findings that survived. No model prose involved.

    The model used to write `overall` in the same breath as the findings, which
    made it a description of the set it HOPED to report: on the Camunda row it
    announced a team that "may be understaffed and have unclear role
    responsibilities" above a reading in which neither hypothesis had survived
    grounding. Composing it from the kept findings is the ask_agent SPAN_CATEGORIES
    move - our own sentences, selected by verified evidence - and it makes that
    contradiction unrepresentable rather than merely unlikely.

    What it gives up is the model's ability to notice something across findings
    that none of them states alone. If that turns out to be missed, the upgrade is
    a second call shown only the survivors, not letting this one write ahead of
    its evidence again.
    """
    clauses, seen = [], set()
    for finding in findings:
        if finding.hypothesis in seen or finding.hypothesis not in HYPOTHESES:
            continue
        seen.add(finding.hypothesis)
        clauses.append(HYPOTHESES[finding.hypothesis].summary)
    if not clauses:
        return NOTHING_FOUND
    if len(clauses) == 1:
        body = clauses[0]
    else:
        body = ", ".join(clauses[:-1]) + ", and " + clauses[-1]
    return f"{COUNT_WORDS.get(len(clauses), 'Several things')} this posting gives away: {body}."

# --- the one model call -----------------------------------------------------

def _hypothesis_brief() -> str:
    """Render the hypotheses for the prompt: name, meaning, evidence, counter-example."""
    lines = []
    for name, h in HYPOTHESES.items():
        lines.append(f"- {name}: {h.means}.\n"
                     f"    evidence looks like: {h.looks_like}.\n"
                     f"    NOT this: {h.not_this}.")
    return "\n".join(lines)

def read_posting(description: str, role: str = "", company: str = "") -> PostingReading:
    """Read one posting. Returns an empty findings list when nothing is supported."""
    if not description or len(description) < MIN_DESCRIPTION_LEN:
        return PostingReading(findings=[], overall="Not enough posting text to read.")

    task = Task(
        description=(
            f"Below is the full text of a job posting for {role or 'a role'} at "
            f"{company or 'a company'}.\n\n"
            f"---\n{description}\n---\n\n"
            "Infer what this posting reveals about the TEAM behind it. These are the "
            "only hypotheses you may report, with what each one means and what it is "
            "not:\n\n"
            f"{_hypothesis_brief()}\n\n"
            "Rules you must follow:\n"
            "- Read the NOT this line for a hypothesis before you report it. If your "
            "evidence is the thing it describes, report nothing.\n"
            "- evidence_phrases must be copied EXACTLY from the posting above, "
            "word for word. Never paraphrase, never invent a phrase.\n"
            "- Most postings open with a block about the company - what it sells, how "
            "fast it is growing, its awards and customers. That block is written once "
            "and reused for every job the company posts, so it reveals nothing about "
            "THIS team. Quote it only to support a phrase from the part of the posting "
            "that describes the role itself; a hypothesis resting on it alone will be "
            "discarded.\n"
            "- If you cannot quote the posting for a hypothesis, do not report it.\n"
            f"- Report at most {MAX_FINDINGS} findings. Report zero if the posting "
            "is unremarkable - most postings are.\n"
            "- Do not output confidence, and do not write an overall summary. Both are "
            "recomputed from the evidence after you answer.\n"
            "- Say nothing about salary, the candidate, or whether to apply."
        ),
        expected_output=(
            'A JSON object: {"findings": [{"hypothesis": "backfill", '
            '"evidence_phrases": ["exact text from the posting"], "implication": '
            '"one short sentence"}]}. '
            'Use {"findings": []} if the posting reveals nothing.'
        ),
        agent=archaeologist,
        output_pydantic=PostingReading,
    )
    result = Crew(agents=[archaeologist], tasks=[task]).kickoff()
    reading = result.pydantic if result.pydantic else PostingReading(findings=[])
    reading.findings = _ground(reading.findings, description)
    reading.overall = _compose_overall(reading.findings)
    return reading

def print_reading(reading: PostingReading, role: str = "", company: str = ""):
    print(f"\n🏺 Archaeology: {role or 'role'} @ {company or 'company'}")
    if not reading.findings:
        # overall already carries NOTHING_FOUND here, so there is one empty-state
        # sentence rather than two that have to be kept in agreement.
        print(f"   {reading.overall or NOTHING_FOUND}")
        return
    for f in reading.findings:
        print(f"\n📌 {f.hypothesis}  (confidence {f.confidence})")
        for phrase in f.evidence_phrases:
            print(f'   evidence: "{phrase}"')
        print(f"   {f.implication}")
    if reading.overall:
        print(f"\n   overall: {reading.overall}")

if __name__ == "__main__":
    app_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    apps = list_applications()
    if app_id:
        rows = [a for a in apps if a["id"] == app_id]
    else:
        rows = [a for a in apps if a.get("description")][:1]
    if not rows:
        print("No posting with stored description found. Run scout_run.py first.")
        raise SystemExit(1)
    row = rows[0]
    reading = read_posting(row["description"], row["role"], row["company"])
    print_reading(reading, row["role"], row["company"])
