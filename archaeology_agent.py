"""Week 4: the Archaeology agent - reads what a posting accidentally says about
the team behind it.

Scope (v1): one posting, read in isolation. It reports hypotheses about the
organizational reality behind the text - is this a backfill, a compliance
posting for a role already promised, a team that is understaffed - and cites the
exact phrases that led it there.

Deliberately out of scope: repost detection, cross-company patterns, and any
correlation with outcomes. Those need posting history and rejection data that
tracker.db does not carry yet, and a model asked about them will invent them.

Safety properties this module is built to hold:
  - read-only: it SELECTs through tracker.list_applications() and never writes
  - local-only: same ollama endpoint the rest of the app already uses
  - grounded: every evidence phrase is checked against the real posting text
    before it is returned, so the agent cannot quote a line that was never there
  - never certain: confidence is capped, because none of these hypotheses can
    be verified - the company never tells you why the posting existed
"""
import re
import sys
from typing import Literal

from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel

from tracker import list_applications

MAX_FINDINGS = 4
MAX_CONFIDENCE = 0.8          # unverifiable by construction, so never certain
MIN_PHRASES_FOR_HIGH = 2      # one phrase is a coincidence, not a pattern
HIGH_CONFIDENCE = 0.6         # ceiling for a single-phrase finding
MIN_DESCRIPTION_LEN = 400     # below this there is not enough text to read
MAX_PHRASE_CHARS = 120        # phrases come from scraped postings; keep them short

# Too generic to be evidence: these fragments turn up in nearly every posting, so
# quoting one says nothing about this team and would support any hypothesis at all.
BOILERPLATE = {"growing fast", "fast-paced", "top talent", "wear many hats",
               "join our team", "hit the ground running", "dynamic environment",
               "self-starter", "team player", "cutting edge", "world-class",
               "competitive salary", "make an impact", "exciting opportunity",
               "rockstar", "passionate"}

HYPOTHESES = [
    "backfill",             # someone left and they need a replacement
    "compliance_posting",   # role already promised; posting exists for process
    "likely_ghost",         # no real budget or urgency behind it
    "growth_hire",          # genuinely new headcount on a healthy team
    "understaffed_team",    # one hire expected to cover several jobs
    "unclear_scope",        # they have not decided what this role is
]

llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0,   # inference over evidence, not creativity
)

class Finding(BaseModel):
    hypothesis: Literal[
        "backfill", "compliance_posting", "likely_ghost",
        "growth_hire", "understaffed_team", "unclear_scope",
    ]
    confidence: float
    evidence_phrases: list[str]
    implication: str

class PostingReading(BaseModel):
    findings: list[Finding]
    overall: str

archaeologist = Agent(
    role="Job posting archaeologist",
    goal=("Infer what a posting reveals about the team that wrote it, using only "
          "phrases that actually appear in it."),
    backstory=("You have read thousands of job postings and know that the wording "
               "leaks things the company did not mean to say. You never guess "
               "beyond the text in front of you."),
    llm=llm,
    verbose=False,
)

def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase - scraped postings have ragged spacing,
    and a phrase that differs only by a newline is still the same phrase."""
    return re.sub(r"\s+", " ", text).strip().lower()

def _ground(findings: list[Finding], description: str) -> list[Finding]:
    """Keep only what the posting actually supports.

    Every evidence phrase must appear verbatim in the description and must say
    something a posting would not say by default. A finding whose phrases were all
    invented, or all boilerplate, is dropped entirely - not softened, dropped,
    because a hypothesis with no evidence is just the model's prior.

    Evidence is also exclusive: findings are read strongest-first, and a phrase one
    finding has claimed cannot support a later one. Two contradictory hypotheses
    resting on the same line means the model is guessing, not reading.

    Confidence is recomputed from the phrases that survive here, never from what the
    model claimed for the phrases it hoped to use.
    """
    haystack = _normalize(description)
    claimed = set()
    kept = []
    for finding in sorted(findings, key=lambda f: f.confidence, reverse=True):
        real = []
        for phrase in finding.evidence_phrases:
            if not phrase or len(phrase) > MAX_PHRASE_CHARS:
                continue
            normalized = _normalize(phrase)
            if normalized not in haystack or normalized in claimed:
                continue
            if any(fragment in normalized for fragment in BOILERPLATE):
                continue
            real.append(phrase)
            claimed.add(normalized)
        if not real:
            continue
        finding.evidence_phrases = real
        # a single phrase can suggest, not establish
        ceiling = MAX_CONFIDENCE if len(real) >= MIN_PHRASES_FOR_HIGH else HIGH_CONFIDENCE
        finding.confidence = round(min(max(finding.confidence, 0.0), ceiling), 2)
        kept.append(finding)
    return kept[:MAX_FINDINGS]

def read_posting(description: str, role: str = "", company: str = "") -> PostingReading:
    """Read one posting. Returns an empty findings list when nothing is supported."""
    if not description or len(description) < MIN_DESCRIPTION_LEN:
        return PostingReading(findings=[], overall="Not enough posting text to read.")

    task = Task(
        description=(
            f"Below is the full text of a job posting for {role or 'a role'} at "
            f"{company or 'a company'}.\n\n"
            f"---\n{description}\n---\n\n"
            "Infer what this posting reveals about the team behind it. Rules you must follow:\n"
            f"- Choose hypotheses only from this list: {', '.join(HYPOTHESES)}.\n"
            "- evidence_phrases must be copied EXACTLY from the posting above, "
            "word for word. Never paraphrase, never invent a phrase.\n"
            "- If you cannot quote the posting for a hypothesis, do not report it.\n"
            f"- Report at most {MAX_FINDINGS} findings. Report zero if the posting "
            "is unremarkable - most postings are.\n"
            "- confidence is 0.0 to 1.0. You can never verify these guesses, so "
            "never claim certainty.\n"
            "- Say nothing about salary, the candidate, or whether to apply."
        ),
        expected_output=(
            'A JSON object: {"findings": [{"hypothesis": "backfill", "confidence": 0.6, '
            '"evidence_phrases": ["exact text from the posting"], "implication": '
            '"one short sentence"}], "overall": "one or two sentences"}. '
            'Use {"findings": [], "overall": "..."} if the posting reveals nothing.'
        ),
        agent=archaeologist,
        output_pydantic=PostingReading,
    )
    result = Crew(agents=[archaeologist], tasks=[task]).kickoff()
    reading = result.pydantic if result.pydantic else PostingReading(findings=[], overall="")
    reading.findings = _ground(reading.findings, description)
    return reading

def print_reading(reading: PostingReading, role: str = "", company: str = ""):
    print(f"\n🏺 Archaeology: {role or 'role'} @ {company or 'company'}")
    if not reading.findings:
        print("   Nothing the posting will admit to.")
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
