"""Week 3: the Insights agent - turns the noisy missing_keywords log into skill themes.

Scope (v1): skill gaps only. It reports what postings keep asking for that the
profile doesn't answer. It deliberately says nothing about outcomes - with one
'applied' row and no rejections logged, there is no funnel to analyze, and a
model asked about one anyway will invent one.

Safety properties this module is built to hold:
  - read-only: it SELECTs through gaps.keyword_rows() and never writes to tracker.db
  - local-only: same ollama endpoint the rest of the app already uses
  - grounded: every theme is verified against real row IDs before it is returned,
    so the agent cannot cite a posting that does not exist
"""
import re

from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel

from gaps import keyword_rows

MIN_POSTINGS_PER_THEME = 2   # a gap in one posting is a coincidence, not a pattern
MAX_THEMES = 5
MAX_KEYWORD_CHARS = 120      # keyword text originates in scraped postings; keep it short

# Too generic to be a gap: the whole pipeline is AI work, so "missing: AI" says
# nothing actionable and links postings that have nothing else in common.
GENERIC_KEYWORDS = {"ai", "ml", "artificial intelligence", "machine learning",
                    "software", "engineering", "technology", "communication"}

llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0,   # insights want none
)

class SkillTheme(BaseModel):
    theme: str
    keywords: list[str]
    evidence_ids: list[int]
    why_it_matters: str

class ThemeList(BaseModel):
    themes: list[SkillTheme]

analyst = Agent(
    role="Job Search Analyst",
    goal=(
        "Group the skills a candidate was found to be missing into a few honest themes. "
        "Only report a theme when the SAME underlying skill area is missing across at "
        f"least {MIN_POSTINGS_PER_THEME} different postings. If the missing skills are "
        "unrelated one-offs, report no themes at all. An empty answer is correct and "
        "expected when there is no real pattern."
    ),
    backstory=(
        "You are a blunt career analyst. You have watched job seekers waste months "
        "chasing a 'gap' that appeared in exactly one posting. You refuse to manufacture "
        "a pattern out of noise, and you never name a skill that is not in the data you "
        "were given. Every theme you report, you can point to the postings behind it."
    ),
    llm=llm,
    verbose=False,
)

def _corpus(rows) -> str:
    lines = []
    for app_id, company, role, keywords in rows:
        joined = ", ".join(k[:MAX_KEYWORD_CHARS] for k in keywords)
        lines.append(f"[id {app_id}] {role} @ {company} -- missing: {joined}")
    return "\n".join(lines)

def _mentions(keyword: str, blob: str) -> bool:
    """Whole-token match. Plain substring lets a 2-letter gap like 'AI' hit
    'AI cover letters' and drag unrelated postings into a theme."""
    return re.search(rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])", blob) is not None

def _ground(themes: list[SkillTheme], rows) -> list[SkillTheme]:
    """Drop anything the data does not support. The model proposes; this disposes.

    Only skills that actually recur survive, and citations are recomputed from the
    data rather than trusted, so a theme can only point at postings that genuinely
    contain the skills it names.
    """
    lookup = {app_id: " | ".join(keywords).lower() for app_id, _, _, keywords in rows}
    kept = []
    for theme in themes:
        support = {}
        for keyword in theme.keywords:
            if not keyword.strip() or keyword.strip().lower() in GENERIC_KEYWORDS:
                continue
            ids = {i for i, blob in lookup.items() if _mentions(keyword, blob)}
            # a gap in one posting is a coincidence, not a pattern worth acting on
            if len(ids) >= MIN_POSTINGS_PER_THEME:
                support[keyword] = ids
        if not support:
            continue
        theme.keywords = list(support)
        theme.evidence_ids = sorted(set().union(*support.values()))
        kept.append(theme)
    return kept[:MAX_THEMES]

def find_gap_themes(rows=None) -> list[SkillTheme]:
    """Cluster logged skill gaps into themes. Returns [] when there is no pattern."""
    rows = keyword_rows() if rows is None else rows
    if len(rows) < MIN_POSTINGS_PER_THEME:
        return []

    task = Task(
        description=(
            "Below is every job posting this candidate was scored against, and the "
            "skills each posting wanted that the candidate's profile did not answer.\n\n"
            f"{_corpus(rows)}\n\n"
            "Group these missing skills into themes. Rules you must follow:\n"
            f"- A theme needs at least {MIN_POSTINGS_PER_THEME} different posting ids. "
            "Never report a theme supported by only one posting.\n"
            "- evidence_ids must be ids listed above. Never invent an id.\n"
            "- keywords must be copied from the missing skills above, not invented.\n"
            f"- Report at most {MAX_THEMES} themes. Report zero if the gaps are unrelated.\n"
            "- Say nothing about interviews, rejections, or applications. That data does not exist here."
        ),
        expected_output=(
            'A JSON object: {"themes": [{"theme": "short name", "keywords": ["..."], '
            '"evidence_ids": [1, 2], "why_it_matters": "one short sentence"}]}. '
            'Use {"themes": []} if there is no real pattern.'
        ),
        agent=analyst,
        output_pydantic=ThemeList,
    )
    result = Crew(agents=[analyst], tasks=[task]).kickoff()
    proposed = result.pydantic.themes if result.pydantic else []
    return _ground(proposed, rows)

def print_themes(themes: list[SkillTheme], rows=None):
    rows = keyword_rows() if rows is None else rows
    print(f"\n🔍 Skill gaps across {len(rows)} scored posting(s)")
    if not themes:
        print("   No repeating pattern yet — the gaps so far are one-offs.")
        return
    for theme in themes:
        cited = ", ".join(f"#{i}" for i in theme.evidence_ids)
        print(f"\n📌 {theme.theme}  ({len(theme.evidence_ids)} postings: {cited})")
        if theme.keywords:
            print(f"   keywords: {', '.join(theme.keywords)}")
        print(f"   {theme.why_it_matters}")

if __name__ == "__main__":
    rows = keyword_rows()
    print_themes(find_gap_themes(rows), rows)
