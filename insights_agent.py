"""Week 3: the Insights agent - turns the noisy missing_keywords log into skill themes.

Scope (v1): skill gaps only. It reports what postings keep asking for that the
profile doesn't answer. It deliberately says nothing about outcomes - with one
'applied' row and no rejections logged, there is no funnel to analyze, and a
model asked about one anyway will invent one.

Division of labour: Python finds the candidates, the model names them.

  Handing llama3.1:8b 180-odd raw keyword strings and asking it to spot the
  recurring ones does not work - it cannot count across a list that long, so it
  invents groupings and then labels them confidently. Everything countable is
  therefore done here: normalization, in-posting dedupe, the >=2-postings
  threshold, and every evidence id. The model receives the finished shortlist
  (17 terms on the current corpus) and does the one thing it is actually good
  at - naming a group and saying why it matters.

Safety properties this module is built to hold:
  - read-only: it SELECTs through gaps.keyword_rows() and never writes to tracker.db
  - local-only: same ollama endpoint the rest of the app already uses
  - grounded: terms, groupings and evidence ids are all re-derived from the data
    after the model answers, so the agent cannot cite a posting that does not
    exist and cannot file a posting under a theme it does not support
"""
import os
import re
from collections import defaultdict

from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel

from gaps import keyword_rows

MIN_POSTINGS_PER_THEME = 2   # a gap in one posting is a coincidence, not a pattern
MAX_THEMES = 5
MAX_KEYWORD_CHARS = 120      # keyword text originates in scraped postings; keep it short

# Terms that recur across every domain and so link postings with nothing else in
# common. These are the bridges that produced an "AI and Machine Learning" theme
# containing a JVM backend role: "AI" reaches seven unrelated postings and "API"
# is the only thing a Camunda REST role shares with an LLM role. "testing",
# "cloud" and "experience" never surface as bare terms in the current corpus -
# they live inside longer phrases - but they are the same kind of word and are
# blocked pre-emptively.
#
# Deliberately NOT blocked: monitoring, observability. They are cross-domain too,
# but they are also the only ops signal in the corpus, and the two gates in
# _ground() already stop them from contaminating an unrelated theme. Blocked
# terms are dropped before grouping, so the model never sees them.
NON_DISCRIMINATIVE_TERMS = {
    "ai", "api", "testing", "cloud", "experience",
    "ml", "artificial intelligence", "machine learning",
    "software", "engineering", "technology", "communication",
}

llm = LLM(
    model="ollama/llama3.1:8b",
    base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    temperature=0,   # insights want none
)

class SkillTheme(BaseModel):
    theme: str
    keywords: list[str]
    # Recomputed in _ground() from the data. The model is told not to supply
    # these; the default exists so a model that supplies them anyway still parses.
    evidence_ids: list[int] = []
    why_it_matters: str = ""

class ThemeList(BaseModel):
    themes: list[SkillTheme]

analyst = Agent(
    role="Job Search Analyst",
    goal=(
        "Give a short list of already-verified recurring skill gaps an honest set of "
        "names. The list has been computed from the data before you see it: every term "
        f"on it is known to appear in at least {MIN_POSTINGS_PER_THEME} different "
        "postings. You are not searching for patterns, you are naming the ones found. "
        "Terms that do not belong together stay apart, and an empty answer is correct "
        "when nothing on the list groups."
    ),
    backstory=(
        "You are a blunt career analyst. You have watched job seekers waste months "
        "chasing a 'gap' that appeared in exactly one posting. You refuse to manufacture "
        "a pattern out of noise, and you never name a skill that is not in the data you "
        "were given. You name a group only for what is actually in it - you would rather "
        "leave a term ungrouped than stretch a label to cover it."
    ),
    llm=llm,
    verbose=False,
)

# --- term extraction (deterministic, no model) ------------------------------
# Everything from here to recurring_terms() is plain counting. It is the part the
# model used to be asked to do by eye.

_PUNCT = " \t\r\n.,;:!?-–—/\\|\"'`()[]{}*"
_PAREN = re.compile(r"\(([^)]*)\)")
_COORD = re.compile(r"\s*(?:,|&|\+|\band\b|\bor\b|\bund\b|\boder\b)\s*")
_LIKE = re.compile(r"\s+(?:like|such as|including|incl\.?|e\.g\.?|z\.b\.?)\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(_PUNCT)

def _tech_tokens(raw: str) -> set[str]:
    """Tokens whose capitalization marks them as a self-contained technical name.

    Two or more capitals is the signal: RAG, OWASP, MLOps, LangChain, ChromaDB.
    It is what lets "RAG pipelines", "RAG-Architekturen" and a bare "RAG" count as
    the same gap without stripping head nouns - head-noun stripping is what would
    turn "penetration testing methodologies" into a bare "testing" and bridge a
    security posting to an AI one.
    """
    tokens = set()
    for token in _WORD.findall(raw):
        if len(token) < 2 or sum(c.isupper() for c in token) < 2:
            continue
        if len(token) > 2 and token.endswith("s") and token[:-1].isupper():
            token = token[:-1]        # APIs -> API, LLMs -> LLM; never MLOps -> MLOp
        tokens.add(token.lower())
    return tokens

def _variants(raw: str) -> set[str]:
    """Every normalized term one keyword entry asserts.

    A scraped keyword is free text and frequently packs several gaps into one
    string - "cloud technologies (AWS, Azure or GCP)" is three - so the entry is
    split on the punctuation that means enumeration, and each piece counts.
    """
    found, pending, seen = set(), [raw], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)

        inner = _PAREN.findall(current)
        if inner:
            pending.append(_PAREN.sub(" ", current))   # "vector databases (ChromaDB)"
            pending.extend(inner)                      #   -> both halves
        for splitter in (_LIKE, _COORD):
            parts = splitter.split(current)
            if len(parts) > 1:
                pending.extend(parts)
        if "/" in current and all(len(p.strip()) >= 4 for p in current.split("/")):
            pending.extend(current.split("/"))         # LangChain/LangGraph, but not CI/CD

        normalized = _norm(current)
        if normalized:
            found.add(normalized)
    return {term for term in found | _tech_tokens(raw) if len(term) >= 2}

def _term_index(rows) -> dict[str, set[int]]:
    """term -> the DISTINCT posting ids asserting it.

    A set of ids, not a count, is the whole fix for the inflated raw counts. Row
    #13 lists "LangChain", "LangGraph" and "orchestration frameworks like
    LangChain/LangGraph" as three separate entries and row #15 lists LangGraph
    twice; both collapse to one vote per posting here, because the posting is the
    unit a threshold about postings has to be measured in.
    """
    index = defaultdict(set)
    for app_id, _, _, keywords in rows:
        for raw in keywords:
            for term in _variants(raw):
                index[term].add(app_id)
    return index

def recurring_terms(rows) -> dict[str, set[int]]:
    """The shortlist: terms in >=MIN_POSTINGS_PER_THEME postings, bridges removed.

    This is the agent's entire candidate vocabulary. Nothing outside it can reach
    a theme, whatever the model says.
    """
    return {term: ids for term, ids in sorted(_term_index(rows).items())
            if len(ids) >= MIN_POSTINGS_PER_THEME and term not in NON_DISCRIMINATIVE_TERMS}

def _term_corpus(terms: dict[str, set[int]]) -> str:
    """Render the shortlist for the prompt, commonest first, ties alphabetical."""
    lines = []
    for term, ids in sorted(terms.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cited = ", ".join(f"#{i}" for i in sorted(ids))
        lines.append(f"- {term[:MAX_KEYWORD_CHARS]}  ({len(ids)} postings: {cited})")
    return "\n".join(lines)

# --- grounding --------------------------------------------------------------

def _evidence_threshold(term_count: int) -> int:
    """How many of a theme's terms a posting must contain to count as evidence.

    Two, so that a posting joins a theme by exhibiting the skill area rather than
    by sharing one word with it. One word is what let "observability" alone file a
    Camunda JVM backend role under an AI theme.

    A two-term theme is the exception and needs only one: the connectivity gate
    below has already proved those two terms co-occur in a real posting, so
    demanding both again would shrink the theme to that intersection and throw
    away the postings that motivated it.
    """
    return 1 if term_count <= 2 else 2

def _evidence(selected: dict[str, set[int]]) -> set[int]:
    """The postings that support the theme - NOT the union of its terms' postings."""
    needed = _evidence_threshold(len(selected))
    hits = defaultdict(int)
    for ids in selected.values():
        for app_id in ids:
            hits[app_id] += 1
    return {app_id for app_id, n in hits.items() if n >= needed}

def _is_connected(selected: dict[str, set[int]]) -> bool:
    """Do the theme's terms actually co-occur, or is this two themes in a trenchcoat?

    Terms are nodes; an edge means two terms appeared in the same posting. If the
    graph is not one component the model has grouped things the data never puts
    together - which is what "AI and Machine Learning" was: {owasp, red teaming}
    on postings 5-6 and {mlops, monitoring, observability} on 13-15, with no
    posting linking them. The label cannot be true of both halves, so the theme is
    rejected outright rather than relabelled - a name chosen for the whole set is
    not evidence for a name for one half of it.
    """
    terms = list(selected)
    if len(terms) <= 1:
        return True
    reached, stack = {terms[0]}, [terms[0]]
    while stack:
        current = stack.pop()
        for term in terms:
            if term not in reached and selected[current] & selected[term]:
                reached.add(term)
                stack.append(term)
    return len(reached) == len(terms)

def _resolve(keywords: list[str], terms: dict[str, set[int]]) -> dict[str, str]:
    """Map what the model wrote back onto the shortlist. Anything else is dropped.

    Returns {shortlist term -> the model's spelling}, so a theme keeps the casing
    the model used ("OWASP") while being keyed on the computed term ("owasp").
    """
    resolved = {}
    for keyword in keywords:
        term = _norm(keyword or "")
        if term in terms and term not in resolved:
            resolved[term] = keyword.strip()
    return resolved

def _settle(selected: dict[str, set[int]]) -> tuple[dict[str, set[int]], set[int]]:
    """Iterate terms and evidence to a fixed point.

    Dropping a posting can leave a term supporting nothing, and dropping that term
    lowers the threshold, which can drop another posting. Loop until neither moves.
    """
    while selected:
        evidence = _evidence(selected)
        alive = {term: ids for term, ids in selected.items() if ids & evidence}
        if len(alive) == len(selected):
            return selected, evidence
        selected = alive
    return {}, set()

def _ground(themes: list[SkillTheme], rows) -> list[SkillTheme]:
    """Drop anything the data does not support. The model proposes; this disposes.

    Three things are re-derived rather than trusted: which terms exist, whether a
    grouping is coherent, and which postings back it. The model's only surviving
    contribution is the name and the sentence.
    """
    terms = recurring_terms(rows)
    kept = []
    for theme in themes:
        resolved = _resolve(theme.keywords, terms)
        selected = {term: terms[term] for term in resolved}
        if not selected or not _is_connected(selected):
            continue                                   # incoherent grouping
        selected, evidence = _settle(selected)
        if len(evidence) < MIN_POSTINGS_PER_THEME:
            continue                                   # not a pattern after all
        if not _is_connected(selected):
            continue                                   # settling split it in two
        theme.keywords = [resolved[term] for term in selected]
        theme.evidence_ids = sorted(evidence)
        kept.append(theme)
    return kept[:MAX_THEMES]

# --- the one model call -----------------------------------------------------

def find_gap_themes(rows=None) -> list[SkillTheme]:
    """Cluster logged skill gaps into themes. Returns [] when there is no pattern."""
    rows = keyword_rows() if rows is None else rows
    if len(rows) < MIN_POSTINGS_PER_THEME:
        return []

    terms = recurring_terms(rows)
    if len(terms) < MIN_POSTINGS_PER_THEME:
        return []   # nothing recurs; there is nothing to name, so don't wake the model

    task = Task(
        description=(
            "Below is every skill gap that recurs in this candidate's job pipeline. "
            "The list was computed from the data before you saw it: each term is "
            f"already known to appear in at least {MIN_POSTINGS_PER_THEME} different "
            "postings, with those postings listed after it.\n\n"
            f"{_term_corpus(terms)}\n\n"
            "Give these terms names by grouping them. Rules you must follow:\n"
            "- Use only terms from the list above, copied exactly. Never add a term, "
            "and never split or reword one.\n"
            "- Group two terms only if they share at least one posting number. Terms "
            "that never appear in the same posting belong in different themes.\n"
            "- The theme name must be true of EVERY term in it. If a name only "
            "describes some of them, the group is wrong - split it.\n"
            "- Each term goes in at most one theme. Leave a term out if it fits nowhere; "
            "not every term needs a theme.\n"
            "- Do not output posting ids. They are recomputed from the data.\n"
            f"- Report at most {MAX_THEMES} themes. Report zero if nothing groups.\n"
            "- Say nothing about interviews, rejections, or applications. That data does not exist here."
        ),
        expected_output=(
            'A JSON object: {"themes": [{"theme": "short name", '
            '"keywords": ["term copied from the list", "..."], '
            '"why_it_matters": "one short sentence"}]}. '
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

def print_terms(rows=None):
    """What the model will be handed. Run this when a theme looks wrong."""
    rows = keyword_rows() if rows is None else rows
    raw = sum(len(keywords) for _, _, _, keywords in rows)
    terms = recurring_terms(rows)
    print(f"\n📋 {len(terms)} recurring term(s) from {raw} keyword strings "
          f"across {len(rows)} posting(s)\n")
    print(_term_corpus(terms) or "   (nothing recurs)")

if __name__ == "__main__":
    import sys
    rows = keyword_rows()
    if "--terms" in sys.argv:
        print_terms(rows)          # the deterministic half, no model call
    else:
        print_themes(find_gap_themes(rows), rows)
