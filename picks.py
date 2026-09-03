"""Top Picks: the few postings still worth acting on, with any letter already drafted.

Pure local read over tracker.db plus the drafts/ folder letter_agent already writes.
No model call, no network, nothing new to install - opening this view costs nothing
and cannot fail on a cold ollama.

A "pick" is deliberately narrow: still actionable (not already applied or rejected)
and scoring at least MIN_PICK_SCORE. Calling a 40% match a top pick would make the
view flattery rather than triage.
"""
from pathlib import Path

from tracker import list_applications

DRAFTS_DIR = Path("drafts")
MAX_PICKS = 3
# Same floor the scout uses before it bothers drafting a letter, so a pick is the
# kind of posting that can actually have one waiting.
MIN_PICK_SCORE = 65
# Statuses that mean the decision is already made - nothing left to act on.
CLOSED_STATUSES = {"applied", "rejected"}

def find_draft(app_id: int) -> tuple[str, str] | None:
    """(letter text, path) for a posting, or None if no letter has been drafted.

    Letters are written by letter_agent.save_draft as drafts/NNN-slug.md, so the
    id prefix is the whole lookup - no model call to re-draft anything here.
    """
    matches = sorted(DRAFTS_DIR.glob(f"{app_id:03d}-*.md"))
    if not matches:
        return None
    text = matches[0].read_text().strip()
    # save_draft writes a "# Role @ Company" heading the card already shows.
    lines = text.split("\n")
    if lines and lines[0].startswith("# "):
        text = "\n".join(lines[1:]).strip()
    return text, str(matches[0])

def top_picks(limit: int = MAX_PICKS) -> list[dict]:
    """Best still-open postings, best first. Deterministic: score, then newest, then id."""
    open_apps = [
        a for a in list_applications()
        if (a["status"] or "") not in CLOSED_STATUSES
        and (a["match_score"] or 0) >= MIN_PICK_SCORE
    ]
    open_apps.sort(key=lambda a: (-a["match_score"], a["analyzed_on"] or "", a["id"]))

    picks = []
    for rank, app in enumerate(open_apps[:limit], start=1):
        draft = find_draft(app["id"])
        picks.append({
            "rank": rank,
            "id": app["id"],
            "company": app["company"],
            "role": app["role"],
            "match_score": app["match_score"],
            "status": app["status"],
            "analyzed_on": app["analyzed_on"],
            "missing_keywords": [k.strip() for k in (app["missing_keywords"] or "").split(",") if k.strip()],
            # The link is the point of a pick: a match you then have to go and
            # find yourself has saved you nothing. Postings pasted in by hand
            # have no board behind them, so this stays None for those.
            "url": app.get("url"),
            "source": app.get("source"),
            "letter": draft[0] if draft else None,
            "letter_path": draft[1] if draft else None,
        })
    return picks

def show_picks():
    picks = top_picks()
    if not picks:
        print(f"No open postings at {MIN_PICK_SCORE}% or better yet.")
        return
    print(f"\n⭐ Top {len(picks)} pick(s) still open\n")
    for p in picks:
        origin = f", via {p['source']}" if p["source"] else ""
        print(f"{p['rank']}. {p['match_score']:>3}%  {p['role']} @ {p['company']}  (#{p['id']}, {p['status']}{origin})")
        if p["missing_keywords"]:
            print(f"      gaps: {', '.join(p['missing_keywords'])}")
        print(f"      letter: {p['letter_path'] if p['letter'] else 'not drafted yet'}")
        print(f"      apply:  {p['url'] or 'no link stored'}")

if __name__ == "__main__":
    show_picks()
