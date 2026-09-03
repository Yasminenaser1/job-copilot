"""Sweep every job source, clean the results up, and hand back one list.

v2 read RemoteOK and nothing else. This is the same idea widened: several free,
key-less public APIs, normalized to one shape (sources/common.make_job), with
two rules that matter once there is more than one source:

  * One dead source must not kill the sweep. Each is wrapped; a failure is
    reported and the others still return.
  * The same job cross-posted to two boards must count once. Dedupe is on
    (company, role) and on URL, and the first source in sources.ALL_SOURCES
    wins - so the link you get is from the board listed first.
"""
import re

from sources import ALL_SOURCES
from sources.common import INTERESTING, looks_interesting, strip_html  # re-exported

# Gender markers glued onto European job titles - "(m/w/d)", "(m/f/x)". They are
# noise for dedupe, and stripping only these avoids merging genuinely different
# roles like "Engineer (Backend)" and "Engineer (Frontend)".
GENDER_MARKER = re.compile(r"\(\s*[mwfdx](\s*/\s*[mwfdx])+\s*\)", re.I)

def _norm(text: str) -> str:
    text = GENDER_MARKER.sub(" ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()

def dedupe_key(job: dict) -> tuple[str, str]:
    return (_norm(job.get("company", "")), _norm(job.get("role", "")))

def dedupe(jobs: list[dict]) -> list[dict]:
    """Keep the first sighting of each job; later duplicates are dropped."""
    seen_pairs: set[tuple[str, str]] = set()
    seen_urls: set[str] = set()
    unique = []
    for job in jobs:
        key = dedupe_key(job)
        url = job.get("url", "")
        if key in seen_pairs or (url and url in seen_urls):
            continue
        seen_pairs.add(key)
        if url:
            seen_urls.add(url)
        unique.append(job)
    return unique

def fetch_jobs(verbose: bool = False) -> list[dict]:
    """Every interesting posting across every source, deduped."""
    collected = []
    for source in ALL_SOURCES:
        try:
            found = source.fetch()
        except Exception as e:
            # A board being down, rate-limiting, or changing its JSON shape is a
            # normal Tuesday. Say so out loud and keep going.
            print(f"⚠️  {source.NAME} unavailable ({type(e).__name__}: {e}) - skipping it this run.")
            continue
        if verbose:
            print(f"   {source.NAME}: {len(found)} interesting")
        collected.extend(found)

    unique = dedupe(collected)
    if verbose and len(unique) != len(collected):
        print(f"   deduped: {len(collected)} -> {len(unique)} ({len(collected) - len(unique)} cross-posted)")
    return unique

if __name__ == "__main__":
    print(f"🔭 Sweeping {len(ALL_SOURCES)} source(s)...")
    jobs = fetch_jobs(verbose=True)
    print(f"\n🔭 {len(jobs)} interesting postings found\n")
    for j in jobs[:15]:
        print(f"- [{j['source']}] {j['role']} @ {j['company']}")
        print(f"    {j['url'] or 'no link'}")
