"""Fetch remote job postings from RemoteOK's public API and clean them up."""
import re
import html
import urllib.request
import json

FEED_URL = "https://remoteok.com/api"
# cheap first-pass filter: tags/words that suggest "worth scoring"
INTERESTING = {"ai", "machine learning", "ml", "llm", "python", "automation", "engineer", "backend", "data"}

def strip_html(text: str) -> str:
    """Plumbing job #1: turn tag soup into plain text."""
    text = html.unescape(text)                 # &amp; -> &, etc.
    text = re.sub(r"<[^>]+>", " ", text)       # remove tags
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    return text.strip()

def looks_interesting(job: dict) -> bool:
    """Cheap pre-filter. Lesson learned: feed tags lie (Carpenter tagged infosec!),
    so we trust the ROLE TITLE only. The Scout agent adds real judgment in week 2."""
    title = job.get("position", "").lower()
    # match WHOLE WORDS only: "ai" must not hide inside "mAIntenance" or "mAIl"
    return any(re.search(rf"\b{re.escape(word)}\b", title) for word in INTERESTING)

def fetch_jobs() -> list[dict]:
    req = urllib.request.Request(FEED_URL, headers={"User-Agent": "job-copilot-scout"})
    with urllib.request.urlopen(req) as r:
        data = json.load(r)
    jobs = []
    for item in data:
        if not isinstance(item, dict) or "position" not in item:
            continue  # feed's first element is a legal notice, not a job
        if not looks_interesting(item):
            continue
        jobs.append({
            "company": item.get("company", "?"),
            "role": item.get("position", "?"),
            "posting": strip_html(item.get("description", "")),
            "url": item.get("url", ""),
            "tags": item.get("tags", []),
        })
    return jobs

if __name__ == "__main__":
    jobs = fetch_jobs()
    print(f"🔭 {len(jobs)} interesting postings found\n")
    for j in jobs[:10]:
        print(f"- {j['role']} @ {j['company']}  ({', '.join(j['tags'][:4])})")
