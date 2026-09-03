"""Shared plumbing every job source needs: fetch, clean, filter, normalize.

A source module's only job is to know one API's field names. Everything that
should behave identically across sources - HTML stripping, the title filter,
the shape a job comes out as - lives here, so a new source is ~20 lines and
cannot quietly disagree with the others about what a job looks like.
"""
import html
import json
import re
import urllib.request

USER_AGENT = "job-copilot-scout"
TIMEOUT_SECONDS = 20

# Cheap first-pass filter: words in a ROLE TITLE that suggest "worth scoring".
# Lesson from v2: feed tags lie (a Carpenter posting arrived tagged `infosec`),
# so nothing here is ever matched against tags or descriptions.
INTERESTING = {
    "ai", "machine learning", "ml", "llm", "genai",
    "python", "automation", "engineer", "backend", "data",
}

def fetch_json(url: str) -> dict | list:
    """GET and parse JSON. Raises on network/parse failure - the caller decides
    whether one dead source should stop the whole sweep (it shouldn't)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
        return json.load(r)

def strip_html(text: str) -> str:
    """Turn tag soup into plain text the matcher can actually read."""
    text = html.unescape(text or "")           # &amp; -> &, etc.
    text = re.sub(r"<[^>]+>", " ", text)       # remove tags
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    return text.strip()

def looks_interesting(role: str) -> bool:
    """Title-only pre-filter, whole words only.

    Substring matching is a trap: "ai" hides inside "mAIntenance" and "mAIl
    carrier". The Scout agent supplies the real judgment after this.
    """
    role = (role or "").lower()
    return any(re.search(rf"\b{re.escape(word)}\b", role) for word in INTERESTING)

def make_job(company, role, posting, url, source, tags=None) -> dict:
    """The one shape the rest of the pipeline knows about."""
    return {
        "company": (company or "?").strip() or "?",
        "role": (role or "?").strip() or "?",
        "posting": strip_html(posting),
        "url": (url or "").strip(),
        "source": source,
        "tags": [str(t) for t in (tags or []) if str(t).strip()],
    }
