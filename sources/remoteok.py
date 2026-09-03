"""RemoteOK - remote-first tech board. Public JSON, no key, no account."""
from sources.common import fetch_json, looks_interesting, make_job

NAME = "remoteok"
FEED_URL = "https://remoteok.com/api"

def fetch() -> list[dict]:
    data = fetch_json(FEED_URL)
    jobs = []
    for item in data:
        # The feed's first element is a legal notice, not a job.
        if not isinstance(item, dict) or "position" not in item:
            continue
        if not looks_interesting(item.get("position", "")):
            continue
        jobs.append(make_job(
            company=item.get("company"),
            role=item.get("position"),
            posting=item.get("description", ""),
            url=item.get("url", ""),
            source=NAME,
            tags=item.get("tags", []),
        ))
    return jobs
