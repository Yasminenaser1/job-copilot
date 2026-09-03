"""Remotive - curated remote jobs. Public JSON, no key, no account."""
from sources.common import fetch_json, looks_interesting, make_job

NAME = "remotive"
FEED_URL = "https://remotive.com/api/remote-jobs"

def fetch() -> list[dict]:
    data = fetch_json(FEED_URL)
    jobs = []
    # Top level also carries legal-notice keys; only "jobs" holds postings.
    for item in data.get("jobs", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        if not looks_interesting(item.get("title", "")):
            continue
        jobs.append(make_job(
            company=item.get("company_name"),
            role=item.get("title"),
            posting=item.get("description", ""),
            url=item.get("url", ""),
            source=NAME,
            tags=item.get("tags", []),
        ))
    return jobs
