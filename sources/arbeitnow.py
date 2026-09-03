"""Arbeitnow - European job board (many EU/remote listings). Public JSON, no key.

Postings are a mix of English and German; the shared title filter is English, so
German-titled roles simply never pass it. That is a filter limitation, not a bug
to fix here - it belongs in search preferences.
"""
from sources.common import fetch_json, looks_interesting, make_job

NAME = "arbeitnow"
FEED_URL = "https://www.arbeitnow.com/api/job-board-api"

def fetch() -> list[dict]:
    data = fetch_json(FEED_URL)
    jobs = []
    for item in data.get("data", []) if isinstance(data, dict) else []:
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
