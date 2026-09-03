"""The scout pipeline: sweep every source -> filter -> judge -> score -> log.

Each posting that survives is logged with the URL it came from, so a match in
Top Picks is one click from the actual application form.
"""
from feeds import fetch_jobs
from scout_agent import judge_posting
from letter_agent import draft_letter, save_draft
from match import analyze
from tracker import log_application, update_status, list_applications, tracked_urls

MAX_PER_RUN = 10   # polite ceiling on model calls per run
MIN_POSTING_LEN = 200   # skip stub descriptions not worth scoring

def already_tracked() -> set[tuple[str, str]]:
    """Dedupe guard: (company, role) pairs we've already logged."""
    return {(a["company"].lower(), a["role"].lower()) for a in list_applications()}

def run_scout():
    print("🔭 Sweeping sources...")
    jobs = fetch_jobs(verbose=True)
    seen = already_tracked()
    seen_urls = tracked_urls()
    scored = 0

    for job in jobs:
        if scored >= MAX_PER_RUN:
            print(f"⏸️  Hit per-run cap ({MAX_PER_RUN}); rest will be caught next run.")
            break
        key = (job["company"].lower(), job["role"].lower())
        # A board can re-list the same job under a tweaked title, so the URL is
        # checked too - it is the one identifier that doesn't drift.
        if key in seen or (job["url"] and job["url"] in seen_urls):
            print(f"↩️  Skipping (already tracked): {job['role']} @ {job['company']}")
            continue
        if len(job["posting"]) < MIN_POSTING_LEN:
            print(f"⏭️  Skipping (description too thin): {job['role']} @ {job['company']}")
            continue

        print(f"🤖 Scout judging: {job['role']} @ {job['company']}  [{job['source']}] ...")
        verdict = judge_posting(job["role"], job["company"], job["posting"])
        if not verdict.worth_pursuing:
            print(f"   🚫 Scout rejected: {verdict.reason}")
            continue
        print(f"   👍 Scout approved: {verdict.reason}")
        print(f"⚙️  Scoring: {job['role']} @ {job['company']} ...")
        report = analyze(job["posting"])
        app_id = log_application(job["company"], job["role"], report,
                                 url=job["url"], source=job["source"])
        update_status(app_id, "scouted")
        print(f"   → {report.match_score}%  (logged as #{app_id})  {job['url'] or 'no link'}")
        if report.match_score >= 65:
            print(f"   ✍️  Drafting letter (score ≥ 65)...")
            letter = draft_letter(job["role"], job["company"], job["posting"])
            path = save_draft(app_id, job["company"], job["role"], letter)
            print(f"   💾 Draft saved: {path}")
        scored += 1

    print(f"\n✅ Scout run complete: {scored} new posting(s) scored and logged.")

if __name__ == "__main__":
    run_scout()
