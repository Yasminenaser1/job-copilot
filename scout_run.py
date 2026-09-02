"""Week 1 pipeline: fetch feed -> filter -> score with matcher -> log as 'scouted'."""
from feeds import fetch_jobs
from scout_agent import judge_posting
from letter_agent import draft_letter, save_draft
from match import analyze
from tracker import log_application, update_status, list_applications

MAX_PER_RUN = 10   # polite ceiling on model calls per run
MIN_POSTING_LEN = 200   # skip stub descriptions not worth scoring

def already_tracked() -> set[tuple[str, str]]:
    """Dedupe guard: (company, role) pairs we've already logged."""
    return {(a["company"].lower(), a["role"].lower()) for a in list_applications()}

def run_scout():
    print("🔭 Fetching feed...")
    jobs = fetch_jobs()
    seen = already_tracked()
    scored = 0

    for job in jobs:
        if scored >= MAX_PER_RUN:
            print(f"⏸️  Hit per-run cap ({MAX_PER_RUN}); rest will be caught next run.")
            break
        key = (job["company"].lower(), job["role"].lower())
        if key in seen:
            print(f"↩️  Skipping (already tracked): {job['role']} @ {job['company']}")
            continue
        if len(job["posting"]) < MIN_POSTING_LEN:
            print(f"⏭️  Skipping (description too thin): {job['role']} @ {job['company']}")
            continue

        print(f"🤖 Scout judging: {job['role']} @ {job['company']} ...")
        verdict = judge_posting(job["role"], job["company"], job["posting"])
        if not verdict.worth_pursuing:
            print(f"   🚫 Scout rejected: {verdict.reason}")
            continue
        print(f"   👍 Scout approved: {verdict.reason}")
        print(f"⚙️  Scoring: {job['role']} @ {job['company']} ...")
        report = analyze(job["posting"])
        app_id = log_application(job["company"], job["role"], report)
        update_status(app_id, "scouted")
        print(f"   → {report.match_score}%  (logged as #{app_id})")
        if report.match_score >= 65:
            print(f"   ✍️  Drafting letter (score ≥ 65)...")
            letter = draft_letter(job["role"], job["company"], job["posting"])
            path = save_draft(app_id, job["company"], job["role"], letter)
            print(f"   💾 Draft saved: {path}")
        scored += 1

    print(f"\n✅ Scout run complete: {scored} new posting(s) scored and logged.")

if __name__ == "__main__":
    run_scout()
