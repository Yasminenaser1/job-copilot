"""Eval suite for the source layer. No model, no network - every case is a fixture.

The scout evals test judgment. These test the plumbing underneath it, which is
where the boring failures live: a board renames a field and postings vanish, a
title filter matches "mAIntenance", the same job gets logged twice from two
boards, or a match arrives with no link and costs you the search you were
trying to avoid.
"""
from sources.common import looks_interesting, make_job, strip_html
from sources import remoteok, remotive, arbeitnow
from feeds import dedupe, dedupe_key

# One realistic payload per board, in that board's own field names.
REMOTEOK_PAYLOAD = [
    {"legal": "notice, not a job"},                      # the feed's first element
    {"position": "AI Engineer", "company": "Acme", "description": "<p>Build &amp; ship LLM tools</p>",
     "url": "https://remoteok.com/jobs/1", "tags": ["ai", "python"]},
    {"position": "Maintenance Technician", "company": "Acme", "description": "Fix things",
     "url": "https://remoteok.com/jobs/2", "tags": ["ai"]},   # tag lies; title doesn't
]
REMOTIVE_PAYLOAD = {"0-legal-notice": "...", "jobs": [
    {"title": "Python Automation Engineer", "company_name": "Globex",
     "description": "<div>Automate pipelines</div>", "url": "https://remotive.com/jobs/9", "tags": ["python"]},
    {"title": "Account Executive", "company_name": "Globex", "description": "Sell things",
     "url": "https://remotive.com/jobs/10", "tags": []},
]}
ARBEITNOW_PAYLOAD = {"data": [
    {"title": "Backend Engineer (m/w/d)", "company_name": "Initech",
     "description": "<p>APIs</p>", "url": "https://arbeitnow.com/jobs/3", "tags": ["backend"]},
    {"title": "Hausmeister", "company_name": "Initech", "description": "Hausmeisterei",
     "url": "https://arbeitnow.com/jobs/4", "tags": []},
]}

def _fetch_with(module, payload):
    """Run a source's fetch() against a fixture instead of the live board."""
    real = module.fetch_json
    module.fetch_json = lambda url: payload
    try:
        return module.fetch()
    finally:
        module.fetch_json = real

CASES = []
def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap

@case("title_filter_matches_whole_words_only")
def _():
    assert looks_interesting("AI Engineer")
    assert looks_interesting("Backend Developer, Python")
    # The v2 bug: "ai" hiding inside another word.
    assert not looks_interesting("Maintenance Technician")
    assert not looks_interesting("Mail Carrier")
    assert not looks_interesting("Account Executive")

@case("html_is_stripped_and_entities_decoded")
def _():
    assert strip_html("<p>Build &amp;   ship</p>") == "Build & ship"
    assert strip_html(None) == ""

@case("every_source_returns_the_same_shape")
def _():
    fixtures = [(remoteok, REMOTEOK_PAYLOAD), (remotive, REMOTIVE_PAYLOAD), (arbeitnow, ARBEITNOW_PAYLOAD)]
    for module, payload in fixtures:
        for job in _fetch_with(module, payload):
            assert set(job) == {"company", "role", "posting", "url", "source", "tags"}, job
            assert job["source"] == module.NAME
            assert "<" not in job["posting"]

@case("every_source_filters_out_non_matching_titles")
def _():
    assert [j["role"] for j in _fetch_with(remoteok, REMOTEOK_PAYLOAD)] == ["AI Engineer"]
    assert [j["role"] for j in _fetch_with(remotive, REMOTIVE_PAYLOAD)] == ["Python Automation Engineer"]
    assert [j["role"] for j in _fetch_with(arbeitnow, ARBEITNOW_PAYLOAD)] == ["Backend Engineer (m/w/d)"]

@case("junk_rows_never_become_jobs")
def _():
    # RemoteOK's legal notice has no "position"; a None entry is defensive.
    assert len(_fetch_with(remoteok, REMOTEOK_PAYLOAD + [None, "nonsense"])) == 1
    assert _fetch_with(remotive, {"nothing": "here"}) == []
    assert _fetch_with(arbeitnow, []) == []

@case("missing_fields_do_not_crash_a_sweep")
def _():
    jobs = _fetch_with(remoteok, [{"position": "AI Engineer"}])
    assert jobs[0]["company"] == "?" and jobs[0]["url"] == "" and jobs[0]["posting"] == ""

@case("every_job_carries_a_link")
def _():
    """The whole point of the change: a match you can't open is a match you
    still have to go and find."""
    for module, payload in [(remoteok, REMOTEOK_PAYLOAD), (remotive, REMOTIVE_PAYLOAD), (arbeitnow, ARBEITNOW_PAYLOAD)]:
        for job in _fetch_with(module, payload):
            assert job["url"].startswith("https://"), job

@case("cross_posted_job_counts_once")
def _():
    a = make_job("Acme", "AI Engineer", "x", "https://remoteok.com/1", "remoteok")
    b = make_job("acme", "  ai engineer ", "x", "https://remotive.com/2", "remotive")
    kept = dedupe([a, b])
    assert len(kept) == 1 and kept[0]["source"] == "remoteok", kept

@case("gender_marker_does_not_hide_a_duplicate")
def _():
    a = make_job("Initech", "Backend Engineer", "x", "https://remoteok.com/1", "remoteok")
    b = make_job("Initech", "Backend Engineer (m/w/d)", "x", "https://arbeitnow.com/2", "arbeitnow")
    assert len(dedupe([a, b])) == 1

@case("different_roles_at_one_company_both_survive")
def _():
    """Dedupe must not over-merge: parenthesised specialisms are real differences."""
    a = make_job("Initech", "Engineer (Backend)", "x", "https://x/1", "remoteok")
    b = make_job("Initech", "Engineer (Frontend)", "x", "https://x/2", "remoteok")
    assert len(dedupe([a, b])) == 2
    assert dedupe_key(a) != dedupe_key(b)

@case("same_url_twice_counts_once")
def _():
    a = make_job("Acme", "AI Engineer", "x", "https://remoteok.com/1", "remoteok")
    b = make_job("Acme Inc", "AI Engineer II", "x", "https://remoteok.com/1", "remoteok")
    assert len(dedupe([a, b])) == 1

@case("jobs_without_links_are_not_deduped_against_each_other")
def _():
    """Empty URLs are absence, not a shared identity."""
    a = make_job("Acme", "AI Engineer", "x", "", "manual")
    b = make_job("Globex", "Python Engineer", "x", "", "manual")
    assert len(dedupe([a, b])) == 2

@case("tracker_stores_and_returns_the_link")
def _():
    import tempfile, os
    import tracker
    from match import MatchReport
    old_path = tracker.DB_PATH
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd); os.unlink(path)
    tracker.DB_PATH = path
    try:
        report = MatchReport(match_score=80, matching_skills=["python"], missing_keywords=["k8s"],
                             projects_to_emphasize=["job-copilot"], one_line_verdict="fine")
        app_id = tracker.log_application("Acme", "AI Engineer", report,
                                         url="https://remoteok.com/1", source="remoteok")
        row = [a for a in tracker.list_applications() if a["id"] == app_id][0]
        assert row["url"] == "https://remoteok.com/1" and row["source"] == "remoteok"
        # A hand-pasted posting has no board behind it and must still log.
        manual = tracker.log_application("Globex", "Data Engineer", report)
        manual_row = [a for a in tracker.list_applications() if a["id"] == manual][0]
        assert manual_row["url"] is None and manual_row["source"] is None
    finally:
        tracker.DB_PATH = old_path
        if os.path.exists(path):
            os.unlink(path)

@case("old_tracker_db_gains_the_new_columns")
def _():
    """A database written before this change must migrate, not explode."""
    import sqlite3, tempfile, os
    import tracker
    old_path = tracker.DB_PATH
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    with sqlite3.connect(path) as conn:   # the pre-change schema, verbatim
        conn.execute("""CREATE TABLE applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT, company TEXT, role TEXT,
            match_score INTEGER, missing_keywords TEXT,
            status TEXT DEFAULT 'analyzed', analyzed_on TEXT)""")
        conn.execute("INSERT INTO applications (company, role, match_score) VALUES ('Old', 'Role', 70)")
    tracker.DB_PATH = path
    try:
        rows = tracker.list_applications()
        assert len(rows) == 1, "the pre-existing row must survive the migration"
        assert rows[0]["url"] is None and rows[0]["source"] is None
    finally:
        tracker.DB_PATH = old_path
        os.unlink(path)

def run_feed_evals() -> tuple[int, int]:
    passed = 0
    print("🔭 Feed/source eval cases (no model, no network)...")
    for name, fn in CASES:
        try:
            fn()
            print(f"✅ PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL  {name}\n       ↳ {e}")
        except Exception as e:
            print(f"❌ ERROR {name}\n       ↳ {type(e).__name__}: {e}")
    return passed, len(CASES)

if __name__ == "__main__":
    p, t = run_feed_evals()
    print(f"\n📊 {p}/{t} passed")
