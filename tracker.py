"""Application tracker - SQLite log of every posting analyzed."""
import json
import sqlite3
import sys
from datetime import date
from match import MatchReport

DB_PATH = "tracker.db"

# Columns added after the first version shipped. Existing databases get them via
# _migrate(); rows logged before the change keep NULL, which the UI renders as
# "no link" rather than pretending it has one.
ADDED_COLUMNS = {
    "url": "TEXT",      # where to actually apply
    "source": "TEXT",   # which board it came from
    "description": "TEXT",  # the posting text itself
}

# --- missing_keywords codec -------------------------------------------------
# Stored as a JSON array, not a comma-join. A keyword is free text written by the
# model, so it can contain a comma - "cloud technologies (AWS, Azure or GCP)" is a
# real example - and ", ".join() flattens that into something no split() can undo.
# JSON escapes the delimiter, needs no dependency, and stays plain TEXT in SQLite.

def encode_keywords(keywords: list[str]) -> str:
    return json.dumps([k.strip() for k in (keywords or []) if k and k.strip()],
                      ensure_ascii=False)

def decode_keywords(raw) -> list[str]:
    """Read either format. Rows written before the change are comma-joined, and
    stay readable forever - the leading '[' is what tells the two apart."""
    if raw is None:
        return []
    if isinstance(raw, list):          # already decoded upstream
        return [str(k).strip() for k in raw if str(k).strip()]
    raw = str(raw).strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except ValueError:
            pass                        # not JSON after all; fall through to legacy
        else:
            if isinstance(parsed, list):
                return [str(k).strip() for k in parsed if str(k).strip()]
    return _split_legacy(raw)

def _split_legacy(raw: str) -> list[str]:
    """Best-effort recovery of a pre-JSON comma-join.

    Only commas at bracket depth zero separate keywords, which is what puts
    "cloud technologies (AWS, Azure or GCP)" back together. A keyword whose comma
    sat outside any bracket is not recoverable - that information was destroyed
    when it was written, and this deliberately does not guess at it.
    """
    parts, buf, depth = [], [], 0
    for ch in raw:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT,
                role TEXT,
                match_score INTEGER,
                missing_keywords TEXT,
                status TEXT DEFAULT 'analyzed',
                analyzed_on TEXT,
                url TEXT,
                source TEXT,
                description TEXT
            )
        """)
        _migrate(conn)

def _migrate(conn):
    """Add any column an older tracker.db predates. Safe to run every open."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(applications)")}
    for column, coltype in ADDED_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE applications ADD COLUMN {column} {coltype}")

def log_application(company: str, role: str, report: MatchReport,
                    url: str | None = None, source: str | None = None,
                    description: str | None = None) -> int:
    """Log a scored posting. url/source are optional so a hand-pasted posting -
    which has no board behind it - logs exactly as it always did."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO applications (company, role, match_score, missing_keywords, analyzed_on, url, source, description)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (company, role, report.match_score, encode_keywords(report.missing_keywords),
             date.today().isoformat(), url or None, source or None, description or None),
        )
        return cur.lastrowid

def update_status(app_id: int, status: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE applications SET status = ? WHERE id = ?", (status, app_id))

def list_applications() -> list[dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # id breaks score ties so row order is stable run to run - the Insights agent
        # feeds these rows straight into its prompt, and a shuffled corpus is a changed prompt
        rows = conn.execute("SELECT * FROM applications ORDER BY match_score DESC, id").fetchall()
        apps = []
        for r in rows:
            app = dict(r)
            # Hand callers a real list. The string form never leaves this module,
            # so there is nowhere left for a split(",") to reintroduce the bug.
            app["missing_keywords"] = decode_keywords(app.get("missing_keywords"))
            apps.append(app)
        return apps

def tracked_urls() -> set[str]:
    """Every posting URL already logged - the scout's dedupe guard across runs."""
    return {a["url"] for a in list_applications() if a.get("url")}

def rewrite_legacy_keywords(dry_run: bool = True) -> list[tuple[int, str, list[str]]]:
    """Opt-in: rewrite pre-JSON comma-joined rows in place.

    Not required - decode_keywords() reads the old format forever, so this only
    stops the legacy path from running on every read. Back tracker.db up first;
    the bracket-aware recovery is a best effort, not a guarantee.
    """
    init_db()
    changed = []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT id, missing_keywords FROM applications").fetchall():
            raw = row["missing_keywords"]
            if raw is None or str(raw).strip().startswith("["):
                continue                       # already JSON, or never had any
            keywords = decode_keywords(raw)
            changed.append((row["id"], str(raw), keywords))
            if not dry_run:
                conn.execute("UPDATE applications SET missing_keywords = ? WHERE id = ?",
                             (encode_keywords(keywords), row["id"]))
    return changed

def _run_rewrite(apply_it: bool):
    changed = rewrite_legacy_keywords(dry_run=not apply_it)
    if not changed:
        print("No legacy comma-joined rows left - nothing to rewrite.")
        return
    for app_id, raw, keywords in changed:
        print(f"#{app_id}  {len(keywords)} keyword(s)")
        print(f"   was: {raw}")
        print(f"   now: {keywords}")
    if apply_it:
        print(f"\nRewrote {len(changed)} row(s).")
    else:
        print(f"\n{len(changed)} row(s) would change. Back up tracker.db, then re-run "
              f"with --rewrite-keywords --apply to write them.")

if __name__ == "__main__":
    if "--rewrite-keywords" in sys.argv:
        _run_rewrite(apply_it="--apply" in sys.argv)
        sys.exit(0)
    apps = list_applications()
    if not apps:
        print("No applications logged yet.")
    for a in apps:
        link = a.get("url") or "-"
        print(f"#{a['id']}  {a['match_score']:>3}%  {a['company']:<20} {a['role']:<30} [{a['status']}]  {a['analyzed_on']}  {link}")
