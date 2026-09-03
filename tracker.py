"""Application tracker - SQLite log of every posting analyzed."""
import sqlite3
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
            (company, role, report.match_score, ", ".join(report.missing_keywords),
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
        return [dict(r) for r in rows]

def tracked_urls() -> set[str]:
    """Every posting URL already logged - the scout's dedupe guard across runs."""
    return {a["url"] for a in list_applications() if a.get("url")}

if __name__ == "__main__":
    apps = list_applications()
    if not apps:
        print("No applications logged yet.")
    for a in apps:
        link = a.get("url") or "-"
        print(f"#{a['id']}  {a['match_score']:>3}%  {a['company']:<20} {a['role']:<30} [{a['status']}]  {a['analyzed_on']}  {link}")
