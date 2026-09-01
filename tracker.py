"""Application tracker - SQLite log of every posting analyzed."""
import sqlite3
from datetime import date
from match import MatchReport

DB_PATH = "tracker.db"

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
                analyzed_on TEXT
            )
        """)

def log_application(company: str, role: str, report: MatchReport) -> int:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO applications (company, role, match_score, missing_keywords, analyzed_on) VALUES (?, ?, ?, ?, ?)",
            (company, role, report.match_score, ", ".join(report.missing_keywords), date.today().isoformat()),
        )
        return cur.lastrowid

def update_status(app_id: int, status: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE applications SET status = ? WHERE id = ?", (status, app_id))

def list_applications() -> list[dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM applications ORDER BY match_score DESC").fetchall()
        return [dict(r) for r in rows]

if __name__ == "__main__":
    apps = list_applications()
    if not apps:
        print("No applications logged yet.")
    for a in apps:
        print(f"#{a['id']}  {a['match_score']:>3}%  {a['company']:<20} {a['role']:<30} [{a['status']}]  {a['analyzed_on']}")
