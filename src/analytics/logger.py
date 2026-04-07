import sqlite3
import os
from datetime import datetime
from pathlib import Path


DB_PATH = os.path.join("data", "analytics.db")


class AnalyticsLogger:
    """
    Logs every chatbot interaction to a SQLite database.

    Captures: session_id, question, answer, sources used,
    confidence level, escalation flag, response time, timestamp.

    SQLite chosen for simplicity — no server setup needed.
    Production upgrade path: swap for PostgreSQL.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Creates the interactions table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT NOT NULL,
                    session_id      TEXT NOT NULL,
                    question        TEXT NOT NULL,
                    answer          TEXT NOT NULL,
                    sources         TEXT,
                    confidence      TEXT,
                    escalated       INTEGER DEFAULT 0,
                    response_ms     INTEGER,
                    turn_number     INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON interactions(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON interactions(session_id)
            """)
            conn.commit()
        print(f"[Analytics] Database ready: {self.db_path}")

    def log(
        self,
        session_id:  str,
        question:    str,
        answer:      str,
        sources:     list,
        confidence:  str,
        escalated:   bool,
        response_ms: int,
        turn_number: int = 1
    ):
        """Logs one interaction to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO interactions
                (timestamp, session_id, question, answer,
                 sources, confidence, escalated, response_ms, turn_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                session_id,
                question,
                answer,
                ", ".join(sources) if sources else "",
                confidence,
                1 if escalated else 0,
                response_ms,
                turn_number
            ))
            conn.commit()

    def get_summary(self) -> dict:
        """Returns key analytics metrics for the dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) as total FROM interactions")
            total = cur.fetchone()["total"]

            cur.execute("SELECT COUNT(DISTINCT session_id) as sessions FROM interactions")
            sessions = cur.fetchone()["sessions"]

            cur.execute("SELECT COUNT(*) as escalated FROM interactions WHERE escalated=1")
            escalated = cur.fetchone()["escalated"]

            cur.execute("SELECT AVG(response_ms) as avg_ms FROM interactions")
            avg_ms = cur.fetchone()["avg_ms"] or 0

            cur.execute("""
                SELECT confidence, COUNT(*) as count
                FROM interactions
                GROUP BY confidence
            """)
            confidence_counts = {row["confidence"]: row["count"] for row in cur.fetchall()}

            cur.execute("""
                SELECT sources, COUNT(*) as count
                FROM interactions
                WHERE sources != ''
                GROUP BY sources
                ORDER BY count DESC
                LIMIT 5
            """)
            top_sources = [
                {"source": row["sources"], "count": row["count"]}
                for row in cur.fetchall()
            ]

            cur.execute("""
                SELECT question, COUNT(*) as count
                FROM interactions
                GROUP BY question
                ORDER BY count DESC
                LIMIT 10
            """)
            top_questions = [
                {"question": row["question"], "count": row["count"]}
                for row in cur.fetchall()
            ]

        return {
            "total_interactions": total,
            "unique_sessions":    sessions,
            "total_escalations":  escalated,
            "escalation_rate":    round((escalated / total * 100), 1) if total else 0,
            "avg_response_ms":    round(avg_ms),
            "confidence_counts":  confidence_counts,
            "top_sources":        top_sources,
            "top_questions":      top_questions,
        }

    def get_recent(self, limit: int = 20) -> list:
        """Returns the most recent N interactions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT id, timestamp, session_id, question,
                       confidence, escalated, response_ms, sources
                FROM interactions
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cur.fetchall()]