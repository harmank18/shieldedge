"""
Local SQLite Storage Module for ShieldEdge.
Logs threat detection events in a privacy-preserving manner using SHA-256 hashes.
"""

import os
import sqlite3
import hashlib

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shieldedge_alerts.db")


def hash_input(raw_input: str) -> str:
    """Returns SHA-256 hash of input to ensure zero raw sensitive data is persisted."""
    return hashlib.sha256(raw_input.encode("utf-8")).hexdigest()


class AlertLogger:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_type TEXT,
                input_hash TEXT,
                verdict TEXT,
                confidence REAL,
                remediation TEXT,
                is_false_positive INTEGER DEFAULT 0
            )
            """)
            conn.commit()

    def log_alert(self, input_type: str, raw_input: str, verdict: str, confidence: float, remediation: str = ""):
        inp_hash = hash_input(raw_input)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO alerts (input_type, input_hash, verdict, confidence, remediation)
            VALUES (?, ?, ?, ?, ?)
            """, (input_type, inp_hash, verdict, confidence, remediation))
            conn.commit()

    def get_recent_alerts(self, limit: int = 10):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, input_type, verdict, confidence FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
            return cursor.fetchall()
