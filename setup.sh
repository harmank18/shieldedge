#!/usr/bin/env bash
set -e

echo "=== ShieldEdge: On-Device Setup for Linux / Arm ==="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required."
    exit 1
fi

# Create virtual environment if not present
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
fi

echo "[2/3] Activating virtual environment & installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/3] Creating SQLite log database..."
python3 -c "
import sqlite3
conn = sqlite3.connect('shieldedge_alerts.db')
cursor = conn.cursor()
cursor.execute('''
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
''')
conn.commit()
conn.close()
print('SQLite database initialized successfully.')
"

echo "=== Setup complete! ShieldEdge is ready. ==="
