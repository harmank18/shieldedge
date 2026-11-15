"""
ShieldEdge Threat Intelligence Dashboard.
Real-time terminal dashboard powered by Rich library.
Reads from local SQLite alert log — zero cloud dependency.

Usage:
  python3 shieldedge.py --dashboard
  python3 -m shieldedge.dashboard
"""

import os
import sys
import sqlite3
import time
from datetime import datetime, timedelta
from collections import Counter

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "shieldedge_alerts.db")

console = Console()


def get_all_alerts(db_path: str = DB_PATH) -> list:
    """Fetch all alerts from SQLite."""
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, timestamp, input_type, verdict, confidence FROM alerts ORDER BY id DESC")
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def compute_stats(alerts: list) -> dict:
    """Compute dashboard statistics from alert rows."""
    if not alerts:
        return {
            "total": 0, "threats": 0, "safe": 0,
            "by_type": {}, "by_verdict": {}, "recent_24h": 0,
            "top_verdicts": [], "threat_rate": 0.0
        }

    total = len(alerts)
    verdicts = [r[3] for r in alerts]
    types = [r[2] for r in alerts]
    confidences = [r[4] for r in alerts]

    threats = sum(1 for v in verdicts if v != "SAFE")
    safe_count = total - threats
    threat_rate = round(threats / max(1, total) * 100, 1)

    by_verdict = dict(Counter(verdicts))
    by_type = dict(Counter(types))

    # Last 24 hours
    now = datetime.utcnow()
    recent_24h = 0
    for r in alerts:
        try:
            ts = datetime.fromisoformat(r[1].replace("Z", ""))
            if (now - ts) <= timedelta(hours=24):
                recent_24h += 1
        except Exception:
            pass

    avg_conf = round(sum(confidences) / max(1, len(confidences)) * 100, 1)

    return {
        "total": total,
        "threats": threats,
        "safe": safe_count,
        "threat_rate": threat_rate,
        "by_type": by_type,
        "by_verdict": by_verdict,
        "recent_24h": recent_24h,
        "avg_confidence": avg_conf,
    }


def render_summary_panel(stats: dict) -> Panel:
    """Render top-level stats panel."""
    total = stats["total"]
    threats = stats["threats"]
    safe = stats["safe"]
    rate = stats["threat_rate"]
    recent = stats["recent_24h"]
    avg_conf = stats.get("avg_confidence", 0)

    threat_color = "red" if rate > 30 else "yellow" if rate > 10 else "green"

    text = Text()
    text.append(f"  Total Scans:          {total}\n", style="bold white")
    text.append(f"  Threats Detected:     {threats}  ", style="bold red")
    text.append(f"({rate}% threat rate)\n", style=threat_color)
    text.append(f"  Clean / Safe:         {safe}\n", style="bold green")
    text.append(f"  Scans (Last 24h):     {recent}\n", style="cyan")
    text.append(f"  Avg Confidence:       {avg_conf}%\n", style="white")

    return Panel(text, title="[bold cyan]📊 ShieldEdge — Threat Intelligence Dashboard[/bold cyan]",
                 border_style="cyan", padding=(1, 2))


def render_verdict_table(stats: dict) -> Panel:
    """Render verdict breakdown table."""
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Verdict", style="bold", width=14)
    table.add_column("Count", justify="right", width=8)
    table.add_column("Bar", width=25)

    by_verdict = stats.get("by_verdict", {})
    total = max(1, stats["total"])

    verdict_order = ["PHISHING", "SCAM", "ANOMALY", "SUSPICIOUS", "SAFE"]
    verdict_colors = {
        "PHISHING": "red", "SCAM": "red", "ANOMALY": "dark_orange",
        "SUSPICIOUS": "yellow", "SAFE": "green"
    }

    for v in verdict_order:
        count = by_verdict.get(v, 0)
        if count == 0:
            continue
        bar_len = int(count / total * 20)
        bar = "█" * bar_len
        color = verdict_colors.get(v, "white")
        table.add_row(
            f"[{color}]{v}[/{color}]",
            str(count),
            f"[{color}]{bar}[/{color}]"
        )

    return Panel(table, title="[bold]Verdict Breakdown[/bold]", border_style="magenta")


def render_type_table(stats: dict) -> Panel:
    """Render input type breakdown table."""
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold blue")
    table.add_column("Input Type", style="bold", width=14)
    table.add_column("Count", justify="right", width=8)

    by_type = stats.get("by_type", {})
    type_icons = {"URL": "🌐", "SMS": "📱", "NETWORK_FLOW": "🔌"}

    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        icon = type_icons.get(t, "•")
        table.add_row(f"{icon} {t}", str(count))

    return Panel(table, title="[bold]By Input Type[/bold]", border_style="blue")


def render_recent_alerts(alerts: list, n: int = 15) -> Panel:
    """Render table of most recent alerts."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
    table.add_column("#", justify="right", width=4)
    table.add_column("Timestamp", width=20)
    table.add_column("Type", width=14)
    table.add_column("Verdict", width=12)
    table.add_column("Conf", justify="right", width=6)

    verdict_colors = {
        "PHISHING": "red", "SCAM": "red", "ANOMALY": "dark_orange",
        "SUSPICIOUS": "yellow", "SAFE": "green"
    }

    for row in alerts[:n]:
        aid, ts, itype, verdict, conf = row
        color = verdict_colors.get(verdict, "white")
        # Shorten timestamp
        ts_short = ts[:19] if len(ts) > 19 else ts
        table.add_row(
            str(aid),
            ts_short,
            itype,
            f"[{color}]{verdict}[/{color}]",
            f"{conf*100:.0f}%"
        )

    return Panel(table, title=f"[bold]Recent {min(n, len(alerts))} Alerts[/bold]", border_style="white")


def show_dashboard(db_path: str = DB_PATH, live: bool = False, refresh_interval: int = 5):
    """
    Main dashboard entry point.

    Args:
        db_path: Path to SQLite alerts database
        live: If True, auto-refresh every refresh_interval seconds
        refresh_interval: Seconds between refreshes in live mode
    """
    if not RICH_AVAILABLE:
        print("[ERROR] 'rich' library not installed. Run: pip install rich")
        sys.exit(1)

    def _render():
        alerts = get_all_alerts(db_path)
        stats = compute_stats(alerts)

        summary = render_summary_panel(stats)
        verdict_tbl = render_verdict_table(stats)
        type_tbl = render_type_table(stats)
        recent = render_recent_alerts(alerts, n=15)

        console.print(summary)
        console.print(Columns([verdict_tbl, type_tbl], equal=True))
        console.print(recent)

        if not alerts:
            console.print(
                Panel(
                    "[yellow]No alerts logged yet. Run ShieldEdge scans to populate the dashboard.[/yellow]",
                    border_style="yellow"
                )
            )

        footer = Text()
        footer.append(f"\nDatabase: {db_path}  |  ", style="dim")
        footer.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        console.print(footer)

    if live:
        console.print("[cyan]Live mode — press Ctrl+C to exit[/cyan]")
        try:
            while True:
                console.clear()
                _render()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard closed.[/yellow]")
    else:
        _render()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ShieldEdge Threat Dashboard")
    p.add_argument("--live", action="store_true", help="Auto-refresh every 5 seconds")
    p.add_argument("--db", default=DB_PATH, help="Path to alerts database")
    args = p.parse_args()
    show_dashboard(db_path=args.db, live=args.live)
