"""
ShieldEdge REST API Server
Exposes on-device threat detection capabilities as a JSON REST API.
Runs entirely locally — zero cloud dependency.

Endpoints:
  POST /analyze/url     — URL phishing detection
  POST /analyze/sms     — SMS scam/smishing detection
  POST /analyze/flow    — Network flow anomaly detection
  GET  /alerts          — Recent threat log
  GET  /health          — Health check / model status
  GET  /stats           — Detection statistics

Usage:
  python3 -m shieldedge.api_server
  python3 shieldedge.py --serve
"""

import os
import time
from datetime import datetime, timezone

try:
    from flask import Flask, request, jsonify, abort
except ImportError:
    raise ImportError("Flask is required: pip install flask")

from shieldedge.url_detector import URLDetector
from shieldedge.sms_detector import SMSDetector
from shieldedge.network_detector import NetworkDetector
from shieldedge.decision_engine import DecisionEngine
from shieldedge.storage import AlertLogger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_MODEL  = os.path.join(BASE_DIR, "models", "url_rf_model.joblib")
SMS_MODEL  = os.path.join(BASE_DIR, "models", "sms_pipeline.joblib")
NET_MODEL  = os.path.join(BASE_DIR, "models", "network_isolation_forest.joblib")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Lazy-loaded singletons
_url_det = None
_sms_det = None
_net_det = None
_engine  = None
_logger  = None
_start_time = time.time()
_request_count = {"url": 0, "sms": 0, "flow": 0}
_threat_count  = {"url": 0, "sms": 0, "flow": 0}


def _get_url_det():
    global _url_det
    if _url_det is None:
        _url_det = URLDetector(URL_MODEL)
    return _url_det


def _get_sms_det():
    global _sms_det
    if _sms_det is None:
        _sms_det = SMSDetector(SMS_MODEL)
    return _sms_det


def _get_net_det():
    global _net_det
    if _net_det is None:
        _net_det = NetworkDetector(NET_MODEL)
    return _net_det


def _get_engine():
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    return _engine


def _get_logger():
    global _logger
    if _logger is None:
        _logger = AlertLogger()
    return _logger


def _run_pipeline(raw_result: dict, input_type_key: str, explain: bool = False) -> dict:
    """Run decision engine + logging and return structured JSON response."""
    t0 = time.perf_counter()
    engine = _get_engine()
    logger = _get_logger()

    eval_res = engine.evaluate(raw_result)
    verdict = eval_res["verdict"]
    conf = eval_res["confidence"]

    _request_count[input_type_key] += 1
    if verdict not in ("SAFE",):
        _threat_count[input_type_key] += 1

    remediation_text = ""
    if explain and eval_res.get("requires_advisory"):
        from shieldedge.llama_runner import LlamaAdvisoryRunner
        remediation_text = LlamaAdvisoryRunner().template_advisory(eval_res)["advisory"]

    logger.log_alert(
        input_type=eval_res["input_type"],
        raw_input=raw_result.get("input", ""),
        verdict=verdict,
        confidence=conf,
        remediation=remediation_text
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    res = {
        "verdict": verdict,
        "confidence": conf,
        "input_type": eval_res["input_type"],
        "signals": eval_res.get("signals", []),
        "summary": eval_res.get("input_summary", ""),
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if explain:
        res["advisory"] = remediation_text
    return res


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check — reports model availability."""
    return jsonify({
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "models": {
            "url_model": os.path.exists(URL_MODEL),
            "sms_pipeline": os.path.exists(SMS_MODEL),
            "network_iforest": os.path.exists(NET_MODEL),
        },
        "version": "3.0.0",
        "advisory": "template playbook (use the CLI for streamed Llama explanations)"
    })


@app.route("/analyze/url", methods=["POST"])
def analyze_url():
    """
    Analyze a URL for phishing.
    Body: {"url": "https://example.com", "explain": false}
    """
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        abort(400, description="Missing required field: 'url'")
    if len(url) > 2048:
        abort(400, description="URL too long (max 2048 chars)")

    try:
        raw = _get_url_det().predict(url)
    except Exception:
        abort(400, description="Could not parse URL")
    response = _run_pipeline(raw, "url", explain=bool(data.get("explain", False)))
    response["features"] = raw.get("features", {})
    return jsonify(response)


@app.route("/analyze/sms", methods=["POST"])
def analyze_sms():
    """
    Analyze an SMS for scam/smishing.
    Body: {"message": "Your account is blocked...", "explain": false}
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        abort(400, description="Missing required field: 'message'")
    if len(message) > 1600:
        abort(400, description="Message too long (max 1600 chars)")

    raw = _get_sms_det().predict(message)
    response = _run_pipeline(raw, "sms", explain=bool(data.get("explain", False)))
    return jsonify(response)


@app.route("/analyze/flow", methods=["POST"])
def analyze_flow():
    """
    Analyze a network flow for anomalies.
    Body: {"dst_port": 4444, "byte_rate": 2000000, "packet_rate": 600, "explain": false}
    """
    data = request.get_json(silent=True) or {}
    try:
        flow = {
            "dst_port":    int(data.get("dst_port", 0)),
            "byte_rate":   float(data.get("byte_rate", 0)),
            "packet_rate": float(data.get("packet_rate", 0)),
        }
    except (TypeError, ValueError):
        abort(400, description="Invalid flow parameters. Expected: dst_port (int), byte_rate (float), packet_rate (float)")

    raw = _get_net_det().predict_flow(flow)
    response = _run_pipeline(raw, "flow", explain=bool(data.get("explain", False)))
    return jsonify(response)


@app.route("/alerts", methods=["GET"])
def get_alerts():
    """
    Return recent threat alerts from local SQLite log.
    Query params: ?limit=20
    """
    try:
        limit = int(request.args.get("limit", 20))
        limit = max(1, min(limit, 100))
    except ValueError:
        limit = 20

    logger = _get_logger()
    rows = logger.get_recent_alerts(limit)

    alerts = []
    for row in rows:
        alerts.append({
            "id":         row[0],
            "timestamp":  row[1],
            "type":       row[2],
            "verdict":    row[3],
            "confidence": row[4],
        })

    return jsonify({
        "count": len(alerts),
        "alerts": alerts
    })


@app.route("/stats", methods=["GET"])
def stats():
    """Return detection statistics for this server session."""
    total_req = sum(_request_count.values())
    total_threats = sum(_threat_count.values())
    return jsonify({
        "uptime_seconds": round(time.time() - _start_time, 1),
        "total_requests": total_req,
        "total_threats_detected": total_threats,
        "threat_rate_pct": round(total_threats / max(1, total_req) * 100, 1),
        "breakdown": {
            "url":  {"requests": _request_count["url"],  "threats": _threat_count["url"]},
            "sms":  {"requests": _request_count["sms"],  "threats": _threat_count["sms"]},
            "flow": {"requests": _request_count["flow"], "threats": _threat_count["flow"]},
        }
    })


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "message": str(e.description)}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "message": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal Server Error"}), 500


def create_app():
    return app


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"[ShieldEdge API] Starting on http://localhost:{port}")
    print(f"[ShieldEdge API] Endpoints: /health, /analyze/url, /analyze/sms, /analyze/flow, /alerts, /stats")
    # Local-only by default: bind to loopback so other devices cannot query it.
    app.run(host="127.0.0.1", port=port, debug=False)
