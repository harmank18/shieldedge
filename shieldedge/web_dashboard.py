"""
ShieldEdge web dashboard: a local, offline control room for the device.

    python shieldedge.py --web            # http://127.0.0.1:8080
    ssh -L 8080:localhost:8080 raspi5     # view the Pi's dashboard from a laptop

What it shows
  * Live feed: every item analysed by the real detectors, colour-coded by verdict.
    By default a *labelled replay* feeds it: real CIC-IDS-2017 Friday flows in a
    scripted scenario (normal -> port scan -> DDoS -> botnet) and held-out URLs/SMS
    never used for training (data/samples/replay_*.csv). Replay items carry their
    dataset label, so agreement with the ground truth is visible (mistakes too).
  * Threat detail: signals, then the Llama 3.2 1B explanation streamed live from
    ExecuTorch, plus the vetted playbook steps. Llama runs on demand, one at a time.
  * Network chart: flows and anomalies per second.
  * Try it: analyse any URL or SMS typed into the page (logged as a hash only).
  * Device: per-core CPU load, temperature, RAM, whether llama_main is running and
    how many external network connections this process holds (expected: 0).
  * Model quality: held-out metrics read from results/, and the local audit log.

Privacy: binds to 127.0.0.1 only. Typed inputs live in memory for display; the
SQLite audit log stores SHA-256 hashes, as everywhere else in ShieldEdge.
"""

import ipaddress
import itertools
import json
import os
import queue
import socket
import threading
import time
from collections import Counter, deque

import pandas as pd
import psutil
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from shieldedge.advisory_playbook import remediation_steps
from shieldedge.decision_engine import THREAT_VERDICTS, DecisionEngine
from shieldedge.device import device_info, device_slug
from shieldedge.llama_runner import LlamaAdvisoryRunner
from shieldedge.network_detector import NetworkDetector, cic_to_features
from shieldedge.prompt_constructor import construct_prompt
from shieldedge.sms_detector import SMSDetector
from shieldedge.storage import AlertLogger
from shieldedge.url_detector import URLDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
MODELS = os.path.join(BASE_DIR, "models")
SAMPLES = os.path.join(BASE_DIR, "data", "samples")
RESULTS = os.path.join(BASE_DIR, "results")

FLOW_INTERVAL_S = 0.2     # replay: 5 flows per second
MESSAGE_INTERVAL_S = 2.5  # replay: one URL or SMS every 2.5 s
MAX_TEXT = 1600

KIND = {"URL": "URL", "SMS": "SMS", "NETWORK_FLOW": "NETWORK"}
ATTACK_LABELS = {"phishing", "scam"}  # message labels that count as threats


def is_threat_label(label):
    if label is None:
        return None
    lab = str(label)
    return lab.lower() in ATTACK_LABELS or (lab not in ("benign", "genuine") and lab.upper() != "BENIGN")


def sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


class Hub:
    """In-memory event store + fan-out to connected browsers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events = deque(maxlen=500)
        self.by_id = {}
        self.ids = itertools.count(1)
        self.clients = set()
        self.counts = {k: Counter() for k in ("URL", "SMS", "NETWORK")}
        self.agreement = {k: Counter() for k in ("URL", "SMS", "NETWORK")}  # replay vs dataset label
        self.seconds = {}  # unix second -> [flows, anomalies]
        self.phase = ""
        self.replay_on = True
        self.started = time.time()

    def publish(self, ev: dict):
        with self.lock:
            self.events.append(ev)
            self.by_id[ev["id"]] = ev
            if len(self.by_id) > 700:
                keep = {e["id"] for e in self.events}
                self.by_id = {i: e for i, e in self.by_id.items() if i in keep}
            self.counts[ev["kind"]][ev["verdict"]] += 1
            if ev["label"] is not None:
                truth, flagged = is_threat_label(ev["label"]), ev["verdict"] != "SAFE"
                self.agreement[ev["kind"]]["correct" if truth == flagged else "wrong"] += 1
            if ev["kind"] == "NETWORK":
                sec = int(ev["ts"])
                b = self.seconds.setdefault(sec, [0, 0])
                b[0] += 1
                b[1] += ev["verdict"] != "SAFE"
                if len(self.seconds) > 400:
                    for s in sorted(self.seconds)[:-300]:
                        del self.seconds[s]
            for q in list(self.clients):
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass


def public(ev: dict) -> dict:
    """Event fields sent to the browser (evidence stays server-side)."""
    return {k: ev[k] for k in ("id", "ts", "time", "kind", "source", "display", "verdict", "confidence",
                               "signals", "label", "dataset", "phase", "threat", "advisory", "alert_id", "steps", "caveat")}


class Dashboard:
    def __init__(self, replay: bool = True):
        self.url = URLDetector(os.path.join(MODELS, "url_model.joblib"))
        self.sms = SMSDetector(os.path.join(MODELS, "sms_pipeline.joblib"))
        self.net = NetworkDetector(os.path.join(MODELS, "network_model.joblib"))
        self.engine = DecisionEngine()
        self.runner = LlamaAdvisoryRunner()
        self.logger = AlertLogger()
        self.hub = Hub()
        self.hub.replay_on = replay
        self.llm_lock = threading.Lock()
        self.llm_busy_id = None
        psutil.cpu_percent(percpu=True)  # prime the CPU counters

    # ------------------------------------------------------------------ analysis
    def make_event(self, det: dict, source: str, display: str, label=None, dataset=None, phase=None) -> dict:
        evidence = self.engine.evaluate(det)
        now = time.time()
        return {
            "id": next(self.hub.ids), "ts": now, "time": time.strftime("%H:%M:%S", time.localtime(now)),
            "kind": KIND.get(evidence["input_type"], evidence["input_type"]), "source": source,
            "display": display, "verdict": evidence["verdict"], "confidence": evidence["confidence"],
            "signals": evidence["signals"], "label": label, "dataset": dataset, "phase": phase,
            "threat": evidence["verdict"] in THREAT_VERDICTS, "advisory": evidence["requires_advisory"],
            "alert_id": None, "evidence": evidence,
            "steps": remediation_steps(evidence["input_type"], evidence["verdict"]) if evidence["requires_advisory"] else [],
            "caveat": evidence["caveat"] if evidence["requires_advisory"] else "",
        }

    def analyse_text(self, text: str) -> dict:
        text = text.strip()
        looks_like_url = " " not in text and "\n" not in text and "." in text and len(text) <= 2048
        if looks_like_url:
            det, display = self.url.predict(text), text
        else:
            det, display = self.sms.predict(text), text
        ev = self.make_event(det, "you", display)
        if ev["verdict"] != "SAFE":
            ev["alert_id"] = self.logger.log_alert(ev["evidence"]["input_type"], text, ev["verdict"],
                                                   ev["confidence"])
        self.hub.publish(ev)
        return ev

    # ------------------------------------------------------------------ replay
    def replay_loop(self):
        flows_path = os.path.join(SAMPLES, "replay_flows.csv")
        msgs_path = os.path.join(SAMPLES, "replay_messages.csv")
        if not (os.path.exists(flows_path) and os.path.exists(msgs_path)):
            print("[dashboard] replay files missing - run data/build_dashboard_replay.py; replay disabled")
            self.hub.replay_on = False
            return
        raw = pd.read_csv(flows_path, low_memory=False)
        feats = cic_to_features(raw)
        labels, phases = raw["Label"].astype(str).values, raw["Phase"].astype(str).values
        msgs = pd.read_csv(msgs_path)
        i = j = 0
        next_msg = time.time()
        while True:
            if not self.hub.replay_on:
                time.sleep(0.3)
                continue
            t0 = time.time()
            det = self.net.predict_frame(feats.iloc[i:i + 1], source="replay")[0]
            self.hub.phase = phases[i]
            self.hub.publish(self.make_event(det, "replay", det["flow_summary"], label=labels[i],
                                             dataset="CIC-IDS-2017 Friday", phase=phases[i]))
            i = (i + 1) % len(raw)
            if time.time() >= next_msg:
                m = msgs.iloc[j]
                det = self.url.predict(m.text) if m.kind == "URL" else self.sms.predict(m.text)
                self.hub.publish(self.make_event(det, "replay", str(m.text), label=m.label, dataset=m.source))
                j = (j + 1) % len(msgs)
                next_msg = time.time() + MESSAGE_INTERVAL_S
            time.sleep(max(0.0, FLOW_INTERVAL_S - (time.time() - t0)))

    # ------------------------------------------------------------------ device
    def device(self) -> dict:
        info = device_info()
        temp = None
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temp = int(f.read().strip()) / 1000
        except (OSError, ValueError):
            pass
        mem = psutil.virtual_memory()
        llama = [p for p in psutil.process_iter(["name", "memory_info"]) if p.info["name"] == "llama_main"]
        me = psutil.Process()
        try:
            conns = me.net_connections(kind="inet")
        except AttributeError:  # psutil < 6
            conns = me.connections(kind="inet")

        def external(c):
            if not c.raddr:
                return False
            try:
                return not ipaddress.ip_address(c.raddr.ip).is_loopback
            except ValueError:
                return True

        stats = psutil.net_if_stats()
        ifaces = [n for n, addrs in psutil.net_if_addrs().items()
                  if n != "lo" and stats.get(n) and stats[n].isup and any(a.family == socket.AF_INET for a in addrs)]
        return {
            "device": info["device"], "arch": info["arch"], "ram_gb": info["ram_gb"],
            "cpu": psutil.cpu_percent(percpu=True), "temp_c": temp,
            "ram_used_gb": round((mem.total - mem.available) / 2**30, 2), "ram_total_gb": round(mem.total / 2**30, 1),
            "llama_running": bool(llama),
            "llama_rss_mb": round(llama[0].info["memory_info"].rss / 2**20) if llama and llama[0].info["memory_info"] else None,
            "app_rss_mb": round(me.memory_info().rss / 2**20),
            "external_connections": sum(external(c) for c in conns),
            "interfaces_up": ifaces,
            "llama_model": self.runner.model_name if self.runner.has_executorch else None,
        }

    def quality(self) -> dict:
        out = {"detection": [], "llm": None, "prompt_eval": None}
        p = os.path.join(RESULTS, "evaluation_results.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            out["detection"] = [
                # F1 from the raw counts: the CSV value is already rounded to 4 places and would round again.
                {"task": r.task, "dataset": r.dataset, "n": int(r.n), "precision": r.precision, "recall": r.recall,
                 "f1": 2 * r.tp / max(1, 2 * r.tp + r.fp + r.fn), "fpr": r.fpr} for r in d.itertuples()]
        slug = device_slug()
        for name in (f"llm_benchmark_{slug}_llama3_2_1b_int4.json", "llm_benchmark_raspberry-pi-5_llama3_2_1b_int4.json"):
            q = os.path.join(RESULTS, name)
            if os.path.exists(q):
                b = json.load(open(q))
                s = b["summary"]
                out["llm"] = {"device": b["device"]["device"], "threads": b.get("cpu_threads"),
                              "ttft_ms": s["ttft_ms"]["mean"], "decode_tok_s": s["decode_tok_s"]["mean"],
                              "prefill_tok_s": s["prefill_tok_s"]["mean"], "generation_ms": s["generation_ms"]["mean"],
                              "peak_rss_mb": s["peak_rss_mb"]["mean"], "runs": b["runs"]}
                break
        q = os.path.join(RESULTS, f"prompt_eval_{slug}.json")
        if os.path.exists(q):
            out["prompt_eval"] = json.load(open(q))["summary"]
        return out


def create_app(dash: Dashboard) -> Flask:
    app = Flask(__name__, static_folder=None)
    hub = dash.hub

    @app.after_request
    def no_cache(resp):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.route("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.route("/static/<path:name>")
    def static_files(name):
        return send_from_directory(WEB_DIR, name)

    @app.route("/api/events")
    def events():
        q = queue.Queue(maxsize=1000)
        with hub.lock:
            snapshot = list(hub.events)[-80:]
            hub.clients.add(q)

        def stream():
            try:
                yield "retry: 2000\n\n"
                for ev in snapshot:
                    yield sse(public(ev))
                while True:
                    try:
                        yield sse(public(q.get(timeout=15)))
                    except queue.Empty:
                        yield ": ping\n\n"
            finally:
                with hub.lock:
                    hub.clients.discard(q)

        return Response(stream(), mimetype="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.route("/api/event/<int:eid>")
    def event(eid):
        ev = hub.by_id.get(eid)
        if not ev:
            abort(404)
        return jsonify(public(ev))

    @app.route("/api/explain/<int:eid>")
    def explain(eid):
        ev = hub.by_id.get(eid)
        if not ev:
            abort(404)
        evidence = ev["evidence"]

        def stream():
            if not evidence["requires_advisory"]:
                yield sse({"type": "error", "message": "Only flagged items get an explanation."})
                return
            if not dash.llm_lock.acquire(blocking=False):
                yield sse({"type": "busy", "id": dash.llm_busy_id})
                return
            dash.llm_busy_id = eid
            q = queue.Queue()

            def work():
                try:
                    res = dash.runner.generate_advisory(evidence, construct_prompt(evidence),
                                                        stream=lambda t: q.put(("token", t)))
                    q.put(("done", res))
                except Exception as exc:  # never crash the dashboard because of the runner
                    q.put(("error", str(exc)))
                finally:
                    dash.llm_busy_id = None
                    dash.llm_lock.release()

            threading.Thread(target=work, daemon=True).start()
            yield sse({"type": "start", "llm": dash.runner.has_executorch, "model": dash.runner.model_name})
            while True:
                kind, val = q.get()
                if kind == "token":
                    yield sse({"type": "token", "text": val})
                elif kind == "done":
                    yield sse({"type": "done", "explanation": val["explanation"], "source": val["source"],
                               "remediation": val["remediation"], "metrics": val["metrics"],
                               "validation": val["validation"], "caveat": evidence.get("caveat", "")})
                    return
                else:
                    yield sse({"type": "error", "message": val})
                    return

        return Response(stream(), mimetype="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            abort(400, description="Type a link or an SMS first.")
        if len(text) > MAX_TEXT:
            abort(400, description=f"Too long (max {MAX_TEXT} characters).")
        return jsonify(public(dash.analyse_text(text)))

    @app.route("/api/stats")
    def stats():
        with hub.lock:
            now = int(time.time())
            timeline = [[s, *hub.seconds.get(s, [0, 0])] for s in range(now - 120, now)]
            return jsonify({
                "counts": {k: dict(v) for k, v in hub.counts.items()},
                "agreement": {k: dict(v) for k, v in hub.agreement.items()},
                "timeline": timeline, "phase": hub.phase, "replay_on": hub.replay_on,
                "uptime_s": round(time.time() - hub.started), "llm_busy_id": dash.llm_busy_id,
            })

    @app.route("/api/replay", methods=["POST"])
    def replay():
        hub.replay_on = bool((request.get_json(silent=True) or {}).get("on", not hub.replay_on))
        return jsonify({"replay_on": hub.replay_on})

    @app.route("/api/device")
    def device():
        return jsonify(dash.device())

    @app.route("/api/quality")
    def quality():
        return jsonify(dash.quality())

    @app.route("/api/alerts")
    def alerts():
        rows = dash.logger.get_recent_alerts(12)
        return jsonify({
            "alerts": [{"id": r[0], "time": r[1], "type": r[2], "verdict": r[3], "confidence": r[4],
                        "marked_safe": bool(r[5])} for r in rows],
            "feedback": dash.logger.feedback_stats(),
        })

    @app.route("/api/alerts/<int:aid>/safe", methods=["POST"])
    def mark_safe(aid):
        return jsonify({"ok": dash.logger.mark_false_positive(aid)})

    @app.route("/api/alerts/clear", methods=["POST"])
    def clear():
        return jsonify({"deleted": dash.logger.clear()})

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": str(e.description)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "not found"}), 404

    return app


def run(host: str = "127.0.0.1", port: int = 8080, replay: bool = True):
    dash = Dashboard(replay=replay)
    threading.Thread(target=dash.replay_loop, daemon=True).start()
    app = create_app(dash)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[dashboard] WARNING: binding to {host} exposes the dashboard to the network.")
    print(f"[dashboard] ShieldEdge dashboard on http://{host}:{port}  "
          f"(from a laptop: ssh -L {port}:localhost:{port} <pi> then open http://localhost:{port})")
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
