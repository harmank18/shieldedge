"""
Unified command-line interface for ShieldEdge.

Detection (URL / SMS / network flow / QR) -> decision engine -> on-device
Llama explanation (ExecuTorch) + vetted remediation playbook -> local,
hash-only SQLite audit log.
"""

import argparse
import os
import platform
import statistics
import sys
import time

import psutil

from shieldedge.decision_engine import DecisionEngine
from shieldedge.device import device_name
from shieldedge.llama_runner import LlamaAdvisoryRunner
from shieldedge.network_detector import NetworkDetector
from shieldedge.prompt_constructor import construct_prompt
from shieldedge.sms_detector import SMSDetector
from shieldedge.storage import AlertLogger
from shieldedge.url_detector import URLDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_MODEL_PATH = os.path.join(BASE_DIR, "models", "url_model.joblib")
SMS_MODEL_PATH = os.path.join(BASE_DIR, "models", "sms_pipeline.joblib")
NET_MODEL_PATH = os.path.join(BASE_DIR, "models", "network_isolation_forest.joblib")

RED, YELLOW, GREEN, DIM, BOLD, RESET = "\033[91m", "\033[93m", "\033[92m", "\033[2m", "\033[1m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    RED = YELLOW = GREEN = DIM = BOLD = RESET = ""

BANNER = r"""
  ____  _     _      _     _ _____    _
 / ___|| |__ (_) ___| | __| | ____|__| | __ _  ___
 \___ \| '_ \| |/ _ \ |/ _` |  _| / _` |/ _` |/ _ \
  ___) | | | | |  __/ | (_| | |__| (_| | (_| |  __/
 |____/|_| |_|_|\___|_|\__,_|_____\__,_|\__, |\___|
                                        |___/
 On-device threat, phishing & scam detection  |  PS3, Bharat AI-SoC Challenge
"""


def verdict_label(verdict: str) -> str:
    if verdict in {"PHISHING", "SCAM", "ANOMALY"}:
        return f"{RED}{BOLD}[!] {verdict}{RESET}"
    if verdict == "SUSPICIOUS":
        return f"{YELLOW}{BOLD}[?] {verdict}{RESET}"
    return f"{GREEN}{BOLD}[OK] {verdict}{RESET}"


def run_pipeline(detection_result, engine, runner, logger, use_llm=True, quiet=False):
    t0 = time.perf_counter()
    ev = engine.evaluate(detection_result)
    t_detect = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 64)
    print(f"VERDICT:     {verdict_label(ev['verdict'])}")
    print(f"CONFIDENCE:  {ev['confidence'] * 100:.1f}%")
    print(f"INPUT:       {ev['input_type']}  {DIM}{ev['input_summary']}{RESET}")
    if ev["signals"]:
        print("\nSIGNALS:")
        for s in ev["signals"]:
            print(f"  - {s}")
    print("=" * 64)

    advisory_text = ""
    if ev["requires_advisory"] and not quiet:
        if use_llm and runner.has_executorch:
            print(f"\n{BOLD}Explanation{RESET} {DIM}(Llama 3.2 1B on-device via ExecuTorch, {runner.model_name}){RESET}")
            adv = runner.generate_advisory(ev, construct_prompt(ev), stream=lambda s: print(s, end="", flush=True))
            print()
            if not adv["source"].startswith("Llama"):
                print(f"{YELLOW}[fallback] {adv['source']}{RESET}")
                print(adv["explanation"])
        else:
            if use_llm:
                print(f"{DIM}(ExecuTorch model/runner not found - using template explanation){RESET}")
            adv = runner.template_advisory(ev)
            print(f"\n{BOLD}Explanation{RESET} {DIM}({adv['source']}){RESET}")
            print(adv["explanation"])

        print(f"\n{BOLD}What to do{RESET} {DIM}(vetted ShieldEdge playbook){RESET}")
        for i, step in enumerate(adv["remediation"], 1):
            print(f"  {i}. {step}")
        print(f"\n{DIM}Note: {ev['caveat']}{RESET}")
        advisory_text = adv["advisory"]

        m = adv.get("metrics")
        if m:
            print(f"\n{DIM}[ExecuTorch] load {m['model_load_ms']} ms | TTFT {m['ttft_ms']} ms | "
                  f"prefill {m['prefill_tok_s']:.1f} tok/s | decode {m['decode_tok_s']:.1f} tok/s | "
                  f"{m['prompt_tokens']}+{m['generated_tokens']} tokens{RESET}")
        print("=" * 64)

    alert_id = logger.log_alert(ev["input_type"], str(detection_result.get("input", "")),
                                ev["verdict"], ev["confidence"], advisory_text)
    rss = psutil.Process().memory_info().rss / 2**20
    print(f"{DIM}Detection latency {t_detect:.2f} ms | Python RSS {rss:.0f} MB | alert #{alert_id} "
          f"(mark as safe: --mark-safe {alert_id}){RESET}")
    return ev


def run_benchmark(url_det, sms_det, net_det, n=200):
    cases = {
        "URL": (url_det.predict, "http://sbi-net-banking-update.xyz/login"),
        "SMS": (sms_det.predict, "Your SBI account is blocked. Update KYC immediately at http://sbi-kyc.top"),
        "Network": (net_det.predict_flow, {"dst_port": 4444, "byte_rate": 2_000_000, "packet_rate": 800}),
    }
    print(f"\nDetection latency benchmark on: {device_name()} ({platform.machine()})")
    print(f"{n} timed calls per detector after 10 warm-up calls\n")
    print(f"{'Detector':<10} {'mean ms':>9} {'std ms':>9} {'p95 ms':>9}")
    for name, (fn, arg) in cases.items():
        for _ in range(10):
            fn(arg)
        times = []
        for _ in range(n):
            t = time.perf_counter()
            fn(arg)
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        print(f"{name:<10} {statistics.mean(times):9.3f} {statistics.stdev(times):9.3f} {times[int(0.95 * n)]:9.3f}")
    print(f"\nPython process RSS with all three detectors loaded: {psutil.Process().memory_info().rss / 2**20:.0f} MB")


def main():
    p = argparse.ArgumentParser(description="ShieldEdge: on-device threat, phishing & scam detection")
    g = p.add_argument_group("analysis")
    g.add_argument("--url", help="analyse a URL for phishing")
    g.add_argument("--sms", help="analyse SMS text for scam/smishing")
    g.add_argument("--flows", metavar="CSV", help="analyse a pre-captured flow CSV (CICFlowMeter/CICIDS columns), one verdict per row")
    g.add_argument("--flow", nargs=3, metavar=("DST_PORT", "BYTE_RATE", "PKT_RATE"),
                   help="quick single-flow check from 3 numbers (other features unknown; prefer --flows)")
    g.add_argument("--qr", help="decode a QR code image and analyse any URL in it")
    g.add_argument("--fast", "--no-llm", dest="fast", action="store_true", help="skip Llama; use the template explanation")
    g.add_argument("--max-rows", type=int, default=50, help="max rows printed for --flows (default 50)")
    t = p.add_argument_group("tools")
    t.add_argument("--alerts", action="store_true", help="show the recent local audit log")
    t.add_argument("--mark-safe", type=int, metavar="ID", help="mark alert ID as a false positive")
    t.add_argument("--clear-logs", action="store_true", help="delete all locally stored alerts")
    t.add_argument("--benchmark", action="store_true", help="detector latency benchmark")
    t.add_argument("--dashboard", action="store_true", help="terminal dashboard of the audit log")
    t.add_argument("--serve", action="store_true", help="start the local REST API")
    t.add_argument("--host", default="127.0.0.1", help="REST API bind address (default 127.0.0.1, local only)")
    t.add_argument("--port", type=int, default=5050, help="REST API port (default 5050)")
    args = p.parse_args()

    if len(sys.argv) == 1:
        print(BANNER)
        p.print_help()
        return
    print(BANNER)

    logger = AlertLogger()
    if args.alerts:
        rows = logger.get_recent_alerts(15)
        print(f"{'ID':<5} {'Time (UTC)':<20} {'Type':<13} {'Verdict':<11} {'Conf':>6}  Feedback")
        for r in rows:
            print(f"{r[0]:<5} {r[1]:<20} {r[2]:<13} {r[3]:<11} {r[4] * 100:5.1f}%  {'marked safe' if r[5] else ''}")
        fb = logger.feedback_stats()
        print(f"\nFlagged: {fb['flagged']} | marked safe by user: {fb['marked_safe_by_user']}")
        print("Only SHA-256 hashes of inputs are stored - never the raw SMS/URL text.")
        return
    if args.mark_safe is not None:
        ok = logger.mark_false_positive(args.mark_safe)
        print(f"Alert #{args.mark_safe} marked as safe." if ok else f"No alert with id {args.mark_safe}.")
        return
    if args.clear_logs:
        print(f"Deleted {logger.clear()} stored alerts.")
        return
    if args.dashboard:
        from shieldedge.dashboard import show_dashboard
        show_dashboard()
        return
    if args.serve:
        from shieldedge.api_server import app
        print(f"[ShieldEdge API] http://{args.host}:{args.port}  (POST /analyze/url|sms|flow, GET /alerts /stats /health)")
        app.run(host=args.host, port=args.port, debug=False)
        return

    url_det = URLDetector(URL_MODEL_PATH)
    sms_det = SMSDetector(SMS_MODEL_PATH)
    net_det = NetworkDetector(NET_MODEL_PATH)
    engine = DecisionEngine()
    runner = LlamaAdvisoryRunner()
    use_llm = not args.fast

    if args.benchmark:
        run_benchmark(url_det, sms_det, net_det)
    elif args.url:
        run_pipeline(url_det.predict(args.url), engine, runner, logger, use_llm)
    elif args.sms:
        run_pipeline(sms_det.predict(args.sms), engine, runner, logger, use_llm)
    elif args.flow:
        print(f"{YELLOW}Note: only port, byte rate and packet rate are known; for full analysis use --flows <csv>.{RESET}")
        flow = {"dst_port": int(args.flow[0]), "byte_rate": float(args.flow[1]), "packet_rate": float(args.flow[2])}
        run_pipeline(net_det.predict_flow(flow), engine, runner, logger, use_llm)
    elif args.flows:
        results = net_det.predict_csv(args.flows)
        flagged = [r for r in results if r["verdict"] != "SAFE"]
        has_truth = any("label" in r for r in results)
        print(f"\n{'Row':>5}  {'Verdict':<12} {'Conf':>6}  {'Dataset label' if has_truth else '':<20}Flow")
        for r in results[: args.max_rows]:
            truth = r.get("label", "")[:18]
            print(f"{r['row']:>5}  {verdict_label(r['verdict']):<12} {r['confidence'] * 100:5.1f}%  {truth:<20}{r['flow_summary']}")
        if len(results) > args.max_rows:
            print(f"  ... {len(results) - args.max_rows} more rows")
        print(f"\n{len(flagged)} of {len(results)} flows flagged.")
        for r in flagged:
            logger.log_alert("NETWORK_FLOW", r["input"], r["verdict"], r["confidence"])
        if flagged:
            worst = max(flagged, key=lambda r: r["confidence"])
            print(f"\nMost anomalous flow: row {worst['row']}")
            run_pipeline(worst, engine, runner, logger, use_llm)
    elif args.qr:
        from shieldedge.qr_scanner import QRScanner, print_scan_result
        scan = QRScanner(url_det).scan(args.qr)
        print_scan_result(scan)
        for r in scan.get("results", []):
            if r.get("is_url") and r.get("url_analysis", {}).get("verdict") in {"PHISHING", "SUSPICIOUS"}:
                run_pipeline(url_det.predict(r["raw_data"]), engine, runner, logger, use_llm)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
