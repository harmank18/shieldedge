"""
Benchmark the on-device Llama advisory (ExecuTorch llama_main) and label it with the device.

For each run: model load time, time to first token, prefill and decode
throughput, total generation time (from ExecuTorch's PyTorchObserver), wall
time, and peak resident memory of the llama_main process (sampled every 20 ms).
Runs cycle through 5 fixed evidence cases (URL, SMS, Hinglish SMS, network,
suspicious URL) and also record whether the explanation passed validation.

Usage:
  python scripts/benchmark_llm.py                 # 10 runs, INT4 program
  python scripts/benchmark_llm.py --runs 20 --pte executorch/llama3_2_1b_bf16.pte
Outputs results/llm_benchmark_<device>_<program>.json and .md
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import psutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shieldedge.decision_engine import DecisionEngine  # noqa: E402
from shieldedge.device import device_info, device_slug  # noqa: E402
from shieldedge.llama_runner import (ASSISTANT_MARKER, ET_DIR, MAX_NEW_TOKENS, LlamaAdvisoryRunner,  # noqa: E402
                                     clean_llm_text, parse_observer, validate_explanation)
from shieldedge.prompt_constructor import construct_prompt  # noqa: E402

CASES = [
    {"input_type": "URL", "input": "http://sbi-net-banking-update.xyz/login", "verdict": "PHISHING", "confidence": 0.99,
     "signals": ["Uses plain HTTP, so the connection is not encrypted",
                 "Uses the name 'SBI' but sbi-net-banking-update.xyz is not an official SBI website",
                 "Uses a cheap domain ending (.xyz) that is common in scams",
                 "Address contains words like login, verify, KYC or update"]},
    {"input_type": "SMS", "input": "x", "verdict": "SCAM", "confidence": 0.93,
     "signals": ["Pressure or threat words: blocked, 24 hours", "Mentions a bank, government or company: KYC, SBI",
                 "Contains a link to sbi-kyc-verify.top, which pretends to be SBI but is not an official SBI website"]},
    {"input_type": "SMS", "input": "x", "verdict": "SCAM", "confidence": 0.88,
     "signals": ["Hindi/Hinglish pressure phrases: band ho jayega, turant", "Asks for money to a UPI ID",
                 "Promises money, prizes or easy income: loan approved"]},
    {"input_type": "NETWORK_FLOW", "flow_summary": "TCP to port 9878, 0.000s, 2 pkts", "verdict": "ANOMALY",
     "confidence": 0.99, "signals": ["Very short connection that was reset immediately, typical of port scanning",
                                     "Traffic pattern is more unusual than 99.9% of normal flows"]},
    {"input_type": "URL", "input": "https://secure-update-account.weebly.com/", "verdict": "SUSPICIOUS",
     "confidence": 0.78, "signals": ["Hosted on a free website-builder or file-sharing platform that anyone can use",
                                     "Address contains words like login, verify, KYC or update"]},
]


def run_once(runner, prompt, max_new_tokens):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write(prompt)
        ppath = tf.name
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{ET_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [runner.executorch_runner_bin, "--model_path", runner.pte_model_path, "--tokenizer_path",
           runner.tokenizer_path, "--prompt_file", ppath, "--max_new_tokens", str(max_new_tokens), "--temperature", "0"]
    peak = [0]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)

    def sample():
        try:
            p = psutil.Process(proc.pid)
            while proc.poll() is None:
                peak[0] = max(peak[0], p.memory_info().rss)
                time.sleep(0.02)
        except psutil.Error:
            pass

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    out = proc.communicate()[0].decode("utf-8", errors="replace")
    wall = (time.perf_counter() - t0) * 1000
    th.join()
    os.remove(ppath)
    m = parse_observer(out) or {}
    m["wall_ms"] = wall
    m["peak_rss_mb"] = peak[0] / 2**20
    return m, out.split(ASSISTANT_MARKER)[-1].split("PyTorchObserver")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--pte", default=None, help="ExecuTorch program (default: same as the app)")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    runner = LlamaAdvisoryRunner(pte_model_path=args.pte)
    if not runner.has_executorch:
        sys.exit("llama_main, the .pte program or tokenizer.model is missing - see docs/PI_DEPLOYMENT.md")
    engine = DecisionEngine()
    info = device_info()
    print(f"Benchmarking {runner.model_name} on {info['device']} ({args.runs} runs, warm-up run discarded)")

    evs = [engine.evaluate(c) for c in CASES]
    run_once(runner, construct_prompt(evs[0]), 16)  # warm-up (page cache)

    runs = []
    for i in range(args.runs):
        ev = evs[i % len(evs)]
        m, gen = run_once(runner, construct_prompt(ev), args.max_new_tokens)
        text = clean_llm_text(gen)
        m["case"] = i % len(evs)
        m["valid"] = not validate_explanation(text, ev)
        m["explanation"] = text
        runs.append(m)
        print(f"  run {i + 1:>2}: TTFT {m['ttft_ms']:>6} ms | prefill {m['prefill_tok_s']:6.1f} tok/s | "
              f"decode {m['decode_tok_s']:5.1f} tok/s | RSS {m['peak_rss_mb']:6.0f} MB | valid={m['valid']}")

    keys = ["model_load_ms", "ttft_ms", "prefill_tok_s", "decode_tok_s", "generation_ms", "wall_ms", "peak_rss_mb",
            "prompt_tokens", "generated_tokens"]
    summary = {k: {"mean": statistics.mean(r[k] for r in runs), "std": statistics.stdev(r[k] for r in runs)}
               for k in keys}
    result = {
        "device": info, "program": runner.model_name,
        "program_size_mb": round(os.path.getsize(runner.pte_model_path) / 2**20),
        "max_new_tokens": args.max_new_tokens, "runs": len(runs), "date": time.strftime("%Y-%m-%d %H:%M"),
        "validation_pass_rate": sum(r["valid"] for r in runs) / len(runs),
        "summary": summary, "raw": runs,
    }
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    stem = os.path.join(BASE_DIR, "results", f"llm_benchmark_{device_slug()}_{runner.model_name.replace('.pte', '')}")
    with open(stem + ".json", "w") as f:
        json.dump(result, f, indent=2)

    lines = [f"# Llama 3.2 1B-Instruct on {info['device']}", "",
             f"Program `{runner.model_name}` ({result['program_size_mb']} MB), ExecuTorch `llama_main` + XNNPACK, "
             f"greedy decoding, up to {args.max_new_tokens} new tokens, {len(runs)} runs after one warm-up "
             f"({result['date']}, {info['arch']}, {info['cpus']} CPUs, {info['ram_gb']} GB RAM).", "",
             "| Metric | Mean | Std |", "|---|---|---|"]
    for k in keys:
        lines.append(f"| {k} | {summary[k]['mean']:.1f} | {summary[k]['std']:.1f} |")
    lines += ["", f"Explanations passing validation: {result['validation_pass_rate']:.0%}"]
    with open(stem + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nMean TTFT {summary['ttft_ms']['mean']:.0f} ms, decode {summary['decode_tok_s']['mean']:.1f} tok/s, "
          f"peak RSS {summary['peak_rss_mb']['mean']:.0f} MB -> {stem}.json/.md")


if __name__ == "__main__":
    main()
