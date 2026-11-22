"""
ShieldEdge evaluation suite - evaluates the *deployed* decision path
(model + rule layer + thresholds, exactly what the CLI shows) on held-out data.

Positive class = any warning shown to the user (verdict != SAFE).
ROC-AUC uses the detector's continuous score.

  URL      Mitake held-out split (domain-grouped, from train_url_classifier.py)
           + curated India URL challenge set (evaluation only)
  SMS      UCI held-out split (deduplicated, from train_sms_classifier.py)
           + team-authored India SMS test set (evaluation only)
  Network  CIC-IDS-2017 Friday (benign + attacks), model trained on Monday benign;
           per-attack-category detection rates reported

Writes results/evaluation_results.csv and results/benchmarks.md (labelled with the device).
Run after the three training scripts:  python models/evaluate_models.py
"""

import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shieldedge.device import device_info  # noqa: E402
from shieldedge.network_detector import NetworkDetector  # noqa: E402
from shieldedge.sms_detector import SMSDetector  # noqa: E402
from shieldedge.url_detector import URLDetector  # noqa: E402

MODELS = os.path.join(BASE_DIR, "models")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
CURATED = os.path.join(BASE_DIR, "data", "curated")
RESULTS = os.path.join(BASE_DIR, "results")


def metrics(task, dataset, y, verdicts, scores, note=""):
    y = np.asarray(y).astype(bool)
    flag = np.asarray(verdicts) != "SAFE"
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    fn, tn = int((~flag & y).sum()), int((~flag & ~y).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "task": task, "dataset": dataset, "n": len(y), "positives": int(y.sum()),
        "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(1e-9, prec + rec), "fpr": fp / max(1, fp + tn),
        "roc_auc": roc_auc_score(y, scores) if 0 < y.sum() < len(y) else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "note": note,
    }


def latency_ms(fn, arg, n=200):
    for _ in range(10):
        fn(arg)
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn(arg)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.mean(times), statistics.stdev(times)


def main():
    rows, extra = [], {}
    url_det = URLDetector(os.path.join(MODELS, "url_model.joblib"))
    sms_det = SMSDetector(os.path.join(MODELS, "sms_pipeline.joblib"))
    net_det = NetworkDetector(os.path.join(MODELS, "network_model.joblib"))

    # ---------------- URL
    print("URL: held-out Mitake split ...")
    t = pd.read_csv(os.path.join(PROCESSED, "url_test.csv"))
    res = url_det.predict_batch(t.url.tolist())
    rows.append(metrics("URL phishing", "Mitake held-out (unseen domains)", t.label == 1,
                        [r["verdict"] for r in res], [r["score"] for r in res]))
    base_path = os.path.join(MODELS, "url_model_baseline.joblib")
    if os.path.exists(base_path):  # same split, previous configuration (train_url_classifier.py --baseline)
        res_b = URLDetector(base_path).predict_batch(t.url.tolist())
        rows.append(metrics("URL phishing", "Mitake held-out, baseline (char-only, half data)", t.label == 1,
                            [r["verdict"] for r in res_b], [r["score"] for r in res_b], "ablation"))
    c = pd.read_csv(os.path.join(CURATED, "india_urls.csv"))
    res = url_det.predict_batch(c.url.tolist())
    rows.append(metrics("URL phishing", "India challenge set (curated)", c.label == "phishing",
                        [r["verdict"] for r in res], [r["score"] for r in res],
                        "phishing rows are team-authored; indicative only"))

    # ---------------- SMS
    print("SMS: UCI held-out, real smishing, Indian synthetic, India team test ...")
    sms_sets = [
        (os.path.join(PROCESSED, "sms_test.csv"), "UCI SMS Spam held-out (deduplicated)", ""),
        (os.path.join(PROCESSED, "sms_test_smishing.csv"), "Real smishing 2022-25 held-out (IMC reports + Mishra & Soni)",
         "links/numbers masked in the source data"),
        (os.path.join(PROCESSED, "sms_test_indian_synthetic.csv"), "Indian SMS, AI-generated, 7 languages (held-out 40%)",
         "synthetic; same generator as the 40% training part"),
        (os.path.join(CURATED, "india_sms_test.csv"), "India SMS test set (team-authored)", "team-authored; indicative only"),
    ]
    base_sms = os.path.join(MODELS, "sms_pipeline_baseline.joblib")
    sms_base = SMSDetector(base_sms) if os.path.exists(base_sms) else None
    for path, name, note in sms_sets:
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        res = [sms_det.predict(m) for m in d.message.astype(str)]
        rows.append(metrics("SMS scam", name, d.label == "spam", [r["verdict"] for r in res],
                            [r["score"] for r in res], note))
        if sms_base is not None:  # previous model (UCI + team data only), same test set
            res_b = [sms_base.predict(m) for m in d.message.astype(str)]
            rows.append(metrics("SMS scam", name + ", previous model (baseline)", d.label == "spam",
                                [r["verdict"] for r in res_b], [r["score"] for r in res_b], "ablation"))

    # ---------------- Network
    print("Network: CIC-IDS-2017 Friday ...")
    n = pd.read_csv(os.path.join(PROCESSED, "net_test.csv"))
    res = net_det.predict_frame(n)
    v = np.array([r["verdict"] for r in res])
    rows.append(metrics("Network anomaly", "CIC-IDS-2017 Friday (train: Monday benign)", n.label != "BENIGN",
                        v, [r["score"] for r in res]))
    extra["network_detection_rate_by_label"] = {
        lab: round(float((v[n.label.values == lab] != "SAFE").mean()), 4) for lab in sorted(n.label.unique())}
    # Baseline: the previous Isolation Forest, refitted on the same benign data with the same
    # holdout-percentile thresholds (98th percentile = any warning).
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from shieldedge.network_detector import FLOW_FEATURE_NAMES, log1p_clip
    tr = pd.read_csv(os.path.join(PROCESSED, "net_train.csv"))
    ho = pd.read_csv(os.path.join(PROCESSED, "net_holdout.csv"))
    iso = make_pipeline(FunctionTransformer(log1p_clip), StandardScaler(),
                        IsolationForest(n_estimators=200, max_samples=512, random_state=42, n_jobs=-1))
    iso.fit(tr[FLOW_FEATURE_NAMES].values)
    thr = np.quantile(-iso.score_samples(ho[FLOW_FEATURE_NAMES].values), 0.98)
    s_iso = -iso.score_samples(n[FLOW_FEATURE_NAMES].fillna(tr[FLOW_FEATURE_NAMES].median()).values)
    v_iso = np.where(s_iso >= thr, "ANOMALY", "SAFE")
    rows.append(metrics("Network anomaly", "CIC-IDS-2017 Friday, Isolation Forest baseline", n.label != "BENIGN",
                        v_iso, s_iso, "previous model, same data and thresholds"))
    extra["network_detection_rate_by_label_iforest"] = {
        lab: round(float((v_iso[n.label.values == lab] != "SAFE").mean()), 4) for lab in sorted(n.label.unique())}

    # ---------------- per-input latency of the deployed path
    lat = {
        "URL": latency_ms(url_det.predict, "http://sbi-net-banking-update.xyz/login"),
        "SMS": latency_ms(sms_det.predict, "Your SBI account is blocked. Update KYC at http://sbi-kyc.top"),
        "Network (1 flow)": latency_ms(net_det.predict_flow, {"dst_port": 443, "byte_rate": 5000, "packet_rate": 10}),
    }

    info = device_info()
    os.makedirs(RESULTS, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "device", info["device"])
    df.to_csv(os.path.join(RESULTS, "evaluation_results.csv"), index=False, float_format="%.4f")

    # ---------------- report
    print("\n" + df[["task", "dataset", "n", "precision", "recall", "f1", "fpr", "roc_auc"]].to_string(index=False))
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    md = [
        "# ShieldEdge evaluation results",
        "",
        f"Generated by `python models/evaluate_models.py` on **{info['device']}** ({info['arch']}, {info['cpus']} CPUs, "
        f"{info['ram_gb']} GB RAM, {info['os']}) on {time.strftime('%Y-%m-%d')}.",
        "",
        "All numbers come from the deployed decision path (model + rules + thresholds). Positive = any warning "
        "(verdict is not SAFE). None of the test data was used for training or threshold selection.",
        "",
        "## Detection quality",
        "",
        "| Task | Test set | Samples (threats) | Precision | Recall | F1 | FPR | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['task']} | {r['dataset']} | {r['n']:,} ({r['positives']:,}) | {pct(r['precision'])} | "
                  f"{pct(r['recall'])} | {r['f1']:.3f} | {pct(r['fpr'])} | {r['roc_auc']:.3f} |")
    md += ["", "Curated India sets are small and partly team-authored: treat them as sanity checks, not benchmarks.",
           "", "### Network: detection rate per CIC-IDS-2017 label (Friday)", "",
           "| Label | kNN novelty (deployed) | Isolation Forest (baseline) |", "|---|---|---|"]
    for lab, rate in extra["network_detection_rate_by_label"].items():
        md.append(f"| {lab} | {pct(rate)} | {pct(extra['network_detection_rate_by_label_iforest'][lab])} |")
    md += ["", "DDoS and botnet C2 flows have ordinary values feature by feature, so the Isolation Forest's "
               "axis-aligned splits miss them; no benign flow has the same combination of values, which the "
               "nearest-neighbour distance captures. Botnet detection relies partly on its unusual service port "
               "(8080), which is specific to this testbed.",
           "", "## Detection latency (one input, deployed path)", "",
           "| Detector | Mean (ms) | Std (ms) |", "|---|---|---|"]
    for k, (m, s) in lat.items():
        md.append(f"| {k} | {m:.2f} | {s:.2f} |")
    md += ["", "200 timed calls after 10 warm-up calls, single Python process."]

    llm_files = sorted(glob.glob(os.path.join(RESULTS, "llm_benchmark_*.json")))
    if llm_files:
        md += ["", "## On-device Llama (ExecuTorch)", "",
               "| Device | Program | Threads | Load (ms) | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Explanation total (ms) | Peak RSS (MB) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for path in llm_files:
            b = json.load(open(path))
            s = b["summary"]
            f = lambda k: f"{s[k]['mean']:.1f} ± {s[k]['std']:.1f}"  # noqa: E731
            md.append(f"| {b['device']['device']} | {b['program']} | {b.get('cpu_threads', 'runner default')} | {f('model_load_ms')} | {f('ttft_ms')} | "
                      f"{f('prefill_tok_s')} | {f('decode_tok_s')} | {f('generation_ms')} | {f('peak_rss_mb')} |")
        md += ["", "From `scripts/benchmark_llm.py` (mean ± std over all runs; see the JSON files for raw data)."]
    with open(os.path.join(RESULTS, "benchmarks.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\nWrote {os.path.join(RESULTS, 'evaluation_results.csv')} and {os.path.join(RESULTS, 'benchmarks.md')}")


if __name__ == "__main__":
    main()
