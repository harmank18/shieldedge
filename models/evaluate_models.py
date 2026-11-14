"""
ShieldEdge: Unified Model Evaluation Suite
Evaluates all three on-device detector models on held-out test splits:
1. URL Phishing Detector (Random Forest) on real held-out test set
2. SMS Scam Detector (TF-IDF + Calibrated LinearSVC) on real held-out test set
3. Network Flow Anomaly Detector (Isolation Forest) on real CICIDS-2017 test set

Outputs benchmark tables to results/benchmarks.md.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from shieldedge.url_detector import extract_url_features, FEATURE_NAMES
from shieldedge.sms_detector import preprocess_text
from shieldedge.network_detector import FLOW_FEATURE_NAMES

URL_MODEL_PATH = os.path.join(BASE_DIR, "models", "url_rf_model.joblib")
SMS_MODEL_PATH = os.path.join(BASE_DIR, "models", "sms_pipeline.joblib")
NET_MODEL_PATH = os.path.join(BASE_DIR, "models", "network_isolation_forest.joblib")

URL_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "url_dataset.csv")
SMS_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "sms_dataset.csv")
NET_TEST_PATH = os.path.join(BASE_DIR, "data", "raw", "network_test.csv")
BENCHMARK_OUT = os.path.join(BASE_DIR, "results", "benchmarks.md")


def evaluate_url():
    print("Evaluating URL Phishing Detector...")
    model = joblib.load(URL_MODEL_PATH)
    df = pd.read_csv(URL_DATA_PATH)
    
    # Feature extraction on full dataset then use held-out test split (test_size=0.2, seed=42)
    feature_rows = []
    labels = []
    for _, row in df.iterrows():
        url = str(row["url"])
        label = 1 if row["label"] == "phishing" else 0
        feats = extract_url_features(url)
        feature_rows.append([feats[k] for k in FEATURE_NAMES])
        labels.append(label)

    X = np.array(feature_rows)
    y = np.array(labels)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    latency_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "task": "URL Phishing Detection",
        "model": "Random Forest (n=100, max_depth=10)",
        "dataset": "Mitake Benchmark + Curated (Held-out Test)",
        "test_samples": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "fpr": fpr,
        "auc": roc_auc_score(y_test, y_proba),
        "latency_ms": latency_ms
    }


def evaluate_sms():
    print("Evaluating SMS Scam Detector...")
    pipeline = joblib.load(SMS_MODEL_PATH)
    df = pd.read_csv(SMS_DATA_PATH)
    df["clean_msg"] = df["message"].astype(str).apply(preprocess_text)
    df["target"] = df["label"].apply(lambda x: 1 if str(x).strip().lower() == "spam" else 0)

    _, X_test, _, y_test = train_test_split(
        df["clean_msg"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    latency_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "task": "SMS Scam / Smishing Detection",
        "model": "TF-IDF + Calibrated LinearSVC",
        "dataset": "UCI SMS Spam + Indian Scams (Held-out Test)",
        "test_samples": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "fpr": fpr,
        "auc": roc_auc_score(y_test, y_proba),
        "latency_ms": latency_ms
    }


def evaluate_network():
    print("Evaluating Network Anomaly Detector...")
    model = joblib.load(NET_MODEL_PATH)
    df_test = pd.read_csv(NET_TEST_PATH)
    X_test = df_test[FLOW_FEATURE_NAMES].values
    y_test = df_test["is_anomaly"].values

    t0 = time.perf_counter()
    raw_preds = model.predict(X_test)
    scores = -model.score_samples(X_test)
    latency_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    y_pred = np.where(raw_preds == -1, 1, 0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "task": "Network Flow Anomaly Detection",
        "model": "Isolation Forest (contamination=0.03)",
        "dataset": "CICIDS-2017 (Monday Normal + Friday Attacks)",
        "test_samples": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "fpr": fpr,
        "auc": roc_auc_score(y_test, scores),
        "latency_ms": latency_ms
    }


def main():
    print("="*60)
    print("ShieldEdge Model Evaluation Suite")
    print("="*60)

    results = []
    results.append(evaluate_url())
    results.append(evaluate_sms())
    results.append(evaluate_network())

    print("\n" + "="*80)
    print(f"{'Task':<28} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'FPR':<6} | {'AUC':<6} | {'Lat (ms)'}")
    print("-" * 80)
    for r in results:
        print(f"{r['task']:<28} | {r['accuracy']*100:5.1f}% | {r['precision']*100:5.1f}% | {r['recall']*100:5.1f}% | {r['f1']*100:5.1f}% | {r['fpr']*100:5.1f}% | {r['auc']:5.3f} | {r['latency_ms']:.3f} ms")
    print("="*80)

    # Write results to benchmarks.md
    md_content = f"""# ShieldEdge Benchmark & Evaluation Results

> **Test Hardware:** Dell Inspiron 15 3520 (Intel Core i3/i5 12th Gen, x86_64, 15GB RAM, Ubuntu 24.04 LTS)  
> **Evaluation Mode:** Local on-device execution (100% offline, zero cloud calls)  
> **Date Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Classical Detection Models Evaluation

Evaluated strictly on held-out test splits not used during model training.

| Task | Model Architecture | Evaluation Dataset | Test Samples | Accuracy | Precision | Recall | F1-Score | False Positive Rate | ROC-AUC | Avg Latency / Sample |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **URL Phishing** | Random Forest (`n=100`, `depth=10`) | Mitake + Curated Phish (Held-out) | {results[0]['test_samples']:,} | {results[0]['accuracy']*100:.2f}% | {results[0]['precision']*100:.2f}% | {results[0]['recall']*100:.2f}% | {results[0]['f1']*100:.2f}% | {results[0]['fpr']*100:.2f}% | {results[0]['auc']:.4f} | {results[0]['latency_ms']:.3f} ms |
| **SMS Scam** | TF-IDF + Calibrated LinearSVC | UCI SMS Spam + Curated Indian Scams | {results[1]['test_samples']:,} | {results[1]['accuracy']*100:.2f}% | {results[1]['precision']*100:.2f}% | {results[1]['recall']*100:.2f}% | {results[1]['f1']*100:.2f}% | {results[1]['fpr']*100:.2f}% | {results[1]['auc']:.4f} | {results[1]['latency_ms']:.3f} ms |
| **Network Anomaly** | Isolation Forest (`contam=0.03`) | CICIDS-2017 (Monday Normal + Friday Attacks) | {results[2]['test_samples']:,} | {results[2]['accuracy']*100:.2f}% | {results[2]['precision']*100:.2f}% | {results[2]['recall']*100:.2f}% | {results[2]['f1']*100:.2f}% | {results[2]['fpr']*100:.2f}% | {results[2]['auc']:.4f} | {results[2]['latency_ms']:.3f} ms |

---

## 2. On-Device Llama 3.2 1B-Instruct (ExecuTorch) Benchmarks

Measured via ExecuTorch `llama_main` runtime running `llama3_2_1b.pte` compiled with XNNPACK on CPU:

| Metric | Measured Value | Unit |
| :--- | :--- | :--- |
| **Runtime Backend** | ExecuTorch v1.6.0 with XNNPACK | C++ native runtime |
| **Quantization / Precision** | bfloat16 (`bf16`) | Floating point |
| **Model Size (.pte file)** | 2.4 GB | On-disk flatbuffer |
| **Model Load Time** | ~41 ms | Fast mmap initialization |
| **Time to First Token (TTFT)** | ~52 - 86 s | Full context prompt prefill |
| **Prefill Throughput** | ~1.97 - 1.99 | tokens / second |
| **Decode Throughput** | ~1.36 - 1.39 | tokens / second |
| **Peak Process RSS** | ~1.6 - 2.5 GB | RAM consumption during generation |
| **Cloud Dependency** | **0.0% (Zero)** | 100% offline, air-gapped |

---

## 3. Dataset Lineage & Integrity Disclosure

- **URL Phishing Dataset:** 29,959 unique URLs (15,000 benign from Tranco / Alexa tops, 14,959 phishing from verified open feeds, plus 15 curated high-impact Indian banking impersonation URLs).
- **SMS Dataset:** 5,584 messages from the standard UCI Machine Learning SMS Spam Collection augmented with 10 labeled Indian fraud cases (KYC suspension, electricity bill disconnection, lottery scams).
- **Network Dataset:** CICIDS-2017 benchmark traffic (10,000 baseline normal flows from `monday.csv` for training; 4,000 mixed flows from `monday.csv` and `friday.csv` comprising PortScan, DDoS, and Botnet traffic for evaluation).
"""

    os.makedirs(os.path.dirname(BENCHMARK_OUT), exist_ok=True)
    with open(BENCHMARK_OUT, "w") as f:
        f.write(md_content)
    print(f"\nBenchmark report written to: {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
