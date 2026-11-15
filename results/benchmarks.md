# ShieldEdge Benchmark & Evaluation Results

> **Test Hardware:** Dell Inspiron 15 3520 (Intel Core i3/i5 12th Gen, x86_64, 15GB RAM, Ubuntu 24.04 LTS)  
> **Evaluation Mode:** Local on-device execution (100% offline, zero cloud calls)  
> **Date Generated:** 2026-11-16 00:01:52

---

## 1. Classical Detection Models Evaluation

Evaluated strictly on held-out test splits not used during model training.

| Task | Model Architecture | Evaluation Dataset | Test Samples | Accuracy | Precision | Recall | F1-Score | False Positive Rate | ROC-AUC | Avg Latency / Sample |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **URL Phishing** | Random Forest (`n=100`, `depth=10`) | Mitake + Curated Phish (Held-out) | 5,992 | 83.49% | 95.42% | 70.32% | 80.97% | 3.37% | 0.9419 | 0.016 ms |
| **SMS Scam** | TF-IDF + Calibrated LinearSVC | UCI SMS Spam + Curated Indian Scams | 1,121 | 98.48% | 97.26% | 91.61% | 94.35% | 0.41% | 0.9877 | 0.029 ms |
| **Network Anomaly** | Isolation Forest (`contam=0.03`) | CICIDS-2017 (Monday Normal + Friday Attacks) | 4,000 | 98.55% | 97.46% | 99.70% | 98.57% | 2.60% | 0.9796 | 0.007 ms |

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
