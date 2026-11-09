# ShieldEdge: On-Device Threat, Phishing & Scam Detection
### Bharat AI-SoC Student Challenge 2026–27 · Problem Statement 3 (PS3)
**Theme:** Cybersecurity & Digital Safety  
**Platform Target:** Arm Cortex-A Edge Devices (e.g., Raspberry Pi 5 / Arm64 Linux)  

---

## 📌 Project Overview
ShieldEdge is a privacy-preserving, entirely on-device threat detection and explainability assistant built for edge Arm processors. It addresses three critical digital threat vectors facing ordinary citizens in India:

1. **Phishing URLs:** Fast lexical & structural analysis of URLs without network calls to detect spoofed banking and government portals.
2. **SMS Scam & Smishing:** Detection of fake urgency, KYC suspension alerts, and malicious UPI / lottery requests using an optimized NLP pipeline.
3. **Network Traffic Anomalies:** Identification of suspicious outbound C2 (Command & Control) connections and port scans on local networks.
4. **On-Device Advisory Explainer:** A quantized **Llama 3.2 1B-Instruct** model running locally via Meta's **ExecuTorch** (optimized with the **XNNPACK** backend for **Arm NEON SIMD**) that translates structured technical threat indicators into plain-language advice and concrete remediation steps for non-technical users.

---

## 🏗️ System Architecture: Two-Pipeline Decoupling

```
+-----------------------------------------------------------------------+
|                       ShieldEdge On-Device System                     |
|                                                                       |
|  [Input: URL / SMS / Network Flow]                                    |
|                   │                                                   |
|                   ▼                                                   |
|  [DETECTION PIPELINE: Fast & Deterministic ML (<50ms)]                |
|  - Random Forest (URL Lexical Features)                               |
|  - Linear SVM + TF-IDF (SMS Spam / Smishing)                          |
|  - Isolation Forest (Network Flow Outlier Detection)                  |
|                   │                                                   |
|                   ▼                                                   |
|  [DECISION ENGINE] ─── (SAFE) ──────────────► [User Notification]     |
|                   │                                                   |
|          (SUSPICIOUS / MALICIOUS)                                     |
|                   │                                                   |
|                   ▼                                                   |
|  [ADVISORY PIPELINE: On-Device GenAI]                                 |
|  - Structured Prompt Constructor (Guardrailed against Hallucination)  |
|  - Llama 3.2 1B-Instruct (INT4 via ExecuTorch / Arm NEON)             |
|                   │                                                   |
|                   ▼                                                   |
|  [OUTPUT & AUDIT]                                                     |
|  - Threat Explanation & Actionable Remediation                        |
|  - Local SQLite Privacy-Preserving Audit Log (SHA-256 Hashed Inputs)  |
+-----------------------------------------------------------------------+
```

---

## 📁 Repository Structure
```
shieldedge/
├── README.md               # Project documentation & execution guide
├── requirements.txt        # Python dependencies
├── setup.sh                # Edge device setup script
│
├── data/
│   ├── download_datasets.sh# Dataset download helper
│   └── preprocessing/      # Data cleaning and feature extraction
│
├── models/
│   ├── train_url_classifier.py
│   ├── train_sms_classifier.py
│   ├── train_network_detector.py
│   └── evaluate_models.py  # Unified evaluation metrics (F1, FPR, ROC-AUC)
│
├── shieldedge/
│   ├── __init__.py
│   ├── url_detector.py     # Structural URL feature extractor + classifier
│   ├── sms_detector.py     # SMS preprocessor + classifier
│   ├── network_detector.py # Flow feature extractor + classifier
│   ├── decision_engine.py  # Verdict logic + rule filters
│   ├── prompt_constructor.py# Formats structured evidence for Llama
│   ├── llama_runner.py     # ExecuTorch runner wrapper / fallback
│   ├── storage.py          # Local SQLite privacy-safe logger
│   └── ui_cli.py           # Unified CLI interface
│
├── results/
│   └── benchmarks.md       # On-device latency, RAM, and CPU telemetry
└── demo/
    └── demo_script.md      # Proof-of-concept verification script
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
git clone git@github.com:harmank18/shieldedge.git
cd shieldedge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Threat Detection (CLI)
```bash
# Phishing URL Scan
python3 -m shieldedge.ui_cli --url "http://sbi-net-banking-update.xyz/login"

# SMS Scam Scan
python3 -m shieldedge.ui_cli --sms "Dear customer, your SBI account is blocked. Update KYC at bit.ly/fake-kyc"
```
