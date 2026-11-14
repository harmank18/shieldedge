# ShieldEdge: On-Device Threat, Phishing & Scam Detection
### Bharat AI-SoC Student Challenge 2026–27 · Problem Statement 3 (PS3)
**Theme:** Cybersecurity & Digital Safety  
**Platform Target:** Arm Cortex-A Edge Devices / Linux x86_64 Reference Platform  
**Team Repository:** [github.com/harmank18/shieldedge](https://github.com/harmank18/shieldedge)  

---

## 📌 Project Overview

ShieldEdge is a privacy-preserving, 100% on-device cybersecurity assistant designed to protect citizens from digital fraud, phishing, and network anomalies without sending any user data to the cloud.

The system addresses three critical digital threat vectors prevalent in the Indian digital ecosystem:
1. **Phishing URLs:** Fast lexical and structural URL analysis detecting deceptive banking, tax refund, and telecom domains.
2. **SMS Scam & Smishing:** Detection of fake urgency, KYC block notices, and lottery scams via an optimized NLP classification pipeline.
3. **Network Flow Anomalies:** Statistical outlier detection for anomalous outbound connections and potential C2/botnet beaconing.
4. **On-Device Advisory Explainer:** A quantized **Llama 3.2 1B-Instruct** model deployed locally via Meta's **ExecuTorch** (C++ native runtime compiled with XNNPACK) that translates raw security signals into plain-language explanations and concrete remediation steps.

---

## 🏗️ System Architecture: Two-Pipeline Decoupling

```
+-------------------------------------------------------------------------------+
|                         ShieldEdge On-Device System                           |
|                                                                               |
|   [Input: URL / SMS Message / Flow Telemetry]                                 |
|                       │                                                       |
|                       ▼                                                       |
|   [DETECTION PIPELINE: Fast & Deterministic ML (<1 ms)]                       |
|   ├── URL Classifier: Random Forest (13 lexical/structural features)          |
|   ├── SMS Classifier: TF-IDF + Calibrated LinearSVC                           |
|   └── Network Detector: Isolation Forest on Flow Vectors                      |
|                       │                                                       |
|                       ▼                                                       |
|   [DECISION ENGINE: Strict Thresholds & Prompt Injection Guard]               |
|   ├── SAFE ──────────────────────────────► Fast Return (Zero LLM overhead)   |
|   └── SUSPICIOUS / THREAT                                                     |
|                       │ (Sanitized Evidence Summary)                          |
|                       ▼                                                       |
|   [ADVISORY PIPELINE: On-Device ExecuTorch LLM]                               |
|   ├── Model: Llama 3.2 1B-Instruct (.pte flatbuffer, 2.4 GB)                  |
|   ├── Backend: ExecuTorch v1.6.0 with XNNPACK CPU Acceleration                |
|   └── Prompt Guardrails: Rigid Llama 3 chat template + input length capping   |
|                       │                                                       |
|                       ▼                                                       |
|   [OUTPUT & LOCAL AUDIT]                                                      |
|   ├── Actionable Citizen Advice (CERT-In 1930 / DoT 1909 reporting)           |
|   └── Local SQLite Alert Database (Zero Cloud Transmissions)                  |
+-------------------------------------------------------------------------------+
```

---

## 📊 Measured Model Performance

All models are evaluated on held-out test splits from real benchmark datasets:

| Modality | Architecture | Dataset | Test Samples | Accuracy | Precision | Recall | F1-Score | FPR | ROC-AUC | Latency / Sample |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **URL Phishing** | Random Forest (`n=100`, `depth=10`) | Mitake PhishBenign (Held-out) | 5,992 | 80.07% | 82.38% | 76.44% | 79.30% | 16.30% | 0.8958 | 0.010 ms |
| **SMS Scam** | TF-IDF + Calibrated LinearSVC | UCI SMS Spam + Curated Indian Scams | 1,117 | 98.03% | 96.40% | 88.74% | 92.41% | 0.52% | 0.9857 | 0.023 ms |
| **Network Anomaly** | Isolation Forest (`contam=0.03`) | CICIDS-2017 (Monday + Friday Attacks) | 4,000 | 98.55% | 97.46% | 99.70% | 98.57% | 2.60% | 0.9796 | 0.007 ms |

### On-Device ExecuTorch Llama 3.2 1B-Instruct Telemetry

- **Test Machine:** Dell Inspiron 15 3520 (Intel Core x86_64, 15GB RAM, Ubuntu 24.04 LTS)
- **Model Load Time:** 41 ms (fast mmap flatbuffer loading)
- **Prefill Throughput:** ~1.98 tokens / second
- **Decode Throughput:** ~1.36 tokens / second
- **RAM Footprint:** ~1.6 - 2.5 GB peak RSS during inference
- **Cloud Dependency:** **0.0% (Zero)**

---

## 📁 Repository Structure

```
shieldedge/
├── shieldedge.py                 # Root CLI entry point
├── requirements.txt              # Python package requirements
├── setup.sh                      # Setup script
│
├── shieldedge/                   # Core application package
│   ├── __init__.py
│   ├── url_detector.py           # Lexical URL feature extraction & RF predictor
│   ├── sms_detector.py           # SMS text preprocessor & LinearSVC pipeline
│   ├── network_detector.py       # Flow feature extractor & Isolation Forest
│   ├── decision_engine.py        # Threat calibration & injection guardrail
│   ├── prompt_constructor.py     # Llama 3 chat template formatter
│   ├── llama_runner.py           # Native ExecuTorch binary bridge & template fallback
│   ├── storage.py                # Local SQLite privacy-safe audit logger
│   └── ui_cli.py                 # Unified terminal UI & telemetry renderer
│
├── executorch/                   # Compiled ExecuTorch binaries & model
│   ├── llama_main                # Compiled ExecuTorch C++ LLM runner binary (9.2 MB)
│   ├── libllama_runner.so        # Shared runner library (4.7 MB)
│   ├── llama3_2_1b.pte           # Exported Llama 3.2 1B-Instruct program (2.4 GB)
│   └── export_llama.sh           # Reproducible model export script
│
├── models/                       # Training & evaluation scripts
│   ├── train_url_classifier.py   # Trains URL Random Forest
│   ├── train_sms_classifier.py   # Trains SMS LinearSVC pipeline
│   ├── train_network_detector.py # Trains Network Isolation Forest on CICIDS-2017
│   └── evaluate_models.py        # Generates quantitative benchmark report
│
├── data/
│   ├── download_datasets.py      # Dataset loader for benchmark data
│   └── raw/                      # Raw dataset CSV files
│
├── tests/
│   └── test_prompt_injection.py  # Security tests verifying prompt injection defense
│
├── results/
│   └── benchmarks.md             # Full benchmark report and metrics
│
├── demo/
│   └── demo_script.md            # Step-by-step verification and demo commands
│
└── docs/
    └── PROJECT_BRIEF.md          # Competition task specification & requirements
```

---

## 🚀 Quickstart & Usage

### 1. Environment Setup

```bash
git clone https://github.com/harmank18/shieldedge.git
cd shieldedge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Threat Detection (CLI)

```bash
# Analyze a suspicious banking link
python3 shieldedge.py --url "http://sbi-net-banking-update.xyz/login"

# Analyze an incoming SMS scam message
python3 shieldedge.py --sms "Dear customer, your State Bank account will be blocked. Update KYC at http://sbi-verify.xyz"

# Analyze a network flow (port, byte_rate, pkt_rate)
python3 shieldedge.py --flow 4444 2000000 600

# View local security audit log
python3 shieldedge.py --alerts
```

### 3. Run Security & Model Tests

```bash
# Prompt injection containment verification
python3 tests/test_prompt_injection.py

# Quantitative evaluation across all test sets
python3 models/evaluate_models.py
```

---

## 🛡️ Privacy & Legal Disclosures

1. **Air-Gapped & Offline Guarantee:** ShieldEdge makes zero network API calls during inference. All machine learning models and LLM runtimes execute strictly locally on the host CPU.
2. **Local Data Minimization:** Analysis logs are written only to a local SQLite database (`shieldedge_alerts.db`). No telemetry or private user communications leave the device.
3. **Network Telemetry Notice:** Phase 1 uses exported network flow metadata (packet rates, byte rates, destination ports) rather than promiscuous live packet capture to comply with host operating system privacy boundaries.

---

## 👥 Team & Mentor

- **Developer:** Harman Singh ([harmank18](https://github.com/harmank18))
- **Challenge Track:** Bharat AI-SoC Student Challenge 2026–27 (Problem Statement 3: Cybersecurity & Digital Safety)
