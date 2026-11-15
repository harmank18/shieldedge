# ShieldEdge: End-to-End Live Demonstration Script

This document provides exact, copy-pasteable commands to demonstrate and verify all ShieldEdge capabilities on an air-gapped or offline device.

---

## 🎬 Demonstration Overview

ShieldEdge operates strictly on-device without any cloud API dependencies. The demonstration consists of 4 parts:
1. **Phishing URL Detection & Guardrailed Advisory** (Detecting spoofed Indian banking portals)
2. **SMS Scam / Smishing Detection & Remediation** (Detecting urgent KYC suspension messages)
3. **Network Flow Anomaly Detection** (Detecting malicious C2 beaconing and packet bursts)
4. **Adversarial Prompt Injection Defense** (Confirming LLM explainer cannot be hijacked)

---

## 🛠️ Step 0: Environment Activation

Open a terminal and activate the project virtual environment:

```bash
cd /home/harman/shieldedge
source venv/bin/activate
```

---

## 🧪 Demo 1: Phishing URL Scan

Simulate a user receiving a deceptive link impersonating the State Bank of India (`sbi-net-banking-update.xyz`):

```bash
python3 shieldedge.py --url "http://sbi-net-banking-update.xyz/login"
```

### Expected Output:
- **Verdict:** `[!] PHISHING DETECTED` (Confidence: ~100%)
- **Identified Signals:** Insecure HTTP protocol, High-risk `.xyz` TLD, Brand impersonation (SBI), Domain hyphenation spoofing.
- **AI Advisory Explainer:** Powered by on-device **ExecuTorch (Llama 3.2 1B-Instruct)** explaining the specific risk and warning the user not to enter banking credentials.
- **Telemetry:** Execution latency and process RSS memory footprint.

---

## 🧪 Demo 2: SMS Scam / Smishing Scan

Simulate an incoming SMS message impersonating bank authority with false urgency:

```bash
python3 shieldedge.py --sms "Dear customer, your State Bank account will be blocked in 24 hours. Update KYC immediately at http://sbi-kyc-verify.in/login"
```

### Expected Output:
- **Verdict:** `[!] SCAM DETECTED` (Confidence: >95%)
- **Identified Signals:** Urgency keywords (`blocked`, `immediately`), Authority mentions (`State Bank`, `KYC`), Embedded link.
- **AI Advisory Explainer:** Clear remediation steps recommending reporting to Indian cyber authorities (DoT Chakshu / 1909 or CyberCrime helpline 1930).

---

## 🧪 Demo 3: Network Traffic Flow Anomaly Scan

Simulate an anomalous outbound flow with Trojan/C2 port activity and high packet burst:

```bash
python3 shieldedge.py --flow 4444 2000000 600
```

### Expected Output:
- **Verdict:** `[!] ANOMALY DETECTED` (Confidence: ~88%)
- **Identified Signals:** Outbound connection to Trojan/C2 port (4444), Abnormal packet rate (>500 pkts/s), High outbound data burst (>1 MB/s).
- **AI Advisory Explainer:** Clear guidance to disconnect the affected device from the local network and inspect for malware.

---

## 🧪 Demo 4: View Local Privacy Audit Log

Inspect the locally persisted SQLite database of security events (no user data leaves the machine):

```bash
python3 shieldedge.py --alerts
```

---

## 🧪 Demo 5: Prompt Injection Security Verification

Verify that an attacker attempting to embed jailbreak prompts inside messages cannot override the verdict or hijack the Llama explainer:

```bash
python3 tests/test_prompt_injection.py
```

### Expected Output:
- `[✓] SMS Prompt Injection Test: PASSED (Payload neutralized, verdict preserved)`
- `[✓] URL Parameter Injection Test: PASSED (Input bounded and classified)`

---

## 📊 Demo 6: Run Full Model Evaluation Suite

Run quantitative evaluation across held-out test splits for all three models:

```bash
python3 models/evaluate_models.py
```

Results are logged directly to `results/benchmarks.md`.

---

## 🧪 Demo 7: QR Code Threat Scanner

Scan a physical or digital QR code image for malicious URLs and credential harvesting links:

```bash
# Generate a test phishing QR code
python3 -c "from shieldedge.qr_scanner import generate_test_qr; generate_test_qr('http://sbi-net-banking-update.xyz/login/verify', '/tmp/test_qr.png')"

# Scan the QR code with ShieldEdge
python3 shieldedge.py --qr /tmp/test_qr.png
```

### Expected Output:
- **QR Codes Found:** 1
- **Overall Verdict:** `PHISHING`
- **Extracted URL & Signals:** Insecure protocol, spoofed brand, suspicious TLD, authentication path.
- **Triggered Remediation Pipeline:** Automatic link analysis and advisory.

---

## 🧪 Demo 8: Hindi / Hinglish Regional SMS Scam Detection

Simulate an incoming Hinglish scam message targeting Indian banking users:

```bash
python3 shieldedge.py --sms "Aapka SBI khata band ho jayega. Turant kyc update karo: http://sbi-verify.xyz Helpline: 9876543210"
```

### Expected Output:
- **Verdict:** `[!] SCAM DETECTED` (Confidence: 100%)
- **Identified Signals:** Hindi/Hinglish scam phrases (`band ho jayega`, `turant`), Financial authority mentions (`SBI`, `KYC`), Embedded phishing link.

---

## 🧪 Demo 9: Terminal Threat Intelligence Dashboard

Launch the interactive, terminal-based visual dashboard powered by Rich:

```bash
python3 shieldedge.py --dashboard
```

Displays total scan count, threat rate, breakdown by threat type (URL/SMS/Network), verdict distribution bar chart, and recent security logs from the local SQLite database.

---

## 🧪 Demo 10: Local Privacy-First REST API Service

Run ShieldEdge as a background daemon or microservice accessible to native apps via JSON REST API:

```bash
# Start server on port 5050
python3 shieldedge.py --serve &

# Query URL endpoint
curl -s -X POST http://localhost:5050/analyze/url \
  -H "Content-Type: application/json" \
  -d '{"url": "http://sbi-net-banking-update.xyz/login"}'

# Query SMS endpoint
curl -s -X POST http://localhost:5050/analyze/sms \
  -H "Content-Type: application/json" \
  -d '{"message": "Aapki bijli 2 ghante mein band ho jayegi. Bijli bill pay karo: http://electricity-pay.top"}'

# Check health and stats
curl -s http://localhost:5050/health
curl -s http://localhost:5050/stats
```
