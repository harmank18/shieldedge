# ShieldEdge: On-Device Threat, Phishing & Scam Detection
### Bharat AI-SoC Student Challenge 2026–27 · Problem Statement 3 (PS3)
**Theme:** Cybersecurity & Digital Safety  
**Organized by:** Arm · MeitY C2S Programme · IIT Delhi  
**Phase 1 Submission Deadline:** 23 November 2026  

---

## 📌 Project Overview
ShieldEdge is an on-device, privacy-preserving threat detection assistant designed for edge Arm architectures (e.g., Raspberry Pi 5 / Android). It analyzes phishing URLs, smishing SMS messages, and network flow anomalies locally with sub-50ms deterministic inference, paired with a quantized on-device Llama model to provide plain-language security advisory and remediation.

---

## 🌿 Team Collaboration & Branching Strategy

To maintain a clean and conflict-free workflow, all active development occurs in dedicated team branches. Work is reviewed and merged into `main` via Pull Requests.

* 🧑‍💻 **`harman-devs`**: Core detection pipeline, ML classifiers, telemetry benchmarking, CLI interface.
* 🧑‍💻 **`arnav-devs`**: Dataset augmentation (Indian scam patterns), web/UI interface.
* 🧑‍💻 **`genda-devs`**: Advisory prompt engineering, ExecuTorch export, technical report drafting.

---

### Quickstart for Collaborators

1. **Clone and checkout your branch:**
   ```bash
   git clone https://github.com/harmank18/shieldedge.git
   cd shieldedge
   git checkout <your-branch-name>
   ```

2. **Push your changes:**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   git push origin <your-branch-name>
   ```

3. **Merging:**
   Open a **Pull Request (PR)** on GitHub to merge your branch into `main` after team review.
