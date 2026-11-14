# ShieldEdge: Project Brief, Corrections and Plan

Bharat AI-SoC Challenge 2026-27, PS3. Written 14 Nov 2026. Keep this file in the repo (for example as `docs/PROJECT_BRIEF.md`) so the coding agent reads it first.

## 1. Challenge facts (from the official Arm Developer Labs page)

| Item | Value |
|---|---|
| Phase 1 submission deadline | **23 Nov 2026** (the design doc's 30 Nov is wrong) |
| Phase 1 shortlist | Mid-Dec 2026 |
| Phase 2 (PoC) submission | **5 Feb 2027** (the design doc's 10 Feb is wrong) |
| Finals | 15-16 Feb 2027, live virtual demo and jury Q&A |
| Team | 1-3 members from the same college, with a team leader and a faculty mentor |
| Phase 1 deliverables | Source code on GitHub, a working application, a demo video, a technical report, submitted through the official form |
| Hard requirement | A Llama-family model deployed on-device with ExecuTorch |

The official page gives no video length or report page limit. The design doc's "video up to 5 min, report up to 8 pages" is unconfirmed, so check the submission form.

Support: support@armbharatchallenge.com, and tech-queries@armbharatchallenge.com with "PS3" in the subject.

## 2. What ShieldEdge is

Two pipelines on one device, with no cloud.

1. **Detection (deterministic ML):** URL classifier (Random Forest), SMS classifier (TF-IDF + linear SVM), network-flow anomaly detector (Isolation Forest). Output: verdict plus confidence plus evidence.
2. **Advisory (Llama 3.2 1B-Instruct, INT4, via ExecuTorch/XNNPACK on the Arm CPU):** turns the structured evidence into a plain-language explanation and remediation steps. Llama never decides the verdict. If its output is empty or malformed, a template fallback is shown.

Also: a SQLite audit log that stores only SHA-256 hashes of inputs, never raw SMS text.

## 3. What has been done so far

This comes from a log the coding agent produced. **I could not open the repo (`github.com/harmank18/shieldedge`, public) and have not seen the code**, so treat everything below as unverified.

- Repo created and code pushed over SSH.
- Files reported: `shieldedge.py` (CLI), `shieldedge/` package (`url_detector`, `sms_detector`, `network_detector`, `decision_engine`, `prompt_constructor`, `llama_runner`, `storage`, `ui_cli`), three trained `.joblib` models (URL 81 KB, SMS 244 KB, network 1.0 MB), `data/download_datasets.py`, `requirements.txt`, `setup.sh`, `README.md`.
- One example run per detector, with a laptop benchmark: URL 4.03 ms, SMS 1.65 ms, network 8.45 ms per query, peak RAM about 159 MB.
- A README was shared. It describes the project well, but see the corrections below.

## 4. Corrections and risks (fix these first)

1. **Llama is not running yet.** The runner was described as a fallback explainer and ExecuTorch setup was still an open next step. The README claims Llama via ExecuTorch works. Do not claim it until it runs.
2. **Training data may be synthetic.** `download_datasets.py` is described as an "Indian scam synthesizer". The model files are tiny and confidences of 100% and 99.1% look suspicious. The design doc says to use only verified labelled sources. Check the data and replace or clearly separate any synthetic data.
3. **Metric claims are unproven.** The summary said FPR under 2% and F1 above 0.92, but no evaluation output or held-out test numbers were shown.
4. **The network detector may be hand-tuned.** It was called with three numbers (`--flow 4444 2500000 850`) and port 4444 may be hard-coded as a suspicious port. The plan was to train on CICIDS 2017.
5. **Benchmarks are laptop numbers.** Label them as laptop, never as Raspberry Pi results.
6. **README does not match the repo.** It lists `data/download_datasets.sh`, `models/train_*.py`, `evaluate_models.py`, `results/benchmarks.md` and `demo/demo_script.md`, none of which appeared in the log's file tree. It runs `python3 -m shieldedge.ui_cli`, but the log's tests used `python3 shieldedge.py`. It also says three threat vectors and lists four, lacks dataset sources, results, limitations, the network-capture legality warning and team/mentor info.
7. **Design doc inconsistencies.** Latency estimates disagree (3-8 s, 10-30 s and 15-60 s in different sections), so measure them. Its KV-cache estimate (100-200 MB for 512 tokens) looks too high. From the Llama 3.2 1B config as I recall it (16 layers, 8 KV heads, head size 64) it is nearer 16 MB at 16-bit, but verify against the model card and measure peak RAM on the device.
8. **Network capture legality.** Only capture traffic on networks you own or have written permission to monitor. Put this in the README.

## 5. Rules for the coding agent

- Never invent metrics. Only report numbers from commands that were actually run, and say which device produced them.
- Do not generate labelled data and present it as real. If synthetic data is used, mark it as synthetic everywhere (code, README, report).
- Keep train, validation and test data separate. Use a temporal split for URLs and a stratified split for SMS. Evaluate on a realistic class ratio, not only 50/50.
- Never pass raw SMS or URL text into the Llama prompt as instructions. Only sanitised, length-limited summaries and detector signals.
- Keep the template fallback working at all times.
- Commit in small steps and keep `main` runnable.
- When a command fails, show the full error and fix the cause. Do not silently skip a step.

## 6. Plan, 14 to 23 Nov

| Days | Goal | Done when |
|---|---|---|
| 14-15 | Llama 3.2 1B-Instruct weights downloaded, ExecuTorch built, first export attempted | An export runs, even if imperfect |
| 16-17 | Llama runs from a prompt and is wired into the CLI | Verdict plus real Llama explanation in the terminal |
| 16-18 | Real datasets, proper train/test split, evaluation script | Held-out numbers with an `evaluate_models.py` that anyone can rerun |
| 18-19 | Pi 5 test if available; measure load time, time to first token, tokens per second, peak RAM (at least 10 runs, mean and standard deviation) | `results/benchmarks.md` with each number labelled by device |
| 20-21 | Fix the README to match the repo; add results, limitations, legal note, team and mentor | README is truthful and complete |
| 22 | Demo video and technical report | Both finished |
| 23 | Submit through the official form | Confirmation received |

Treat the 22nd and 23rd as buffer.

## 7. Tasks for the agent, in order

1. **Audit.** Run `git log --oneline | head -20`, `ls -R` (skip `venv` and `.git`), and read `data/download_datasets.py`. Report what is real, what is synthetic, and which README claims are untrue.
2. **Llama export.** Get Llama 3.2 1B-Instruct weights (Meta licence required), build ExecuTorch from `github.com/pytorch/executorch`, and follow the current Llama example README to produce an INT4 `.pte` with the XNNPACK backend. Also read Arm's learning path "Run Llama 3 on a Raspberry Pi 5 using ExecuTorch" (learn.arm.com). Flags change between versions, so use the current docs and write the exact commands that worked into `executorch/export_llama.sh`.
3. **Wire Llama in.** Make `llama_runner.py` call the real ExecuTorch runner with a 256-token prompt cap and a 150-token reply cap. Keep the fallback. Check output for the required sections and length.
4. **Fix the data and evaluation.** Use PhishTank (verified phishing) and Tranco (benign) for URLs, UCI SMS Spam plus manually collected Indian scam messages for SMS (clearly labelled), and CICIDS 2017 for the network model. Add the training scripts and an `evaluate_models.py` that outputs precision, recall, F1, FPR and ROC-AUC on a held-out set.
5. **Test the prompt.** Run at least 20 different evidence inputs and one prompt-injection test. Confirm Llama never adds signals that were not supplied.
6. **Measure.** Record load time, time to first token, tokens per second and peak RAM with at least 10 runs each. Label every number with the device. Add a laptop-versus-Pi comparison if a Pi is available.
7. **Docs.** Rewrite the README so every command in it works from a fresh clone. Add the results table, limitations (false positives and negatives, no live capture in Phase 1), the network-capture legal warning, data sources, and the team and mentor.
8. **Submission pack.** Demo script, demo video, and the technical report.

## 8. Open questions (needed from the team)

- Is there a Raspberry Pi 5 or other Arm board available? If not, the report must say Pi measurements are missing.
- Who is on the team, and is there a faculty mentor? Has the submission form been filed?
- Is the Llama licence accepted and the Hugging Face token set up?
- Is the video and report length limit confirmed from the form or the mentoring session?

## 9. Learning plan (3-person split, if there are three people)

- **Person A, ML and data:** ML basics, classical models, evaluation, NLP basics. Owns the detectors, datasets and evaluation numbers.
- **Person B, Llama and edge:** transformers, LLMs in practice, edge AI, Arm basics. Owns Llama via ExecuTorch, the prompt builder and Pi deployment.
- **Person C, security and engineering:** networking, cybersecurity basics, Python/Git/Linux/SQLite. Owns the network side, CLI, logging, README, video and report.

Everyone must be able to answer: why does Llama only explain, and how do we know the numbers are real?

Suggested resources: Karpathy's "Deep Dive into LLMs like ChatGPT" (about 3.5 h, YouTube), MIT EfficientML.ai Lecture 5 on quantisation (about 75 min, YouTube), Arm's ExecuTorch course (learn.arm.com/courses/executorch-on-arm).
