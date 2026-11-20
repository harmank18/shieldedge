"""
Build the replay files for the web dashboard (python shieldedge.py --web).

  data/samples/replay_flows.csv     ~1,300 real CIC-IDS-2017 *Friday* flows (raw CICFlowMeter
                                    columns, IP addresses removed) arranged as a scripted
                                    scenario: normal traffic, then a port scan, a DDoS flood and
                                    botnet connection attempts, each mixed with normal traffic.
                                    The order is ours (this copy of the dataset has no usable
                                    timestamps); every flow and its label are real.
  data/samples/replay_messages.csv  held-out URLs and SMS (never used for training) plus the
                                    team-curated India sets, shuffled.

Everything the dashboard shows from these files is labelled "replay".
Run after the training scripts (needs data/processed/): python data/build_dashboard_replay.py
"""

import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.download_datasets import CURATED_DIR, cicids_path  # noqa: E402

SAMPLES = os.path.join(BASE_DIR, "data", "samples")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
SEED = 7

# (phase name, attack label or None, attack flows, benign flows mixed in)
SCENARIO = [
    ("normal", None, 0, 220),
    ("port scan", "Portscan", 160, 90),
    ("normal", None, 0, 160),
    ("DDoS flood", "DDoS", 220, 80),
    ("normal", None, 0, 140),
    ("botnet", "Botnet - Attempted", 60, 70),
    ("botnet", "Botnet", 20, 30),
    ("normal", None, 0, 120),
]


def build_flows():
    fri = pd.read_csv(cicids_path("friday"), low_memory=False)
    fri = fri.drop(columns=["Src IP dec", "Dst IP dec"], errors="ignore")
    benign = fri[fri.Label == "BENIGN"].sample(frac=1, random_state=SEED)
    used = 0
    parts = []
    for i, (phase, label, n_att, n_ben) in enumerate(SCENARIO):
        seg = [benign.iloc[used:used + n_ben]]
        used += n_ben
        if label:
            seg.append(fri[fri.Label == label].sample(n_att, random_state=SEED + i))
        seg = pd.concat(seg).sample(frac=1, random_state=SEED + i).assign(Phase=phase)
        parts.append(seg)
    out = pd.concat(parts, ignore_index=True)
    path = os.path.join(SAMPLES, "replay_flows.csv")
    out.to_csv(path, index=False)
    print(f"{path}: {len(out)} flows, {out.Label.value_counts().to_dict()}")


def build_messages():
    rows = []
    url_test = pd.read_csv(os.path.join(PROCESSED, "url_test.csv"))
    for label, n in ((1, 50), (0, 100)):
        s = url_test[url_test.label == label].sample(n, random_state=SEED)
        rows += [("URL", u, "phishing" if label else "benign", "Mitake held-out") for u in s.url]
    india_urls = pd.read_csv(os.path.join(CURATED_DIR, "india_urls.csv"))
    rows += [("URL", u, lab, "India curated") for u, lab in zip(india_urls.url, india_urls.label)]
    sms_test = pd.read_csv(os.path.join(PROCESSED, "sms_test.csv"))
    for label, n in (("spam", 40), ("ham", 60)):
        s = sms_test[sms_test.label == label].sample(n, random_state=SEED)
        rows += [("SMS", m, "scam" if label == "spam" else "genuine", "UCI held-out") for m in s.message]
    india_sms = pd.read_csv(os.path.join(CURATED_DIR, "india_sms_test.csv"))
    rows += [("SMS", m, "scam" if lab == "spam" else "genuine", "India curated")
             for m, lab in zip(india_sms.message, india_sms.label)]
    out = pd.DataFrame(rows, columns=["kind", "text", "label", "source"]).sample(frac=1, random_state=SEED)
    path = os.path.join(SAMPLES, "replay_messages.csv")
    out.to_csv(path, index=False)
    print(f"{path}: {len(out)} items, {out.groupby(['kind', 'label']).size().to_dict()}")


if __name__ == "__main__":
    os.makedirs(SAMPLES, exist_ok=True)
    build_flows()
    build_messages()
