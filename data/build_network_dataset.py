"""
Build the ShieldEdge network datasets from CIC-IDS-2017 (bvk/CICIDS-2017 on Hugging Face).

  net_train.csv    60,000 benign flows from Monday          -> benign reference set for the kNN novelty model
  net_holdout.csv  20,000 other benign flows from Monday    -> score thresholds (benign only)
  net_test.csv     20,000 benign flows from *Friday* + up to 4,000 flows of each Friday
                   attack class (DDoS, PortScan, Botnet, Botnet - Attempted)
                   -> evaluation on a different day than training
  data/samples/demo_flows.csv  20 real Friday flows in raw CICFlowMeter format for the demo

Run: python data/build_network_dataset.py
"""

import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.download_datasets import cicids_path  # noqa: E402
from shieldedge.network_detector import cic_to_features  # noqa: E402

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
SAMPLES = os.path.join(BASE_DIR, "data", "samples")
SEED = 42


def main():
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(SAMPLES, exist_ok=True)

    mon = cic_to_features(pd.read_csv(cicids_path("monday"), low_memory=False)).dropna()
    mon = mon[mon.label == "BENIGN"]
    train = mon.sample(60_000, random_state=SEED)
    holdout = mon.drop(train.index).sample(20_000, random_state=SEED)
    train.to_csv(os.path.join(PROCESSED, "net_train.csv"), index=False)
    holdout.to_csv(os.path.join(PROCESSED, "net_holdout.csv"), index=False)
    print(f"Monday benign: {len(mon):,} -> train {len(train):,}, holdout {len(holdout):,}")

    fri_raw = pd.read_csv(cicids_path("friday"), low_memory=False)
    fri = cic_to_features(fri_raw).dropna()
    parts = [fri[fri.label == "BENIGN"].sample(20_000, random_state=SEED)]
    for label, g in fri[fri.label != "BENIGN"].groupby("label"):
        parts.append(g.sample(min(len(g), 4_000), random_state=SEED))
    test = pd.concat(parts)
    test.to_csv(os.path.join(PROCESSED, "net_test.csv"), index=False)
    print("Friday test set:", test.label.value_counts().to_dict())

    # Demo file: raw CICFlowMeter columns (as a capture tool would export them), real Friday flows.
    picks = {"BENIGN": 12, "Portscan": 3, "Botnet - Attempted": 3, "DDoS": 2}
    demo = pd.concat([fri_raw[fri_raw.Label == k].sample(n, random_state=7) for k, n in picks.items()])
    demo = demo.sample(frac=1, random_state=7).drop(columns=["Src IP dec", "Dst IP dec"], errors="ignore")
    demo.to_csv(os.path.join(SAMPLES, "demo_flows.csv"), index=False)
    print(f"Demo sample: {os.path.join(SAMPLES, 'demo_flows.csv')} ({len(demo)} flows, IP addresses removed)")


if __name__ == "__main__":
    main()
