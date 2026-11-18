"""
Train the ShieldEdge network anomaly detector (unsupervised).

Model     : log1p -> StandardScaler -> IsolationForest(200 trees), fitted on 60k
            benign CIC-IDS-2017 Monday flows only.
Thresholds: from the anomaly-score distribution of 20k *held-out benign* flows:
              ANOMALY    >= 99th percentile   (~1% of normal flows)
              SUSPICIOUS >= 98th percentile   (~2% of normal flows raise any alert)
            No attack data is used for training or threshold selection.
Evaluate  : models/evaluate_models.py (Friday benign + Friday attacks, i.e. a different day).

Run: python data/build_network_dataset.py && python models/train_network_detector.py
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shieldedge.network_detector import FLOW_FEATURE_NAMES, log1p_clip  # noqa: E402

PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_OUT = os.path.join(BASE_DIR, "models", "network_isolation_forest.joblib")


def main():
    train = pd.read_csv(os.path.join(PROCESSED, "net_train.csv"))
    holdout = pd.read_csv(os.path.join(PROCESSED, "net_holdout.csv"))

    pipe = make_pipeline(FunctionTransformer(log1p_clip), StandardScaler(),
                         IsolationForest(n_estimators=200, max_samples=512, random_state=42, n_jobs=-1))
    pipe.fit(train[FLOW_FEATURE_NAMES].values)

    s = -pipe.score_samples(holdout[FLOW_FEATURE_NAMES].values)
    bundle = {
        "pipeline": pipe,
        "feature_names": FLOW_FEATURE_NAMES,
        "threshold_anomaly": float(np.quantile(s, 0.99)),
        "threshold_suspicious": float(np.quantile(s, 0.98)),
        "score_p50": float(np.quantile(s, 0.50)),
        "score_p999": float(np.quantile(s, 0.999)),
        "benign_score_quantiles": np.quantile(s, np.linspace(0, 1, 1001)),
        "medians": train[FLOW_FEATURE_NAMES].median().to_dict(),
        "meta": {
            "dataset": "CIC-IDS-2017 (bvk/CICIDS-2017) Monday benign: 60k train, 20k threshold holdout",
            "sklearn": sklearn.__version__, "trained": time.strftime("%Y-%m-%d"),
        },
    }
    joblib.dump(bundle, MODEL_OUT, compress=3)
    print(f"Thresholds: SUSPICIOUS>={bundle['threshold_suspicious']:.4f}, ANOMALY>={bundle['threshold_anomaly']:.4f}")
    print(f"Saved {MODEL_OUT} ({os.path.getsize(MODEL_OUT) / 1e6:.2f} MB). Run models/evaluate_models.py for metrics.")


if __name__ == "__main__":
    main()
