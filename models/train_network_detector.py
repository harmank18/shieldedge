"""
Training script for Network Flow Anomaly Detector.
Trains an Isolation Forest model on real baseline traffic flows from CICIDS-2017
and evaluates detection performance against real intrusion/attack flows (DDoS, PortScan, Botnet).
Exports the model to models/network_isolation_forest.joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

from shieldedge.network_detector import FLOW_FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "network_train.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "network_test.csv")
MODEL_OUT = os.path.join(BASE_DIR, "models", "network_isolation_forest.joblib")


def main():
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError("Network dataset not found. Run download_datasets.py first.")

    print(f"Loading real baseline flows from {TRAIN_DATA_PATH}...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    X_train = df_train[FLOW_FEATURE_NAMES].values

    print(f"Training Isolation Forest on {len(X_train)} real baseline flows...")
    model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
    model.fit(X_train)

    print(f"Loading test set with real intrusion flows from {TEST_DATA_PATH}...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    X_test = df_test[FLOW_FEATURE_NAMES].values
    y_test = df_test["is_anomaly"].values

    # Isolation Forest: -1 = anomaly, 1 = normal
    raw_preds = model.predict(X_test)
    y_pred = np.where(raw_preds == -1, 1, 0)
    scores = -model.score_samples(X_test)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, scores)

    print("\n=== Network Anomaly Detector Performance on CICIDS-2017 Test Set ===")
    print(f"Accuracy:        {(tp + tn) / len(y_test):.4f}")
    print(f"Precision:       {tp / max(1, tp + fp):.4f}")
    print(f"Recall:          {tp / max(1, tp + fn):.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"False Pos Rate:  {fpr:.4f}")
    print(f"ROC-AUC:         {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {tn} | FP: {fp}\nFN: {fn} | TP: {tp}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nModel exported successfully to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
