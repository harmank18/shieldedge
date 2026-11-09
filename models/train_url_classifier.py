"""
Training script for the URL Phishing Detector.
Extracts lexical features, trains a Random Forest Classifier, 
computes evaluation metrics (F1, FPR, ROC-AUC), and saves model.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

from shieldedge.url_detector import extract_url_features, FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "url_dataset.csv")
MODEL_OUT = os.path.join(BASE_DIR, "models", "url_rf_model.joblib")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run download_datasets.py first.")

    print(f"Loading URL dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    print("Extracting features from URLs...")
    feature_rows = []
    labels = []
    for _, row in df.iterrows():
        url = str(row["url"])
        label = 1 if row["label"] == "phishing" else 0
        feats = extract_url_features(url)
        feature_rows.append([feats[k] for k in FEATURE_NAMES])
        labels.append(label)

    X = np.array(feature_rows)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n=== Model Performance on Test Set ===")
    print(f"Accuracy:        {(tp + tn) / len(y_test):.4f}")
    print(f"Precision:       {tp / max(1, tp + fp):.4f}")
    print(f"Recall:          {tp / max(1, tp + fn):.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"False Pos Rate:  {fpr:.4f} (Target: < 0.02)")
    print(f"ROC-AUC:         {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN: {tn} | FP: {fp}\nFN: {fn} | TP: {tp}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"\nModel exported successfully to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
