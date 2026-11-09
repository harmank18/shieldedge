"""
Training script for the SMS Spam/Smishing Detector.
Uses TF-IDF + Calibrated Linear Support Vector Classifier (LinearSVC)
to predict scam probability and output evaluation metrics.
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

from shieldedge.sms_detector import preprocess_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "sms_dataset.csv")
MODEL_OUT = os.path.join(BASE_DIR, "models", "sms_pipeline.joblib")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run download_datasets.py first.")

    print(f"Loading SMS dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["clean_msg"] = df["message"].astype(str).apply(preprocess_text)
    df["target"] = df["label"].apply(lambda x: 1 if str(x).strip().lower() == "spam" else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_msg"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Pipeline: TF-IDF + Calibrated LinearSVC (so predict_proba is supported)
    base_svc = LinearSVC(random_state=42, dual=False, max_iter=2000)
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words="english")),
        ("clf", calibrated_svc)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

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
    joblib.dump(pipeline, MODEL_OUT)
    print(f"\nModel exported successfully to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
