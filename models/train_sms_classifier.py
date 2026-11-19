"""
Train the ShieldEdge SMS scam classifier.

Data  : UCI SMS Spam Collection (exact duplicates removed *before* splitting -
        the raw corpus has ~400 duplicates that otherwise leak into the test set)
        + data/curated/india_sms_train.csv (team-authored Indian messages,
        clearly marked; the separate india_sms_test.csv is never used here).
Split : stratified 80/20 on the de-duplicated UCI data. The UCI test part and
        the curated India test set are evaluated separately by evaluate_models.py.
Model : TF-IDF word 1-2-grams + character 2-5-grams (char_wb, catches obfuscations
        such as "fr33", "c1aim", "£1000") -> LinearSVC with sigmoid calibration
        (CalibratedClassifierCV, 5 folds) so outputs are probabilities.
        Chosen by 5-fold CV on the training split only: F1 0.935 (word 1-3-grams)
        -> 0.959 (word + char), AUC 0.991 -> 0.994.
Out   : models/sms_pipeline.joblib, data/processed/sms_test.csv

Run: python models/train_sms_classifier.py
"""

import os
import sys
import time

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.download_datasets import CURATED_DIR, uci_sms_path  # noqa: E402
from shieldedge.sms_detector import preprocess_text  # noqa: E402

MODEL_OUT = os.path.join(BASE_DIR, "models", "sms_pipeline.joblib")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
SEED = 42


def main():
    uci = pd.read_csv(uci_sms_path(), sep="\t", header=None, names=["label", "message"])
    n_raw = len(uci)
    uci = uci.drop_duplicates("message").reset_index(drop=True)
    uci["source"] = "uci"
    print(f"UCI SMS: {n_raw} messages, {n_raw - len(uci)} exact duplicates removed -> {len(uci)}")

    train_uci, test_uci = train_test_split(uci, test_size=0.2, random_state=SEED, stratify=uci.label)
    india = pd.read_csv(os.path.join(CURATED_DIR, "india_sms_train.csv"))
    train = pd.concat([train_uci, india], ignore_index=True)
    print(f"Train: {len(train_uci)} UCI + {len(india)} team-authored Indian = {len(train)} "
          f"({(train.label == 'spam').mean():.1%} spam) | UCI test: {len(test_uci)}")

    pipeline = Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ])),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=5, method="sigmoid")),
    ])
    pipeline.fit(train.message.astype(str).map(preprocess_text), (train.label == "spam").astype(int))

    bundle = {
        "pipeline": pipeline,
        "threshold_scam": 0.80,
        "threshold_suspicious": 0.50,
        "meta": {
            "dataset": "UCI SMS Spam Collection (deduplicated) + team-authored Indian SMS (india_sms_train.csv)",
            "n_train": int(len(train)), "n_test_uci": int(len(test_uci)),
            "sklearn": sklearn.__version__, "trained": time.strftime("%Y-%m-%d"),
        },
    }
    joblib.dump(bundle, MODEL_OUT, compress=3)
    os.makedirs(PROCESSED, exist_ok=True)
    test_uci[["label", "message"]].to_csv(os.path.join(PROCESSED, "sms_test.csv"), index=False)
    print(f"Saved {MODEL_OUT} ({os.path.getsize(MODEL_OUT) / 1e6:.2f} MB). Run models/evaluate_models.py for metrics.")


if __name__ == "__main__":
    main()
