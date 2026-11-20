"""
Train the ShieldEdge SMS scam classifier.

Data (training parts only; every test part is held out):
  * UCI SMS Spam Collection, exact duplicates removed before splitting (80% train+val / 20% test,
    the same test split as earlier versions; 15% of the rest is validation).
  * Real smishing (rasheq/scam-shield-sms, real sources only): IMC 2025 public smishing reports
    (scams) and the Mishra & Soni SMS phishing dataset (both labels). Its own train/val/test
    split is kept.
  * rasheq synthetic messages (AI-written OTP/bank/delivery "safe" service messages and
    hard cases), training only - they teach the model what genuine service SMS look like.
  * Ridham115 Indian SMS (AI-generated, partly audited; English, Hinglish, Hindi, Tamil, Telugu,
    Bengali, Marathi), split 40% train / 20% validation / 40% test, stratified.
  * data/curated/india_sms_train.csv (team-authored Indian messages).
Leakage control: any training message whose normalised, masked text equals a validation or test
message (of any dataset) is dropped. Mishra & Soni partly overlaps UCI.

Model : links/phones/e-mails/long numbers masked (sms_detector.mask_for_model), then TF-IDF word
        1-2-grams + char_wb 2-5-grams -> LinearSVC with sigmoid calibration (5 folds).
Choice: on validation only. At a pooled 3% false-alarm rate, adding the real smishing reports,
        synthetic safe messages and the Indian training part raised recall on real 2025 smishing
        from 0.35 to 0.94 and on Indian messages from 0.51 to 0.90 (UCI: 0.95 -> 0.91).
Thresholds (validation, pooled over the UCI, smishing and Indian validation sets):
        SUSPICIOUS -> pooled validation FPR 3%;  SCAM -> pooled validation FPR 1%.
Out   : models/sms_pipeline.joblib, data/processed/sms_test.csv (UCI),
        sms_test_smishing.csv, sms_test_indian_synthetic.csv

Run: python models/train_sms_classifier.py
"""

import os
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.download_datasets import CURATED_DIR, indian_sms_path, smishing_paths, uci_sms_path  # noqa: E402
from shieldedge.sms_detector import preprocess_text  # noqa: E402

MODEL_OUT = os.path.join(BASE_DIR, "models", "sms_pipeline.joblib")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
SEED = 42
REAL_SOURCES = ("imc2025", "mishra_soni")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", preprocess_text(str(t))).strip()


def load():
    uci = pd.read_csv(uci_sms_path(), sep="\t", header=None, names=["label", "text"])
    n_raw = len(uci)
    uci = uci.drop_duplicates("text").reset_index(drop=True)
    uci["y"] = (uci.label == "spam").astype(int)
    uci["source"] = "uci"
    print(f"UCI SMS: {n_raw} messages, {n_raw - len(uci)} exact duplicates removed -> {len(uci)}")
    u_rest, u_test = train_test_split(uci, test_size=0.2, random_state=SEED, stratify=uci.y)  # same test as before
    u_train, u_val = train_test_split(u_rest, test_size=0.15, random_state=0, stratify=u_rest.y)

    rq = {s: pd.read_csv(p) for s, p in smishing_paths().items()}
    for d in rq.values():
        d["y"] = (d.label == "scam").astype(int)
    real = {s: d[d.source.isin(REAL_SOURCES)] for s, d in rq.items()}
    synth = rq["train"][~rq["train"].source.isin(REAL_SOURCES)]

    ind = pd.read_csv(indian_sms_path())
    ind["y"] = (ind.label == "scam").astype(int)
    ind["source"] = "indian_synthetic"
    i_train, i_rest = train_test_split(ind, test_size=0.6, random_state=0, stratify=ind.y)
    i_val, i_test = train_test_split(i_rest, test_size=2 / 3, random_state=0, stratify=i_rest.y)

    team = pd.read_csv(os.path.join(CURATED_DIR, "india_sms_train.csv")).rename(columns={"message": "text"})
    team["y"] = (team.label == "spam").astype(int)
    team["source"] = "team_authored"
    team_test = pd.read_csv(os.path.join(CURATED_DIR, "india_sms_test.csv"))

    val = {"UCI": u_val, "smishing": real["val"], "indian": i_val}
    test = {"UCI": u_test, "smishing": real["test"], "indian": i_test}
    held = set(map(norm, pd.concat([*(d.text for d in val.values()), *(d.text for d in test.values()),
                                    team_test.message])))
    train = pd.concat([u_train, real["train"], synth, i_train, team], ignore_index=True)
    before = len(train)
    train = train[~train.text.map(norm).isin(held)].drop_duplicates("text")
    print(f"Train: {len(train):,} messages after removing {before - len(train)} that match a val/test message "
          f"or repeat ({train.y.mean():.0%} scam) | by source: {train.source.value_counts().to_dict()}")
    return train, val, test


def threshold_at_fpr(scores_neg, fpr):
    return float(np.quantile(scores_neg, 1 - fpr))


def main():
    t0 = time.time()
    train, val, test = load()
    pipeline = Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ])),
        ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=5, method="sigmoid")),
    ])
    pipeline.fit(train.text.astype(str).map(preprocess_text), train.y)

    scores = {k: pipeline.predict_proba(v.text.astype(str).map(preprocess_text))[:, 1] for k, v in val.items()}
    neg = np.concatenate([scores[k][val[k].y.values == 0] for k in val])
    t_susp, t_scam = threshold_at_fpr(neg, 0.03), threshold_at_fpr(neg, 0.01)
    for k, v in val.items():
        y = v.y.values.astype(bool)
        print(f"  validation {k:<9} recall@SUSPICIOUS {np.mean(scores[k][y] >= t_susp):.3f} "
              f"FPR {np.mean(scores[k][~y] >= t_susp):.3f}")
    print(f"Thresholds: SUSPICIOUS >= {t_susp:.3f}, SCAM >= {t_scam:.3f}")

    bundle = {
        "pipeline": pipeline,
        "threshold_scam": t_scam,
        "threshold_suspicious": t_susp,
        "mask_links": True,
        "meta": {
            "dataset": "UCI SMS (dedup) + real smishing (IMC 2025, Mishra & Soni) + rasheq synthetic service "
                       "messages + Ridham115 Indian synthetic (40%) + team-authored Indian SMS",
            "n_train": int(len(train)), "sklearn": sklearn.__version__, "trained": time.strftime("%Y-%m-%d"),
        },
    }
    joblib.dump(bundle, MODEL_OUT, compress=3)
    os.makedirs(PROCESSED, exist_ok=True)
    test["UCI"][["label", "text"]].rename(columns={"text": "message"}).to_csv(
        os.path.join(PROCESSED, "sms_test.csv"), index=False)
    test["smishing"].assign(label=np.where(test["smishing"].y == 1, "spam", "ham"))[
        ["label", "text", "source"]].rename(columns={"text": "message"}).to_csv(
        os.path.join(PROCESSED, "sms_test_smishing.csv"), index=False)
    test["indian"].assign(label=np.where(test["indian"].y == 1, "spam", "ham"))[
        ["label", "text", "language"]].rename(columns={"text": "message"}).to_csv(
        os.path.join(PROCESSED, "sms_test_indian_synthetic.csv"), index=False)
    print(f"Saved {MODEL_OUT} ({os.path.getsize(MODEL_OUT) / 1e6:.2f} MB) in {time.time() - t0:.0f}s. "
          "Run models/evaluate_models.py for the held-out metrics.")


if __name__ == "__main__":
    main()
