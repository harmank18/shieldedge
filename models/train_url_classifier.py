"""
Train the ShieldEdge URL phishing model (stacked char-n-gram LR + structural features).

Data      : Mitake/PhishingURLsANDBenignURLs (Hugging Face), de-duplicated,
            150k benign + 75k phishing sampled (2:1, closer to real traffic than 50/50).
Split     : grouped by *registered domain* (70% train / 10% validation / 20% test),
            so no website appears in more than one split - a random split leaks
            ~50% of test domains into training and inflates every metric.
Model     : 1) char 3-5-gram TF-IDF -> LogisticRegression; its out-of-fold
               probability (GroupKFold, no leakage) becomes one input of
            2) HistGradientBoosting over 22 structural features + LR probability.
Thresholds: chosen on the validation split only:
              PHISHING   -> validation FPR <= 1%
              SUSPICIOUS -> validation FPR <= 4%
Outputs   : models/url_model.joblib, data/processed/url_test.csv (held-out split)

Run: python models/train_url_classifier.py
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.download_datasets import MITAKE_REPO, mitake_urls_path  # noqa: E402
from shieldedge.url_detector import (FEATURE_NAMES, extract_url_features, feature_vector,  # noqa: E402
                                     normalise_url, split_host)

MODEL_OUT = os.path.join(BASE_DIR, "models", "url_model.joblib")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
SEED = 42
N_BENIGN, N_PHISH = 150_000, 75_000


def threshold_for_fpr(y, p, max_fpr):
    """Smallest threshold whose false-positive rate on (y, p) is <= max_fpr."""
    neg = np.sort(p[y == 0])[::-1]
    k = int(np.floor(max_fpr * len(neg)))
    return float(neg[k] + 1e-6) if k < len(neg) else 0.0


def make_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000,
                           sublinear_tf=True, dtype=np.float32)


def main():
    t0 = time.time()
    df = pd.read_csv(mitake_urls_path()).dropna(subset=["url"])
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""].drop_duplicates("url")
    df = pd.concat([
        df[df.label == 0].sample(N_BENIGN, random_state=SEED),
        df[df.label == 1].sample(N_PHISH, random_state=SEED),
    ]).reset_index(drop=True)
    print(f"Loaded {len(df):,} URLs ({N_BENIGN:,} benign / {N_PHISH:,} phishing)")

    parsed = [normalise_url(u) for u in df.url]
    df["norm"] = [p["norm"] for p in parsed]
    df["domain"] = [split_host(p["host"])[0] or p["host"] for p in parsed]
    y = df.label.values.astype(int)
    X_feat = np.array([feature_vector(extract_url_features(u)) for u in df.url], dtype=float)
    print(f"Features extracted in {time.time() - t0:.0f}s")

    idx = np.arange(len(df))
    rest, test = next(GroupShuffleSplit(1, test_size=0.20, random_state=SEED).split(idx, y, df.domain))
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.125, random_state=SEED).split(rest, y[rest], df.domain.values[rest]))
    train, val = rest[tr_i], rest[va_i]
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:<5} {len(part):>7,} rows, {df.domain.values[part].size and len(set(df.domain.values[part])):>7,} domains, "
              f"{y[part].mean():.1%} phishing")
    assert not set(df.domain.values[test]) & set(df.domain.values[train]), "domain leakage"

    # 1) out-of-fold char-n-gram probabilities for the stacker
    oof = np.zeros(len(train))
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(train, y[train], df.domain.values[train])):
        vec = make_vectorizer()
        lr = LogisticRegression(C=4, max_iter=2000, class_weight="balanced")
        lr.fit(vec.fit_transform(df.norm.values[train][a]), y[train][a])
        oof[b] = lr.predict_proba(vec.transform(df.norm.values[train][b]))[:, 1]
        print(f"  OOF fold {k + 1}/5 done ({time.time() - t0:.0f}s)")
    vec = make_vectorizer()
    char_lr = LogisticRegression(C=4, max_iter=2000, class_weight="balanced")
    char_lr.fit(vec.fit_transform(df.norm.values[train]), y[train])

    # 2) stacker
    stacker = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=48,
                                             l2_regularization=1.0, random_state=SEED)
    stacker.fit(np.column_stack([X_feat[train], oof]), y[train])

    def predict(rows):
        lr_p = char_lr.predict_proba(vec.transform(df.norm.values[rows]))[:, 1]
        return stacker.predict_proba(np.column_stack([X_feat[rows], lr_p]))[:, 1]

    p_val = predict(val)
    t_phish = threshold_for_fpr(y[val], p_val, 0.01)
    t_susp = threshold_for_fpr(y[val], p_val, 0.04)
    print(f"Validation AUC {roc_auc_score(y[val], p_val):.4f} | thresholds: PHISHING>={t_phish:.3f}, SUSPICIOUS>={t_susp:.3f}")

    bundle = {
        "vectorizer": vec,
        "char_lr": char_lr,
        "stacker": stacker,
        "feature_names": FEATURE_NAMES,
        "threshold_phishing": t_phish,
        "threshold_suspicious": t_susp,
        "meta": {
            "dataset": f"{MITAKE_REPO[0]} ({N_BENIGN} benign / {N_PHISH} phishing sampled, seed {SEED})",
            "split": "GroupShuffleSplit by registered domain: 70/10/20",
            "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
            "sklearn": sklearn.__version__,
            "trained": time.strftime("%Y-%m-%d"),
        },
    }
    joblib.dump(bundle, MODEL_OUT, compress=3)
    os.makedirs(PROCESSED, exist_ok=True)
    df.iloc[test][["url", "label", "domain"]].to_csv(os.path.join(PROCESSED, "url_test.csv"), index=False)
    print(f"Saved {MODEL_OUT} ({os.path.getsize(MODEL_OUT) / 1e6:.1f} MB) and held-out test split "
          f"({len(test):,} rows) in {time.time() - t0:.0f}s")
    print("Run models/evaluate_models.py for the held-out metrics.")


if __name__ == "__main__":
    main()
