"""
Train the ShieldEdge URL phishing model (stacked char-n-gram LR + structural features).

Data      : Mitake/PhishingURLsANDBenignURLs (Hugging Face), de-duplicated, at most
            50 URLs per registered domain (otherwise a few giant sites such as
            doubleclick.net, with ~3,000 near-identical URLs each, dominate whichever
            split they land in and swing the false-positive rate by several points).
Split     : by a fixed hash of the *registered domain* (70% train / 10% validation /
            20% test), so no website appears in more than one split (a random split
            leaks ~50% of test domains into training) and the split does not depend
            on the training-set size. Train: 210k benign + 105k phishing (2:1);
            validation and test keep every phishing URL plus twice as many benign.
Model     : 1) char 3-5-gram + word 1-2-gram TF-IDF -> LogisticRegression; its out-of-fold
               probability (GroupKFold, no leakage) becomes one input of
            2) HistGradientBoosting over 22 structural features + LR probability.
Thresholds: chosen on the validation split only:
              PHISHING   -> validation FPR <= 1%
              SUSPICIOUS -> validation FPR <= 4%
Choices   : on the validation split only (fixed domain-grouped split, test untouched),
            going from 225k char-only to 450k char+word raised recall at 1% FPR from
            0.816 to 0.858 (AUC 0.985 -> 0.988); using all 606k gave no further gain.
Outputs   : models/url_model.joblib, data/processed/url_test.csv (held-out split)

Run: python models/train_url_classifier.py
"""

import argparse
import hashlib
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from data.download_datasets import MITAKE_REPO, mitake_urls_path  # noqa: E402
from shieldedge.url_detector import (FEATURE_NAMES, extract_url_features, feature_vector,  # noqa: E402
                                     normalise_url, split_host)

MODEL_OUT = os.path.join(BASE_DIR, "models", "url_model.joblib")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
SEED = 42
TRAIN_BENIGN, TRAIN_PHISH = 210_000, 105_000
MAX_PER_DOMAIN = 50


def threshold_for_fpr(y, p, max_fpr):
    """Smallest threshold whose false-positive rate on (y, p) is <= max_fpr."""
    neg = np.sort(p[y == 0])[::-1]
    k = int(np.floor(max_fpr * len(neg)))
    return float(neg[k] + 1e-6) if k < len(neg) else 0.0


def domain_bucket(domain: str) -> int:
    """Stable 0-99 bucket for a registered domain (same split on every run and sample size)."""
    return int(hashlib.md5(domain.encode("utf-8")).hexdigest()[:8], 16) % 100


def make_vectorizer(words: bool = True):
    if not words:  # --baseline: the previous char-only model
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000,
                               sublinear_tf=True, dtype=np.float32)
    return FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000,
                                 sublinear_tf=True, dtype=np.float32)),
        # word tokens: login, secure, verify, brand names, file names ...
        ("word", TfidfVectorizer(token_pattern=r"[a-z0-9]{2,}", ngram_range=(1, 2), min_df=2,
                                 max_features=200_000, sublinear_tf=True, dtype=np.float32)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true",
                    help="previous configuration (char n-grams only, half the training data) for comparison; "
                         "writes models/url_model_baseline.joblib")
    args = ap.parse_args()
    words = not args.baseline
    n_benign, n_phish = (TRAIN_BENIGN // 2, TRAIN_PHISH // 2) if args.baseline else (TRAIN_BENIGN, TRAIN_PHISH)
    model_out = MODEL_OUT.replace(".joblib", "_baseline.joblib") if args.baseline else MODEL_OUT

    t0 = time.time()
    df = pd.read_csv(mitake_urls_path()).dropna(subset=["url"])
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""].drop_duplicates("url")
    df["domain"] = [split_host(normalise_url(u)["host"])[0] or normalise_url(u)["host"] for u in df.url]
    n_all = len(df)
    df = df.sample(frac=1, random_state=SEED)
    df = df[df.groupby("domain").cumcount() < MAX_PER_DOMAIN]
    bucket = df.domain.map(domain_bucket)
    df["split"] = np.where(bucket < 20, "test", np.where(bucket < 30, "val", "train"))
    rs = np.random.RandomState(SEED)

    def take(part, nb, npos):
        d = df[df.split == part]
        b, p = d[d.label == 0], d[d.label == 1]
        npos = min(npos, len(p)) if npos else len(p)
        nb = min(nb, len(b)) if nb else min(len(b), 2 * npos)
        return pd.concat([b.sample(nb, random_state=rs), p.sample(npos, random_state=rs)])

    df = pd.concat([take("train", n_benign, n_phish), take("val", None, None), take("test", None, None)])
    df = df.reset_index(drop=True)
    print(f"{n_all:,} unique URLs -> max {MAX_PER_DOMAIN}/domain -> {len(df):,} used "
          f"({'baseline: char only' if args.baseline else 'char + word'})")

    df["norm"] = [normalise_url(u)["norm"] for u in df.url]
    y = df.label.values.astype(int)
    X_feat = np.array([feature_vector(extract_url_features(u)) for u in df.url], dtype=float)
    print(f"Features extracted in {time.time() - t0:.0f}s")

    train, val, test = (np.flatnonzero(df.split.values == s) for s in ("train", "val", "test"))
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:<5} {len(part):>7,} rows, {df.domain.values[part].size and len(set(df.domain.values[part])):>7,} domains, "
              f"{y[part].mean():.1%} phishing")
    assert not set(df.domain.values[test]) & set(df.domain.values[train]), "domain leakage"

    # 1) out-of-fold char-n-gram probabilities for the stacker
    oof = np.zeros(len(train))
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(train, y[train], df.domain.values[train])):
        vec = make_vectorizer(words)
        lr = LogisticRegression(C=4, max_iter=2000, class_weight="balanced")
        lr.fit(vec.fit_transform(df.norm.values[train][a]), y[train][a])
        oof[b] = lr.predict_proba(vec.transform(df.norm.values[train][b]))[:, 1]
        print(f"  OOF fold {k + 1}/5 done ({time.time() - t0:.0f}s)")
    vec = make_vectorizer(words)
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
            "dataset": f"{MITAKE_REPO[0]} (max {MAX_PER_DOMAIN}/domain; train {n_benign} benign / {n_phish} phishing)",
            "split": "fixed md5 bucket of the registered domain: 70/10/20",
            "features": "char 3-5-grams + word 1-2-grams" if words else "char 3-5-grams (baseline)",
            "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
            "sklearn": sklearn.__version__,
            "trained": time.strftime("%Y-%m-%d"),
        },
    }
    joblib.dump(bundle, model_out, compress=3)
    os.makedirs(PROCESSED, exist_ok=True)
    df.iloc[test][["url", "label", "domain"]].to_csv(os.path.join(PROCESSED, "url_test.csv"), index=False)
    print(f"Saved {model_out} ({os.path.getsize(model_out) / 1e6:.1f} MB) and held-out test split "
          f"({len(test):,} rows) in {time.time() - t0:.0f}s")
    print("Run models/evaluate_models.py for the held-out metrics.")


if __name__ == "__main__":
    main()
