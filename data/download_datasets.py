"""
Reproducible dataset acquisition for ShieldEdge.

Every dataset used for training/evaluation is a public, labelled benchmark that
is downloaded (and cached) by this script - nothing here is generated:

  URL   Mitake/PhishingURLsANDBenignURLs (Hugging Face)      ~886k labelled URLs
  SMS   UCI SMS Spam Collection (Almeida & Hidalgo, 2011)    5,574 messages
  NET   bvk/CICIDS-2017 (Hugging Face; improved CIC-IDS-2017 labels) Monday + Friday

Small team-authored India-specific sets live in data/curated/ and are clearly
marked as such; they are *not* downloaded here.

Usage:  python data/download_datasets.py          # fetch everything into the HF cache / data/raw
"""

import os
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw")
CURATED_DIR = os.path.join(DATA_DIR, "curated")

MITAKE_REPO = ("Mitake/PhishingURLsANDBenignURLs", "PhishBenignDataset.csv")
CICIDS_REPO = "bvk/CICIDS-2017"
UCI_SMS_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"  # mirror of UCI SMS Spam Collection


def _hf_file(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


def mitake_urls_path() -> str:
    """Path to the Mitake phishing/benign URL CSV (columns: url, label[0 benign, 1 phishing])."""
    return _hf_file(*MITAKE_REPO)


def cicids_path(day: str) -> str:
    """Path to a CICIDS-2017 day CSV ('monday' or 'friday')."""
    return _hf_file(CICIDS_REPO, f"{day}.csv")


def uci_sms_path() -> str:
    """Path to the UCI SMS Spam Collection as TSV (label<TAB>message)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, "sms.tsv")
    if not os.path.exists(path):
        print(f"[SMS] downloading UCI SMS Spam Collection -> {path}")
        urllib.request.urlretrieve(UCI_SMS_URL, path)
    return path


def main():
    print("[URL]", mitake_urls_path())
    print("[SMS]", uci_sms_path())
    for day in ("monday", "friday"):
        print(f"[NET:{day}]", cicids_path(day))
    print("All datasets available. Curated India-specific sets:", CURATED_DIR)


if __name__ == "__main__":
    main()
