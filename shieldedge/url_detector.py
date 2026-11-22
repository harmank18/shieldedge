"""
URL phishing detector for ShieldEdge (fully offline - no DNS/WHOIS/HTTP calls).

Pipeline
  1. normalise_url(): lower-case, strip scheme and leading "www." so that the
     model sees the same representation at training and inference time
     (the training corpus stores URLs without a scheme).
  2. Two views of the URL:
       * character 3-5-gram TF-IDF  -> logistic regression (lexical "shape")
       * 22 hand-crafted structural features (length, TLD risk, brand misuse,
         entropy, free-hosting, ...)
     A gradient-boosted model combines both (stacking).
  3. Rule layer:
       * official-domain allowlist for high-value Indian brands/government
         sites (never applied to free/shared hosting platforms),
       * HTTPS / IP / shortener signals for the explanation.

Every signal string is written in plain language because it is what the
on-device Llama explains to the user.
"""

import math
import os
import re
from urllib.parse import urlsplit

import joblib
import numpy as np

# ----------------------------------------------------------------------------- reference lists
MULTI_PART_TLDS = {
    "co.in", "gov.in", "net.in", "org.in", "ac.in", "edu.in", "mil.in", "nic.in", "res.in", "firm.in", "gen.in", "ind.in",
    "co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "net.au", "org.au", "co.jp", "co.nz", "com.br", "com.sg", "com.cn",
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "click", "win", "online", "vip", "site", "tk", "ml", "cf", "gq", "ga", "biz", "info", "live",
    "icu", "pw", "cc", "work", "link", "mobi", "zip", "mov", "fit", "uno", "fun", "rest", "shop", "store", "world",
    "today", "space", "life", "cyou", "bond", "cfd", "sbs", "monster", "quest", "bar", "loan", "review", "buzz", "ru", "cn",
}

# Official registered domains per brand/agency (verified manually). A brand name
# appearing in any *other* registered domain is treated as impersonation.
OFFICIAL_DOMAINS = {
    "sbi": {"onlinesbi.sbi", "sbi.co.in", "sbi.bank.in", "sbicard.com", "sbilife.co.in", "yonosbi.com"},
    "hdfc": {"hdfcbank.com", "hdfc.com", "hdfclife.com", "hdfcergo.com", "hdfcbank.bank.in"},
    "icici": {"icicibank.com", "icicidirect.com", "iciciprulife.com", "icicilombard.com"},
    "axis": {"axisbank.com", "axisbank.co.in", "axisdirect.in"},
    "kotak": {"kotak.com", "kotaksecurities.com"},
    "pnb": {"pnbindia.in", "pnb.co.in", "netpnb.com"},
    "canarabank": {"canarabank.com", "canarabank.in"},
    "bankofbaroda": {"bankofbaroda.in", "bankofbaroda.com", "bobibanking.com"},
    "yesbank": {"yesbank.in"},
    "indusind": {"indusind.com"},
    "idbi": {"idbibank.in"},
    "rbi": {"rbi.org.in"},
    "npci": {"npci.org.in"},
    "uidai": {"uidai.gov.in"},
    "aadhaar": {"uidai.gov.in"},
    "incometax": {"incometax.gov.in", "incometaxindia.gov.in"},
    "epfo": {"epfindia.gov.in"},
    "nsdl": {"nsdl.co.in", "nsdl.com", "egov-nsdl.co.in"},
    "digilocker": {"digilocker.gov.in"},
    "irctc": {"irctc.co.in", "irctc.com"},
    "lic": {"licindia.in"},
    "trai": {"trai.gov.in"},
    "paytm": {"paytm.com", "paytm.in", "paytmbank.com"},
    "phonepe": {"phonepe.com"},
    "gpay": {"pay.google.com"},
    "bhim": {"bhimupi.org.in"},
    "amazon": {"amazon.in", "amazon.com", "amazonpay.in", "amzn.in", "amzn.to"},
    "flipkart": {"flipkart.com"},
    "myntra": {"myntra.com"},
    "meesho": {"meesho.com"},
    "airtel": {"airtel.in", "airtel.com"},
    "jio": {"jio.com", "jiomart.com", "jiocinema.com"},
    "bsnl": {"bsnl.co.in", "bsnl.in"},
    "zomato": {"zomato.com"},
    "swiggy": {"swiggy.com", "swiggy.in"},
    "zerodha": {"zerodha.com"},
    "groww": {"groww.in"},
    "whatsapp": {"whatsapp.com", "whatsapp.net", "wa.me"},
    "instagram": {"instagram.com"},
    "facebook": {"facebook.com", "fb.com"},
    "netflix": {"netflix.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "microsoftonline.com", "outlook.com"},
    "apple": {"apple.com", "icloud.com"},
    "google": {"google.com", "google.co.in", "youtube.com", "gmail.com"},
    "paypal": {"paypal.com"},
}
ALL_OFFICIAL = set().union(*OFFICIAL_DOMAINS.values()) | {"gov.in", "nic.in", "india.gov.in", "cybercrime.gov.in", "sancharsaathi.gov.in"}

# Platforms where anyone can publish content: never trusted by the allowlist.
SHARED_HOSTING = (
    "sites.google.com", "docs.google.com", "forms.gle", "drive.google.com", "storage.googleapis.com",
    "weebly.com", "wixsite.com", "web.app", "firebaseapp.com", "github.io", "blogspot.com", "000webhostapp.com",
    "netlify.app", "vercel.app", "pages.dev", "workers.dev", "glitch.me", "herokuapp.com", "duckdns.org",
    "ngrok.io", "ngrok-free.app", "repl.co", "azurewebsites.net", "blob.core.windows.net", "s3.amazonaws.com",
    "r2.dev", "webflow.io", "square.site", "godaddysites.com", "yolasite.com", "jimdosite.com", "tiiny.site",
)

SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly", "ow.ly", "rb.gy", "cutt.ly",
                     "shorturl.at", "tiny.cc", "t.ly", "s.id", "v.gd", "short.gy"}

AUTH_KEYWORDS = ("login", "verify", "secure", "update", "confirm", "account", "signin", "banking", "netbanking",
                 "credential", "password", "otp", "authenticate", "validation", "kyc", "identity", "wallet", "reward")

HOMOGLYPH_DIGITS = set("013456789")

FEATURE_NAMES = [
    "url_length", "host_length", "dot_count", "has_ip", "suspicious_tld", "brand_impersonation",
    "hyphen_count", "digit_ratio", "directory_depth", "has_shortener", "subdomain_count", "special_char_count",
    "has_at_symbol", "has_double_slash_in_path", "is_punycode", "domain_entropy", "query_param_count",
    "path_has_auth_keywords", "consecutive_hyphens", "tld_in_subdomain", "homoglyph_score", "free_hosting",
]

_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_TOKEN_RE = re.compile(r"[a-z]+")
_GLUE_WORDS = {"bank", "banking", "net", "netbanking", "online", "card", "cards", "kyc", "pay", "upi", "care", "help",
               "support", "reward", "rewards", "update", "verify", "login", "secure", "india", "in", "customer", "app",
               "yono", "life", "loan", "refund", "points", "offer", "offers", "sim"}


# ----------------------------------------------------------------------------- parsing helpers
def normalise_url(url: str) -> dict:
    """Parse a URL without ever raising. Returns scheme, host, path, query and the normalised string."""
    raw = (url or "").strip()
    scheme = ""
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.\-]*)://", raw)
    if m:
        scheme = m.group(1).lower()
        rest = raw[m.end():]
    else:
        rest = raw
    rest_l = rest.lower()
    try:
        parts = urlsplit("http://" + rest)
        host = (parts.hostname or "").lower()
        path, query = parts.path, parts.query
    except ValueError:  # e.g. malformed IPv6 brackets
        host = re.split(r"[/?#:]", rest_l, 1)[0].strip("[]")
        after = rest[len(host):]
        path, _, query = after.partition("?")
    if host.startswith("www."):
        host = host[4:]
    norm = rest_l[4:] if rest_l.startswith("www.") else rest_l
    return {"scheme": scheme, "host": host, "path": path or "", "query": query or "", "norm": norm[:300]}


def split_host(host: str):
    """Return (registered_domain, subdomain_labels, tld)."""
    labels = [p for p in host.split(".") if p]
    if len(labels) < 2:
        return host, [], labels[-1] if labels else ""
    tld = labels[-1]
    n = 2
    if len(labels) >= 3 and f"{labels[-2]}.{labels[-1]}" in MULTI_PART_TLDS:
        tld, n = f"{labels[-2]}.{labels[-1]}", 3
    return ".".join(labels[-n:]), labels[:-n], tld


def is_shared_hosting(host: str) -> bool:
    return any(host == s or host.endswith("." + s) for s in SHARED_HOSTING)


def official_brand(host: str):
    """Brand name if the host is an official domain (and not shared hosting), else None."""
    if is_shared_hosting(host):
        return None
    for brand, domains in OFFICIAL_DOMAINS.items():
        if any(host == d or host.endswith("." + d) for d in domains):
            return brand
    if host.endswith(".gov.in") or host.endswith(".nic.in"):
        return "government of India"
    return None


def impersonated_brand(host: str):
    """Brand whose name appears in the host although the host is not one of its official domains."""
    if official_brand(host) or split_host(host)[0] in ALL_OFFICIAL:
        return None  # official site, or user content on the brand's own platform (e.g. docs.google.com)
    tokens = _TOKEN_RE.findall(host)
    joined = "".join(tokens)
    for brand in OFFICIAL_DOMAINS:
        if len(brand) <= 4:
            # Short names (sbi, lic, jio...) must be a whole token or glued to a scam word
            # ("sbikyc", "hdfcbank-login") - avoids matching "license" or "public".
            hit = brand in tokens or any(t.startswith(brand) and t[len(brand):] in _GLUE_WORDS for t in tokens)
        else:
            hit = brand in joined
        if hit:
            return brand
    return None


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {c: s.count(c) for c in set(s)}
    return -sum(v / len(s) * math.log2(v / len(s)) for v in freq.values())


# ----------------------------------------------------------------------------- features
def extract_url_features(url: str) -> dict:
    """22 structural features computed on the normalised URL (scheme-independent)."""
    p = normalise_url(url)
    host, path, query, norm = p["host"], p["path"], p["query"], p["norm"]
    reg, subs, tld = split_host(host)
    name = reg.split(".")[0] if reg else host
    tld_last = host.rsplit(".", 1)[-1] if "." in host else ""
    sub_str = ".".join(subs)

    return {
        "url_length": len(norm),
        "host_length": len(host),
        "dot_count": norm.count("."),
        "has_ip": int(bool(_IP_RE.match(host))),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS or tld_last in SUSPICIOUS_TLDS),
        "brand_impersonation": int(impersonated_brand(host) is not None),
        "hyphen_count": host.count("-"),
        "digit_ratio": round(sum(c.isdigit() for c in norm) / max(1, len(norm)), 4),
        "directory_depth": len([s for s in path.split("/") if s]),
        "has_shortener": int(host in SHORTENER_DOMAINS),
        "subdomain_count": len(subs),
        "special_char_count": sum(norm.count(c) for c in "@%=?&_!~"),
        "has_at_symbol": int("@" in norm),
        "has_double_slash_in_path": int("//" in path),
        "is_punycode": int("xn--" in host),
        "domain_entropy": round(_entropy(name), 4),
        "query_param_count": (query.count("&") + 1) if query else 0,
        "path_has_auth_keywords": int(any(k in (path + "?" + query).lower() for k in AUTH_KEYWORDS)),
        "consecutive_hyphens": int("--" in host),
        "tld_in_subdomain": int(bool(re.search(r"(^|[.\-])(com|net|org|gov|in)([.\-]|$)", sub_str))),
        "homoglyph_score": round(sum(c in HOMOGLYPH_DIGITS for c in name) / max(1, len(name)), 4),
        "free_hosting": int(is_shared_hosting(host)),
    }


def feature_vector(features: dict) -> list:
    return [features[k] for k in FEATURE_NAMES]


def explain_signals(url: str, f: dict) -> list:
    p = normalise_url(url)
    host = p["host"]
    reg, _, tld = split_host(host)
    s = []
    if p["scheme"] == "http":
        s.append("Uses plain HTTP, so the connection is not encrypted")
    if f["has_ip"]:
        s.append("Uses a raw IP address instead of a website name")
    brand = impersonated_brand(host)
    if brand:
        s.append(f"Uses the name '{brand.upper()}' but {host} is not an official {brand.upper()} website")
    if f["suspicious_tld"]:
        s.append(f"Uses a cheap domain ending (.{tld}) that is common in scams")
    if f["free_hosting"]:
        s.append("Hosted on a free website-builder or file-sharing platform that anyone can use")
    if f["has_shortener"]:
        s.append("Shortened link that hides the real destination")
    if f["has_at_symbol"]:
        s.append("Contains an @ sign, a trick used to hide the real website")
    if f["is_punycode"]:
        s.append("Uses special look-alike characters in the website name")
    if f["hyphen_count"] >= 2:
        s.append("Website name is made of several hyphenated words, typical of fake sites")
    if f["homoglyph_score"] > 0.15 and not f["has_ip"]:
        s.append("Website name swaps letters for look-alike numbers (e.g. 0 for o)")
    if f["subdomain_count"] >= 3 or f["tld_in_subdomain"]:
        s.append("Long chain of sub-domains that can disguise the real website")
    if f["path_has_auth_keywords"]:
        s.append("Address contains words like login, verify, KYC or update")
    if f["url_length"] > 100:
        s.append("Unusually long address")
    return s


# ----------------------------------------------------------------------------- detector
class URLDetector:
    """Loads the stacked model bundle written by models/train_url_classifier.py."""

    def __init__(self, model_path: str = None):
        self.bundle = None
        if model_path and os.path.exists(model_path):
            obj = joblib.load(model_path)
            if isinstance(obj, dict) and "stacker" in obj:
                self.bundle = obj
        self.t_phish = self.bundle["threshold_phishing"] if self.bundle else 0.7
        self.t_susp = self.bundle["threshold_suspicious"] if self.bundle else 0.45

    def model_scores(self, urls, feature_dicts) -> np.ndarray:
        """Raw phishing probabilities for a batch (heuristic fallback if no model)."""
        if self.bundle is None:
            w = {"suspicious_tld": .3, "brand_impersonation": .4, "has_ip": .35, "has_shortener": .25,
                 "has_at_symbol": .3, "is_punycode": .35, "free_hosting": .25, "path_has_auth_keywords": .15}
            return np.array([min(0.99, 0.05 + sum(v for k, v in w.items() if f[k])) for f in feature_dicts])
        norms = [normalise_url(u)["norm"] for u in urls]
        lr_p = self.bundle["char_lr"].predict_proba(self.bundle["vectorizer"].transform(norms))[:, 1]
        X = np.column_stack([np.array([feature_vector(f) for f in feature_dicts], dtype=float), lr_p])
        return self.bundle["stacker"].predict_proba(X)[:, 1]

    def score(self, url: str, features: dict = None) -> float:
        f = features or extract_url_features(url)
        return float(self.model_scores([url], [f])[0])

    def _finalise(self, url: str, f: dict, p: float) -> dict:
        """Apply the rule layer and thresholds to a model probability (shared by single and batch paths)."""
        host = normalise_url(url)["host"]
        signals = explain_signals(url, f)
        if not host or ("." not in host and host != "localhost"):
            return {"input_type": "URL", "input": url, "verdict": "SUSPICIOUS", "confidence": 0.5,
                    "score": 0.5, "features": f, "signals": ["The address could not be read as a normal web link"]}

        # Explainable rule layer on top of the model.
        if f["brand_impersonation"]:
            risky_host = f["suspicious_tld"] or f["free_hosting"] or f["has_ip"]
            p = max(p, self.t_phish if risky_host else self.t_susp)
        if f["has_shortener"]:
            # Destination is unknown, not proven malicious: warn but don't call it phishing.
            p = min(max(p, self.t_susp), self.t_phish - 1e-3)

        official = official_brand(host)
        if official:
            # Allowlisted official domain: never call it phishing, but keep HTTP warning if any.
            p = min(p, 0.05)
            signals = [f"Official {official.upper() if len(official) < 12 else official} domain"] + \
                      [s for s in signals if s.startswith("Uses plain HTTP")]

        if p >= self.t_phish:
            verdict = "PHISHING"
        elif p >= self.t_susp:
            verdict = "SUSPICIOUS"
        else:
            verdict = "SAFE"
        if verdict == "SAFE":
            signals = [s for s in signals if not s.startswith("Address contains words")]

        return {
            "input_type": "URL",
            "input": url,
            "verdict": verdict,
            "score": round(float(p), 4),
            "confidence": round(float(p) if verdict != "SAFE" else 1.0 - float(p), 3),
            "features": f,
            "signals": signals,
        }

    def predict(self, url: str) -> dict:
        f = extract_url_features(url)
        return self._finalise(url, f, self.score(url, f))

    def predict_batch(self, urls) -> list:
        """Same decisions as predict(), vectorised for evaluation."""
        feats = [extract_url_features(u) for u in urls]
        probs = self.model_scores(urls, feats)
        return [self._finalise(u, f, p) for u, f, p in zip(urls, feats, probs)]
