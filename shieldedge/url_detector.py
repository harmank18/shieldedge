"""
URL Feature Extraction and Threat Detection Module for ShieldEdge.
Extracts lexical and structural indicators without making network calls.
"""

import re
import os
import joblib
import numpy as np
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "click", "win", "online", "vip", "site", 
    "tk", "ml", "cf", "gq", "biz", "info", "live"
}

POPULAR_BRANDS = [
    "sbi", "onlinesbi", "hdfc", "icici", "axis", "punjabnational", 
    "rbi", "uidai", "aadhaar", "incometax", "paytm", "phonepe", 
    "gpay", "amazon", "flipkart", "irctc", "trai"
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly", "ow.ly"
}

FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "dot_count",
    "has_ip",
    "is_https",
    "suspicious_tld",
    "brand_in_domain",
    "hyphen_count",
    "digit_ratio",
    "directory_depth",
    "has_shortener",
    "subdomain_count",
    "special_char_count"
]


def extract_url_features(url: str) -> dict:
    """Extracts lexical and structural features from a URL string."""
    if not url.startswith(("http://", "https://")):
        url_parsed = urlparse("http://" + url)
    else:
        url_parsed = urlparse(url)

    domain = url_parsed.netloc.lower().split(":")[0]
    path = url_parsed.path

    # Check for IP address
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    has_ip = 1 if re.match(ip_pattern, domain) else 0

    # Scheme
    is_https = 1 if url_parsed.scheme == "https" else 0

    # TLD extraction
    tld = domain.split(".")[-1] if "." in domain else ""
    suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0

    # Brand impersonation check
    brand_found = 0
    for brand in POPULAR_BRANDS:
        if brand in domain:
            # Check if domain is an official domain for that brand
            legit_brand_domains = [f"{brand}.com", f"{brand}.in", f"{brand}.co.in", f"{brand}.gov.in", f"{brand}.sbi"]
            if not any(domain.endswith(legit) for legit in legit_brand_domains):
                brand_found = 1
                break

    # Counts and ratios
    url_len = len(url)
    domain_len = len(domain)
    dot_count = url.count(".")
    hyphen_count = url.count("-")
    digits = sum(c.isdigit() for c in url)
    digit_ratio = digits / max(1, url_len)
    depth = len([p for p in path.split("/") if p])
    has_shortener = 1 if domain in SHORTENER_DOMAINS else 0
    subdomains = max(0, len(domain.split(".")) - 2)
    specials = sum(url.count(c) for c in ["@", "%", "=", "?", "&", "_", "!"])

    features = {
        "url_length": url_len,
        "domain_length": domain_len,
        "dot_count": dot_count,
        "has_ip": has_ip,
        "is_https": is_https,
        "suspicious_tld": suspicious_tld,
        "brand_in_domain": brand_found,
        "hyphen_count": hyphen_count,
        "digit_ratio": round(digit_ratio, 4),
        "directory_depth": depth,
        "has_shortener": has_shortener,
        "subdomain_count": subdomains,
        "special_char_count": specials
    }
    return features


class URLDetector:
    def __init__(self, model_path: str = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def predict(self, url: str) -> dict:
        features = extract_url_features(url)
        signals = []

        if features["is_https"] == 0:
            signals.append("Insecure HTTP protocol (no SSL/TLS)")
        if features["suspicious_tld"] == 1:
            signals.append("Uses high-risk, cheap/free TLD frequently abused in scams")
        if features["brand_in_domain"] == 1:
            signals.append("Well-known brand name detected inside an unofficial third-party domain")
        if features["has_ip"] == 1:
            signals.append("Uses bare IP address instead of registered domain name")
        if features["has_shortener"] == 1:
            signals.append("Uses URL shortener obscuring final destination")
        if features["digit_ratio"] > 0.15:
            signals.append(f"Unusually high digit ratio ({int(features['digit_ratio']*100)}%) in URL")
        if features["hyphen_count"] >= 2:
            signals.append("Multiple hyphens in domain, typical of domain spoofing")

        if self.model is not None:
            feat_vector = np.array([[features[k] for k in FEATURE_NAMES]])
            proba = self.model.predict_proba(feat_vector)[0]
            # Assuming class 1 is phishing
            phish_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # Deterministic heuristic fallback if model not loaded
            score = 0
            if features["is_https"] == 0: score += 0.2
            if features["suspicious_tld"]: score += 0.35
            if features["brand_in_domain"]: score += 0.4
            if features["has_shortener"]: score += 0.25
            if features["has_ip"]: score += 0.4
            phish_conf = min(0.99, max(0.01, score))

        if phish_conf >= 0.75:
            verdict = "PHISHING"
        elif phish_conf >= 0.50:
            verdict = "SUSPICIOUS"
        else:
            verdict = "SAFE"

        return {
            "input_type": "URL",
            "input": url,
            "verdict": verdict,
            "confidence": round(phish_conf if verdict != "SAFE" else 1.0 - phish_conf, 3),
            "features": features,
            "signals": signals
        }
