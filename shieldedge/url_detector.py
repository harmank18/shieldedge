"""
URL Feature Extraction and Threat Detection Module for ShieldEdge.
Extracts 22 lexical and structural indicators without making network calls.
Enhanced v2: Expanded TLDs, Indian brands, homoglyph detection, Shannon entropy,
punycode detection, multi-part ccTLD resolution, and spoofing detection.
"""

import re
import os
import math
import joblib
import numpy as np
from urllib.parse import urlparse

# Multi-part country code TLDs (e.g. .co.in, .gov.in, .co.uk)
MULTI_PART_TLDS = {
    "co.in", "gov.in", "net.in", "org.in", "ac.in", "edu.in", "mil.in", "nic.in", "res.in",
    "co.uk", "org.uk", "gov.uk", "ac.uk", "net.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.jp", "ne.jp", "or.jp", "go.jp", "ac.jp",
    "co.nz", "org.nz", "net.nz", "govt.nz",
    "com.br", "net.br", "gov.br",
    "com.sg", "edu.sg", "gov.sg",
}

# Suspicious / high-risk TLDs frequently abused in phishing and malware campaigns
SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "click", "win", "online", "vip", "site",
    "tk", "ml", "cf", "gq", "biz", "info", "live",
    "icu", "pw", "cc", "work", "link", "mobi", "name", "pro",
    "tel", "ws", "cat", "zip", "mov", "fit", "uno", "fun",
    "rest", "shop", "store", "world", "today", "tech", "space",
    "life", "cyou", "bond", "cfd", "sbs", "monster", "quest",
    "bar", "coupons", "ru", "cn", "ga", "loan", "review",
}

# Popular brands frequently targeted by Indian and international phishing
POPULAR_BRANDS = [
    # Indian Banks & Financial Services
    "sbi", "onlinesbi", "hdfc", "hdfcbank", "icici", "icicibank", "axis", "axisbank",
    "punjabnational", "pnb", "rbi", "uidai", "aadhaar", "incometax",
    "paytm", "phonepe", "gpay", "amazon", "flipkart", "irctc", "trai",
    # Additional High-Target Indian Entities
    "jio", "bsnl", "lic", "nsdl", "epfo", "digilocker",
    "zomato", "swiggy", "myntra", "meesho", "cred", "razorpay",
    "bajajfinserv", "kotak", "yesbank", "canarabank", "npci",
    "bhim", "mtnl", "airtel", "vodafone", "vi", "boi", "ubi", "iob",
    "idbi", "federalbank", "indusind", "ujjivan", "olx", "naukri",
    "linkedin", "instagram", "facebook", "whatsapp", "telegram",
    "netflix", "hotstar", "jiocinema", "zerodha", "groww", "upstox",
    "angelone", "pmkisan", "ayushman",
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd",
    "buff.ly", "ow.ly", "rb.gy", "cutt.ly", "shorturl.at"
}

AUTH_KEYWORDS = [
    "login", "verify", "secure", "update", "confirm", "account",
    "signin", "banking", "netbanking", "credential", "password",
    "otp", "authenticate", "validation", "kyc", "identity", "bill"
]

HOMOGLYPH_MAP = {
    "0": "o", "1": "l", "3": "e", "4": "a",
    "5": "s", "6": "g", "7": "t", "8": "b"
}

FEATURE_NAMES = [
    "url_length", "domain_length", "dot_count", "has_ip",
    "is_https", "suspicious_tld", "brand_in_domain", "hyphen_count",
    "digit_ratio", "directory_depth", "has_shortener", "subdomain_count",
    "special_char_count",
    # --- v2 features ---
    "has_at_symbol", "has_double_slash_in_path", "is_punycode",
    "domain_entropy", "query_param_count", "path_has_auth_keywords",
    "consecutive_hyphens", "tld_in_subdomain", "homoglyph_score"
]


def _shannon_entropy(s: str) -> float:
    """Computes Shannon entropy of string."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((v / len(s)) * math.log2(v / len(s)) for v in freq.values())


def extract_url_features(url: str) -> dict:
    """Extracts 22 lexical and structural features from a URL string."""
    if not url.startswith(("http://", "https://")):
        url_parsed = urlparse("http://" + url)
    else:
        url_parsed = urlparse(url)

    domain = url_parsed.netloc.lower().split(":")[0]
    path = url_parsed.path

    # Multi-part ccTLD resolution (.co.in, .gov.in, etc.)
    parts = domain.split(".")
    tld_parts_count = 1
    if len(parts) >= 2:
        two_part = f"{parts[-2]}.{parts[-1]}"
        if two_part in MULTI_PART_TLDS:
            tld = two_part
            tld_parts_count = 2
        else:
            tld = parts[-1]
    else:
        tld = parts[0] if parts else ""

    subdomains = max(0, len(parts) - 1 - tld_parts_count)
    domain_name = parts[-1 - tld_parts_count] if len(parts) > tld_parts_count else domain

    # Check for IP address
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    has_ip = 1 if re.match(ip_pattern, domain) else 0

    # Scheme
    is_https = 1 if url_parsed.scheme == "https" else 0

    # Suspicious TLD check
    suspicious_tld = 1 if tld in SUSPICIOUS_TLDS or (parts and parts[-1] in SUSPICIOUS_TLDS) else 0

    # Brand impersonation check:
    # A brand is impersonated IF the brand name appears in domain but the domain
    # does NOT officially belong to that brand.
    brand_found = 0
    for brand in POPULAR_BRANDS:
        if brand in domain:
            # Legitimate brand domain check
            is_legit = (
                domain == f"{brand}.com" or
                domain == f"{brand}.in" or
                domain == f"{brand}.co.in" or
                domain == f"{brand}.gov.in" or
                domain == f"{brand}.sbi" or
                domain.endswith(f".{brand}.com") or
                domain.endswith(f".{brand}.in") or
                domain.endswith(f".{brand}.co.in") or
                domain.endswith(f".{brand}.gov.in") or
                domain.endswith(f".{brand}.sbi") or
                domain.endswith(f".{brand}.net.in")
            )
            if not is_legit:
                brand_found = 1
                break

    url_len = len(url)
    domain_len = len(domain)
    dot_count = url.count(".")
    hyphen_count = url.count("-")
    digits = sum(c.isdigit() for c in url)
    digit_ratio = digits / max(1, url_len)
    depth = len([p for p in path.split("/") if p])
    has_shortener = 1 if domain in SHORTENER_DOMAINS else 0
    specials = sum(url.count(c) for c in ["@", "%", "=", "?", "&", "_", "!"])

    # --- v2 features ---
    has_at = 1 if "@" in url else 0
    after_scheme = url.split("//", 1)[-1] if "//" in url else url
    has_double_slash = 1 if "//" in after_scheme else 0
    is_punycode = 1 if "xn--" in domain else 0

    # Entropy of registered domain name (e.g. "sbi" or "x89a-phish-23")
    domain_ent = round(_shannon_entropy(domain_name), 4)

    query_params = url_parsed.query.count("&") + (1 if url_parsed.query else 0)

    path_lower = path.lower()
    path_has_auth = 1 if any(kw in path_lower for kw in AUTH_KEYWORDS) else 0

    consecutive_hyphens = 1 if "--" in domain else 0

    # TLD keyword (.com, .gov, .org, .in) embedded inside subdomain
    subdomain_str = ".".join(parts[:-1 - tld_parts_count]) if len(parts) > (1 + tld_parts_count) else ""
    tld_in_sub = 1 if bool(re.search(r"(^|\.)(com|net|org|gov|edu|mil|in|co)(\.|$|-)", subdomain_str)) else 0

    # Homoglyph substitution score in registered domain name
    hg_count = sum(1 for c in domain_name if c in HOMOGLYPH_MAP)
    homoglyph_score = round(hg_count / max(1, len(domain_name)), 4)

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
        "special_char_count": specials,
        # v2 features
        "has_at_symbol": has_at,
        "has_double_slash_in_path": has_double_slash,
        "is_punycode": is_punycode,
        "domain_entropy": domain_ent,
        "query_param_count": query_params,
        "path_has_auth_keywords": path_has_auth,
        "consecutive_hyphens": consecutive_hyphens,
        "tld_in_subdomain": tld_in_sub,
        "homoglyph_score": homoglyph_score,
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
            signals.append("Well-known brand name inside unofficial third-party domain")
        if features["has_ip"] == 1:
            signals.append("Uses bare IP address instead of registered domain name")
        if features["has_shortener"] == 1:
            signals.append("Uses URL shortener obscuring final destination")
        if features["digit_ratio"] > 0.15:
            signals.append(f"Unusually high digit ratio ({int(features['digit_ratio']*100)}%) in URL")
        if features["hyphen_count"] >= 2:
            signals.append("Multiple hyphens in domain, typical of domain spoofing")
        if features["has_at_symbol"] == 1:
            signals.append("Contains @ symbol in URL — classic redirect/credential harvesting trick")
        if features["is_punycode"] == 1:
            signals.append("Punycode/internationalized domain — may disguise lookalike characters")
        if features["domain_entropy"] > 3.5:
            signals.append(f"High domain entropy ({features['domain_entropy']:.2f}) — randomly-generated phishing domain")
        if features["path_has_auth_keywords"] == 1:
            signals.append("Path contains authentication/verification keywords (login, verify, kyc...)")
        if features["consecutive_hyphens"] == 1:
            signals.append("Double hyphens in domain — Punycode-style obfuscation or spoofing")
        if features["tld_in_subdomain"] == 1:
            signals.append("TLD keyword (.com/.gov/.in) embedded in subdomain — domain confusion technique")
        if features["homoglyph_score"] > 0.1:
            signals.append(f"Potential character substitution in domain (homoglyph score: {features['homoglyph_score']:.2f})")

        if self.model is not None:
            feat_vector = np.array([[features[k] for k in FEATURE_NAMES]])
            proba = self.model.predict_proba(feat_vector)[0]
            phish_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            score = 0
            if features["is_https"] == 0: score += 0.2
            if features["suspicious_tld"]: score += 0.35
            if features["brand_in_domain"]: score += 0.4
            if features["has_shortener"]: score += 0.25
            if features["has_ip"]: score += 0.4
            if features["has_at_symbol"]: score += 0.3
            if features["is_punycode"]: score += 0.4
            if features["domain_entropy"] > 3.5: score += 0.2
            if features["path_has_auth_keywords"]: score += 0.15
            phish_conf = min(0.99, max(0.01, score))

        # Clean Domain Profile Guard:
        # A URL on a clean domain with ZERO structural anomalies, standard TLD,
        # no brand impersonation, and normal entropy is NOT phishing just because
        # it contains a normal authentication path (e.g. /login, /signin, /netbanking).
        is_clean_profile = (
            features["suspicious_tld"] == 0 and
            features["brand_in_domain"] == 0 and
            features["has_ip"] == 0 and
            features["has_shortener"] == 0 and
            features["has_at_symbol"] == 0 and
            features["has_double_slash_in_path"] == 0 and
            features["is_punycode"] == 0 and
            features["consecutive_hyphens"] == 0 and
            features["tld_in_subdomain"] == 0 and
            features["digit_ratio"] < 0.10 and
            features["domain_entropy"] < 3.20
        )
        if is_clean_profile:
            phish_conf = min(phish_conf, 0.20)
            signals = [s for s in signals if not s.startswith("Path contains authentication")]

        if phish_conf >= 0.70:
            verdict = "PHISHING"
        elif phish_conf >= 0.45:
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
