"""
SMS scam / smishing detector for ShieldEdge.

  * ML: TF-IDF (word 1-3-grams) + calibrated LinearSVC trained on the UCI SMS
    Spam Collection plus a small team-authored set of Indian messages.
  * Rules (explainable, conservative):
      - links inside the SMS are checked with the URL detector's offline
        domain checks (brand misuse, risky TLD, shortener, free hosting);
      - recognised transactional formats (OTP, debit/credit alerts, delivery,
        recharge confirmations) are not flagged unless they contain a risky
        link or a request to pay/share/call;
      - English + Hindi/Hinglish pressure phrases, UPI handles and callback
        numbers become plain-language signals.
  * Phone numbers are masked and links reduced to their hostname before they
    become signals (privacy + prompt-injection safety).
"""

import os
import re

import joblib

from shieldedge.sanitize import mask_phone, safe_host
from shieldedge.url_detector import SHORTENER_DOMAINS, extract_url_features, impersonated_brand, official_brand

URGENCY_KEYWORDS = [
    "suspended", "blocked", "block", "disconnected", "deactivated", "immediately", "24 hours", "24 hrs", "2 hours",
    "urgent", "expire", "expired", "last chance", "account hold", "on hold", "verify now", "action required",
    "final notice", "legal action", "arrest", "penalty", "will be cut", "tonight", "within 1 hour", "by mistake",
]
REWARD_KEYWORDS = [
    "lottery", "lucky draw", "winner", "congratulations", "won ", "you won", "prize", "reward points", "cashback",
    "free recharge", "gift card", "earn rs", "earn ", "per day", "daily payout", "guaranteed returns", "double your money",
    "loan approved", "instant loan", "work from home", "part time",
]
HINDI_KEYWORDS = [
    "band ho jayega", "band kar diya", "sim band", "account band", "khata band", "turant", "abhi karein", "abhi verify",
    "jaldi karo", "foran", "paisa bhejo", "paise transfer", "bhejo", "inaam", "aapne jeet", "jeet liya", "jeete",
    "lucky draw mein", "kyc update karo", "verify karo", "update karo", "confirm karo", "bijli band", "kat jayega",
    "otp share karo", "otp batao", "otp dena", "free mein", "badhai ho", "fine lagega", "lauta dunga",
]
AUTHORITY_KEYWORDS = [
    "sbi", "yono", "hdfc", "icici", "axis", "kotak", "rbi", "kyc", "pan", "aadhaar", "aadhar", "uidai", "trai", "dot ",
    "income tax", "gst", "electricity", "bijli", "phonepe", "paytm", "google pay", "jio", "airtel", "bsnl", "lic",
    "epfo", "pf ", "npci", "fastag", "pm kisan", "ayushman", "customs", "police", "cbi", "fedex", "dhl", "india post",
]
REQUEST_PATTERNS = re.compile(
    r"\b(click|visit|call|whatsapp|send|pay|share|reply|press \d|download|register|claim|redeem|update|verify|join)\b", re.I)
TRANSACTIONAL_PATTERNS = [
    re.compile(r"\b(otp|one time password)\b.*\b(do not|don't|never) share\b", re.I),
    re.compile(r"\bis your (otp|verification code)\b", re.I),
    re.compile(r"\b(debited|credited|spent|withdrawn|paid)\b.*\b(a/c|acct|account|card)\s*(no\.?\s*)?x+\d{2,4}", re.I),
    re.compile(r"\b(a/c|card)\s*x+\d{2,4}\b.*\b(debited|credited|spent|withdrawn)\b", re.I),
    re.compile(r"\b(has been|is) (shipped|delivered|packed)|out for delivery\b", re.I),
    re.compile(r"\brecharge of rs\.?\s*[\d,]+ (is )?successful\b", re.I),
    re.compile(r"\bpnr\s*\d{10}\b", re.I),
    re.compile(r"\bbooking (no\.?|id)?\s*\d+.*confirmed\b", re.I),
]
PAYMENT_ASK = re.compile(r"\b(send|return|transfer|pay|bhejo|bhej de)\b.*(@\w+|rs\.?\s*\d|₹)", re.I)

UPI_PATTERN = re.compile(r"[\w.\-]+@(ok[a-z]+|upi|ybl|axl|paytm|ibl|apl|icici|sbi|hdfcbank|pthdfc|ptaxis|ptsbi|ptyes)\b", re.I)
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9\-]+(\.[a-z0-9\-]+)*\.(com|in|xyz|top|online|site|click|icu|vip|"
                         r"shop|store|live|info|sbs|cfd|cyou|win|link|me|ly|gy|co|app|io)(/\S*)?)", re.I)
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{9}(?!\d)")
SAFE_SENDER_NUMBERS = re.compile(r"\b(1800[\d\-]{6,8}|1930|1947|1909|5676\d{3})\b")


def has_devanagari(text: str) -> bool:
    return any("ऀ" <= c <= "ॿ" for c in text)


def preprocess_text(text: str) -> str:
    """Lower-case, keep Devanagari and the rupee sign, drop other symbols (used for training and inference)."""
    text = text.lower()
    text = re.sub(r"[^\w\s?!₹.,\-:/@ऀ-ॿ]", " ", text)
    return text.strip()


def analyse_links(text: str) -> list:
    """Return [(host, risk_reason or None)] for every link-like token in the SMS."""
    out = []
    for m in URL_PATTERN.finditer(text):
        token = m.group(0).rstrip(".,)")
        if "@" in token:  # part of a UPI id / e-mail
            continue
        host = safe_host(token)
        if not host or "." not in host:
            continue
        if official_brand(host):
            out.append((host, None))
            continue
        f = extract_url_features(token)
        brand = impersonated_brand(host)
        if brand:
            reason = f"pretends to be {brand.upper()} but is not an official {brand.upper()} website"
        elif host in SHORTENER_DOMAINS or host.startswith("t.me"):
            reason = "is a shortened or chat-group link that hides where it goes"
        elif f["suspicious_tld"]:
            reason = "uses a cheap domain ending common in scams"
        elif f["free_hosting"]:
            reason = "is hosted on a free website platform"
        elif f["has_ip"]:
            reason = "is a raw IP address"
        else:
            reason = "is not a known official website"
        out.append((host, reason))
    return out


class SMSDetector:
    def __init__(self, model_path: str = None):
        self.pipeline = None
        self.t_scam, self.t_susp = 0.80, 0.50
        if model_path and os.path.exists(model_path):
            obj = joblib.load(model_path)
            if isinstance(obj, dict):
                self.pipeline = obj["pipeline"]
                self.t_scam = obj.get("threshold_scam", self.t_scam)
                self.t_susp = obj.get("threshold_suspicious", self.t_susp)
            else:
                self.pipeline = obj

    def analyse(self, text: str) -> dict:
        lower = text.lower()
        links = analyse_links(text)
        phones = [p for p in PHONE_PATTERN.findall(text)]
        a = {
            "urgency": [w.strip() for w in URGENCY_KEYWORDS if w in lower],
            "reward": [w.strip() for w in REWARD_KEYWORDS if w in lower],
            "hindi": [w for w in HINDI_KEYWORDS if w in lower],
            "authority": sorted({w.strip() for w in AUTHORITY_KEYWORDS if w in lower}),
            "links": links,
            "risky_links": [(h, r) for h, r in links if r],
            "upi": bool(UPI_PATTERN.search(text)),
            "phones": phones,
            "request": bool(REQUEST_PATTERNS.search(text)),
            "payment_ask": bool(PAYMENT_ASK.search(text)),
            "transactional": any(p.search(text) for p in TRANSACTIONAL_PATTERNS),
            "devanagari": has_devanagari(text),
        }
        return a

    @staticmethod
    def signals_from(a: dict) -> list:
        s = []
        if a["urgency"]:
            s.append(f"Pressure or threat words: {', '.join(a['urgency'][:3])}")
        if a["reward"]:
            s.append(f"Promises money, prizes or easy income: {', '.join(a['reward'][:3])}")
        if a["hindi"]:
            s.append(f"Hindi/Hinglish pressure phrases: {', '.join(a['hindi'][:3])}")
        if a["authority"]:
            s.append(f"Mentions a bank, government or company: {', '.join(x.upper() for x in a['authority'][:3])}")
        for host, reason in a["risky_links"][:2]:
            s.append(f"Contains a link to {host}, which {reason}")
        if a["upi"]:
            s.append("Asks for money to a UPI ID")
        elif a["payment_ask"]:
            s.append("Asks you to send or pay money")
        if a["phones"] and not a["links"]:
            s.append(f"Asks you to contact a private mobile number ({mask_phone(a['phones'][0])})")
        if a["devanagari"]:
            s.append("Written in Devanagari script")
        return s

    def predict(self, text: str) -> dict:
        a = self.analyse(text)
        signals = self.signals_from(a)

        if self.pipeline is not None:
            p = float(self.pipeline.predict_proba([preprocess_text(text)])[0][1])
        else:
            p = min(0.95, 0.05 + 0.2 * len(signals))

        pressure = bool(a["urgency"] or a["reward"] or a["hindi"])
        contact_channel = bool(a["risky_links"] or a["upi"] or (a["phones"] and not a["links"]))

        # Rule layer (each rule is explainable to the user)
        if a["risky_links"] and (pressure or a["authority"] or a["request"]):
            p = max(p, self.t_scam)
        elif contact_channel and pressure and (a["authority"] or a["request"]):
            p = max(p, self.t_scam)
        elif a["payment_ask"] and pressure:
            p = max(p, self.t_susp)
        if a["transactional"] and not a["risky_links"] and not a["payment_ask"] and not (a["phones"] and pressure):
            p = min(p, self.t_susp - 0.01)  # genuine-looking bank/OTP/delivery message
            signals = [x for x in signals if not x.startswith("Mentions a bank")]

        if p >= self.t_scam:
            verdict = "SCAM"
        elif p >= self.t_susp:
            verdict = "SUSPICIOUS"
        else:
            verdict = "SAFE"
        if verdict == "SAFE":
            signals = [x for x in signals if x.startswith("Contains a link")] or []

        return {
            "input_type": "SMS",
            "input": text,
            "verdict": verdict,
            "score": round(p, 4),
            "confidence": round(p if verdict != "SAFE" else 1.0 - p, 3),
            "signals": signals,
        }
