"""
SMS Spam and Smishing Threat Detection Module for ShieldEdge.
Combines TF-IDF NLP model with Indian cybersecurity regex rules.
Enhanced v2: Hindi/Hinglish keyword detection, Devanagari script support,
expanded Indian brand/agency list for comprehensive scam coverage.
"""

import os
import re
import joblib

URGENCY_KEYWORDS = [
    "suspended", "blocked", "block", "disconnected", "deactivated",
    "immediately", "24 hours", "2 hours", "urgent", "expire", "expired",
    "lottery", "winner", "congratulations", "won", "reward points", "refund",
    "last chance", "account hold", "verify now", "action required",
    "limited time", "click here", "call now", "debit card blocked",
]

# Transliterated Hindi/Hinglish phrases commonly used in Indian SMS scams
HINDI_URGENCY_KEYWORDS = [
    "band ho jayega", "band kar diya", "sim band", "account band",
    "turant", "abhi karein", "jaldi karo", "foran",
    "khata band", "paisa bhejo", "paise transfer",
    "inaam", "aapne jeet", "lucky draw mein",
    "kyc update karo", "kyc expire", "kyc verify karo",
    "bijli band", "bijli bill", "bijli kategi",
    "loan approved", "loan mil gaya",
    "otp share karo", "otp batao", "otp dena",
    "aapka account", "aapke khate",
    "verify karo", "update karo", "confirm karo",
    "free mein", "free recharge", "free data milega",
    "ration card band", "pm kisan", "ayushman",
]

FINANCIAL_AGENCY_KEYWORDS = [
    "sbi", "hdfc", "icici", "axis", "rbi", "kyc", "pan", "aadhaar",
    "uidai", "trai", "income tax", "incometax", "electricity", "phonepe", "paytm",
    # v2 additions
    "jio", "bsnl", "lic", "nsdl", "epfo", "digilocker", "npci", "bhim",
    "fastag", "tds refund", "gst", "pension", "ration card",
    "pm kisan", "ayushman", "zerodha", "groww", "cred", "razorpay",
]

UPI_PATTERN = re.compile(r"[\w\.\-_]+@(ok[a-zA-Z]+|upi|ybl|axl|paytm|icici|apl|ibl|pthdfc|ptaxis)")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+|rb\.gy/\S+|cutt\.ly/\S+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")  # Indian mobile numbers in SMS scams


def has_devanagari(text: str) -> bool:
    """Detect Devanagari Unicode block (Hindi, Marathi, Nepali script)."""
    return any('\u0900' <= c <= '\u097F' for c in text)


def preprocess_text(text: str) -> str:
    """Cleans text: lowercase, removes special symbols, preserves Devanagari & rupee."""
    text = text.lower()
    # Preserve Devanagari (\u0900-\u097F) and rupee symbol
    text = re.sub(r"[^\w\s\?\!₹\.,\-:/\u0900-\u097F]", " ", text)
    return text.strip()


class SMSDetector:
    def __init__(self, model_path: str = None):
        self.pipeline = None
        if model_path and os.path.exists(model_path):
            self.pipeline = joblib.load(model_path)

    def extract_signals(self, text: str) -> list:
        signals = []
        lower = text.lower()

        # Check English urgency keywords
        matched_urgency = [w for w in URGENCY_KEYWORDS if w in lower]
        if matched_urgency:
            signals.append(f"High-pressure urgency keywords: {', '.join(matched_urgency[:3])}")

        # Check Hindi/Hinglish urgency patterns
        matched_hindi = [w for w in HINDI_URGENCY_KEYWORDS if w in lower]
        if matched_hindi:
            signals.append(f"Hindi/Hinglish scam phrase detected: {', '.join(matched_hindi[:2])}")

        # Detect Devanagari script (native Hindi in Unicode)
        if has_devanagari(text):
            signals.append("Contains Devanagari script — check for Hindi scam content")

        # Check banking / agency impersonation
        matched_agencies = [w for w in FINANCIAL_AGENCY_KEYWORDS if w in lower]
        if matched_agencies:
            signals.append(f"Financial/Government authority mentions: {', '.join(matched_agencies[:3]).upper()}")

        # Check for UPI IDs
        if UPI_PATTERN.search(text):
            signals.append("Embedded UPI Virtual Payment Address (VPA) detected")

        # Check for embedded links
        links = URL_PATTERN.findall(text)
        if links:
            signals.append(f"Contains embedded web link: {links[0][:60]}")

        # Suspicious phone number embedded (not a link but a callback number)
        phones = PHONE_PATTERN.findall(text)
        if phones and not links:
            signals.append(f"Suspicious callback phone number: {phones[0]}")

        return signals

    def predict(self, text: str) -> dict:
        signals = self.extract_signals(text)
        cleaned = preprocess_text(text)

        if self.pipeline is not None:
            proba = self.pipeline.predict_proba([cleaned])[0]
            scam_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            rule_score = len(signals) * 0.25
            scam_conf = min(0.98, max(0.02, rule_score))

        # Rule boosting if multiple high-risk signals match
        if len(signals) >= 2 and scam_conf < 0.6:
            scam_conf = 0.78

        if scam_conf >= 0.75:
            verdict = "SCAM"
        elif scam_conf >= 0.45:
            verdict = "SUSPICIOUS"
        else:
            verdict = "SAFE"

        return {
            "input_type": "SMS",
            "input": text,
            "verdict": verdict,
            "confidence": round(scam_conf if verdict != "SAFE" else 1.0 - scam_conf, 3),
            "signals": signals
        }
