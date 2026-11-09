"""
SMS Spam and Smishing Threat Detection Module for ShieldEdge.
Combines TF-IDF NLP model with fast Indian cybersecurity regex rules.
"""

import os
import re
import joblib

URGENCY_KEYWORDS = [
    "suspended", "blocked", "block", "disconnected", "deactivated", 
    "immediately", "24 hours", "2 hours", "urgent", "expire", "expired",
    "lottery", "winner", "congratulations", "won", "reward points", "refund"
]

FINANCIAL_AGENCY_KEYWORDS = [
    "sbi", "hdfc", "icici", "axis", "rbi", "kyc", "pan", "aadhaar", 
    "uidai", "trai", "income tax", "incometax", "electricity", "phonepe", "paytm"
]

UPI_PATTERN = re.compile(r"[\w\.\-_]+@(ok[a-zA-Z]+|upi|ybl|axl|paytm|icici|apl)")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|bit\.ly/\S+|tinyurl\.com/\S+")


def preprocess_text(text: str) -> str:
    """Cleans text: lowercase, removes weird symbols, preserves rupees and punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s\?!\₹\.,\-:/]", " ", text)
    return text.strip()


class SMSDetector:
    def __init__(self, model_path: str = None):
        self.pipeline = None
        if model_path and os.path.exists(model_path):
            self.pipeline = joblib.load(model_path)

    def extract_signals(self, text: str) -> list:
        signals = []
        lower = text.lower()

        # Check urgency
        matched_urgency = [w for w in URGENCY_KEYWORDS if w in lower]
        if matched_urgency:
            signals.append(f"High-pressure urgency keywords detected: {', '.join(matched_urgency[:3])}")

        # Check banking / agency impersonation
        matched_agencies = [w for w in FINANCIAL_AGENCY_KEYWORDS if w in lower]
        if matched_agencies:
            signals.append(f"Financial / Government authority mentions: {', '.join(matched_agencies[:3]).upper()}")

        # Check for UPI IDs
        if UPI_PATTERN.search(text):
            signals.append("Embedded UPI Virtual Payment Address (VPA) detected")

        # Check for embedded links
        links = URL_PATTERN.findall(text)
        if links:
            signals.append(f"Contains embedded web link: {links[0]}")

        return signals

    def predict(self, text: str) -> dict:
        signals = self.extract_signals(text)
        cleaned = preprocess_text(text)

        if self.pipeline is not None:
            proba = self.pipeline.predict_proba([cleaned])[0]
            # Assuming index 1 is spam/scam
            scam_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # Heuristic rule fallback
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
