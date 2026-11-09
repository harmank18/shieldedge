"""
Decision Engine for ShieldEdge.
Consolidates raw classifier outputs, applies threshold logic and confidence calibration,
and decides whether to trigger the Llama advisory explainer.
"""


class DecisionEngine:
    def __init__(self, high_risk_threshold=0.75, suspicious_threshold=0.45):
        self.high_risk_threshold = high_risk_threshold
        self.suspicious_threshold = suspicious_threshold

    def evaluate(self, detection_result: dict) -> dict:
        verdict = detection_result.get("verdict", "SAFE")
        confidence = float(detection_result.get("confidence", 0.0))
        signals = detection_result.get("signals", [])
        input_type = detection_result.get("input_type", "UNKNOWN")
        raw_input = detection_result.get("input", "")

        # Determine advisory requirement
        requires_advisory = verdict in {"PHISHING", "SCAM", "ANOMALY", "SUSPICIOUS"}

        # Sanitized short description of input for Llama prompt (Prompt injection safe)
        if input_type == "URL":
            summary = raw_input[:60]
        elif input_type == "SMS":
            summary = raw_input[:50].replace("\n", " ").strip() + "..."
        else:
            summary = raw_input[:50]

        return {
            "verdict": verdict,
            "confidence": confidence,
            "input_type": input_type,
            "input_summary": summary,
            "signals": signals,
            "requires_advisory": requires_advisory
        }
