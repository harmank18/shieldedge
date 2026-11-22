"""
Decision Engine for ShieldEdge.

Takes a detector result and produces the *sanitised* evidence record that the
rest of the system uses:

  * decides whether the Llama advisory is needed (only for non-SAFE verdicts),
  * builds a short target descriptor that never contains raw SMS text
    (hostname for URLs, destination port/protocol for flows, nothing for SMS),
  * sanitises every signal string before it can reach the prompt,
  * attaches the uncertainty caveat required for SUSPICIOUS verdicts.

The verdict itself is always the detector's verdict - Llama can never change it.
"""

from shieldedge.sanitize import safe_host, safe_signal

THREAT_VERDICTS = {"PHISHING", "SCAM", "ANOMALY"}
ADVISORY_VERDICTS = THREAT_VERDICTS | {"SUSPICIOUS"}

CAVEAT_SUSPICIOUS = (
    "ShieldEdge is not certain about this one. Treat it with caution and verify "
    "through an official channel before acting."
)
CAVEAT_GENERAL = "ShieldEdge is an advisory tool and may not detect every threat."


class DecisionEngine:
    def evaluate(self, detection_result: dict) -> dict:
        verdict = detection_result.get("verdict", "SAFE")
        confidence = float(detection_result.get("confidence", 0.0))
        input_type = detection_result.get("input_type", "UNKNOWN")
        raw_input = str(detection_result.get("input", ""))
        signals = [safe_signal(s) for s in detection_result.get("signals", [])]
        signals = [s for s in signals if s]

        if input_type == "URL":
            target = safe_host(raw_input)
        elif input_type == "NETWORK_FLOW":
            target = safe_signal(detection_result.get("flow_summary", ""), 60)
        else:
            target = ""  # raw SMS text is never forwarded

        # Short, display-only summary for the terminal (never sent to Llama).
        if input_type == "SMS":
            summary = f"SMS ({len(raw_input)} chars)"
        elif input_type == "URL":
            summary = target or "(unparseable URL)"
        else:
            summary = target

        caveat = CAVEAT_SUSPICIOUS if verdict == "SUSPICIOUS" else CAVEAT_GENERAL

        return {
            "verdict": verdict,
            "confidence": confidence,
            "input_type": input_type,
            "target": target,
            "input_summary": summary,
            "signals": signals,
            "requires_advisory": verdict in ADVISORY_VERDICTS,
            "caveat": caveat,
        }
