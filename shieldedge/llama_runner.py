"""
On-Device Llama Advisory Runner and Fallback Explainer for ShieldEdge.
Interfaces with ExecuTorch .pte runtime on Arm or falls back to
deterministic rule-based templates if the LLM binary is not loaded.
"""

import os
import subprocess


class LlamaAdvisoryRunner:
    def __init__(self, pte_model_path: str = None, executorch_runner_bin: str = None):
        self.pte_model_path = pte_model_path
        self.executorch_runner_bin = executorch_runner_bin
        self.has_executorch = (
            pte_model_path is not None 
            and os.path.exists(pte_model_path)
            and executorch_runner_bin is not None 
            and os.path.exists(executorch_runner_bin)
        )

    def generate_advisory(self, evidence: dict, prompt: str) -> dict:
        """
        Generates explanation and remediation advice.
        Uses ExecuTorch if configured; otherwise generates high-reliability template advisory.
        """
        if self.has_executorch:
            try:
                cmd = [
                    self.executorch_runner_bin,
                    "--model_path", self.pte_model_path,
                    "--prompt", prompt,
                    "--max_new_tokens", "150"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if res.returncode == 0 and res.stdout.strip():
                    return {
                        "source": "ExecuTorch (Llama 3.2 1B-Instruct)",
                        "advisory": res.stdout.strip()
                    }
            except Exception as e:
                print(f"[Warning] ExecuTorch runner failed: {e}. Falling back to template explainer.")

        # Template-based deterministic advisory
        return {
            "source": "ShieldEdge Deterministic Advisory Engine",
            "advisory": self._generate_template_advisory(evidence)
        }

    def _generate_template_advisory(self, evidence: dict) -> str:
        v = evidence.get("verdict")
        itype = evidence.get("input_type")
        signals = evidence.get("signals", [])
        signals_text = "; ".join(signals) if signals else "anomalous attributes"

        if itype == "URL":
            explanation = (
                f"This web link has been classified as {v} (confidence: {evidence.get('confidence')}). "
                f"It was flagged due to: {signals_text}. "
                "Fraudulent links are commonly used by attackers to harvest banking credentials, OTPs, or install malicious software."
            )
            remediation = (
                "1. Do not click, open, or forward this link.\n"
                "2. If you already accessed it, change your passwords immediately and enable 2-Factor Authentication.\n"
                "3. Contact your bank's official customer support number (listed on your card or official passbook) to verify account status.\n"
                "4. Report this incident to CERT-In (incidents@cert-in.org.in) or call the CyberCrime Helpline (1930)."
            )
        elif itype == "SMS":
            explanation = (
                f"This SMS has been identified as a {v} message (confidence: {evidence.get('confidence')}). "
                f"Indicators observed: {signals_text}. "
                "Scammers commonly use false urgency (threats of disconnection or prizes) to coerce victims into making hasty decisions."
            )
            remediation = (
                "1. Never share OTPs, UPI PINs, passwords, or personal identity documents.\n"
                "2. Do not call numbers or click links provided within the message.\n"
                "3. Verify any account or service status through official banking apps or verified portals.\n"
                "4. Forward this scam message to the Department of Telecommunications (DoT) Chakshu / 1909 spam portal."
            )
        else: # NETWORK_FLOW
            explanation = (
                f"A local network flow anomaly was detected (confidence: {evidence.get('confidence')}). "
                f"Observed behavior: {signals_text}. "
                "This indicates potential malware activity, unauthorized remote access, or a device participating in malicious scanning."
            )
            remediation = (
                "1. Disconnect the affected client device from your Wi-Fi or Ethernet network immediately.\n"
                "2. Perform a full offline malware and anti-virus scan on the device.\n"
                "3. Inspect router management settings and change your Wi-Fi WPA2/WPA3 password.\n"
                "4. Check for unauthorized port forwards or UPnP rules in router configuration."
            )

        return f"[EXPLANATION]\n{explanation}\n\n[REMEDIATION]\n{remediation}"
