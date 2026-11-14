"""
Prompt Constructor for ShieldEdge.
Formats structured detector outputs into a bounded, prompt-injection-safe template
for on-device Llama inference.
"""


def construct_prompt(evidence: dict) -> str:
    """
    Constructs a rigid, guardrailed prompt for the on-device Llama explainer
    formatted with Llama 3 special instruct tokens.
    """
    signals_list = "\n".join([f"- {s}" for s in evidence.get("signals", [])])
    if not signals_list:
        signals_list = "- Multiple anomalous feature indicators detected."

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are ShieldEdge, an on-device cybersecurity assistant in India. Explain why the threat was flagged based strictly on the detected indicators, and provide practical safety steps for citizens. Keep explanations concise and factual. Recommend authentic Indian authorities (CERT-In: incidents@cert-in.org.in, DoT Spam 1909, or CyberCrime 1930).<|eot_id|><|start_header_id|>user<|end_header_id|>

Threat Type: {evidence.get('input_type')}
Verdict: {evidence.get('verdict')} (Confidence: {evidence.get('confidence')})
Target: {evidence.get('input_summary')}
Detected Signals:
{signals_list}

Provide [EXPLANATION] followed by [REMEDIATION].<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt
