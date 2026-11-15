"""
Prompt Constructor for ShieldEdge.
Formats structured detector outputs into a bounded, prompt-injection-safe template
for on-device Llama inference.
"""


def construct_prompt(evidence: dict) -> str:
    """
    Constructs a rigid, guardrailed, latency-optimized prompt for on-device
    Llama inference formatted with Llama 3 special instruct tokens.
    """
    signals = evidence.get("signals", [])
    signals_list = "\n".join([f"- {s}" for s in signals[:5]]) if signals else "- Anomalous feature indicators detected."

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are ShieldEdge, an on-device cybersecurity assistant in India. Explain why the threat was flagged based strictly on the detected indicators, and provide practical safety steps for citizens (CERT-In 1930 / DoT 1909). Be concise.<|eot_id|><|start_header_id|>user<|end_header_id|>

Threat: {evidence.get('input_type')} | Verdict: {evidence.get('verdict')} (Confidence: {evidence.get('confidence')})
Target: {evidence.get('input_summary')}
Signals:
{signals_list}

Provide [EXPLANATION] followed by [REMEDIATION].<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt
