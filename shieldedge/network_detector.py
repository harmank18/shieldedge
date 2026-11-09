"""
Network Flow Anomaly Detection Module for ShieldEdge.
Uses unsupervised Isolation Forest and flow heuristics to detect
suspicious outbound connections, C2 activity, and port scans.
"""

import os
import joblib
import numpy as np

FLOW_FEATURE_NAMES = [
    "duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "packet_rate",
    "byte_rate",
    "dst_port",
    "is_tcp",
    "syn_flag_count",
    "rst_flag_count"
]

SUSPICIOUS_C2_PORTS = {4444, 1337, 6667, 8888, 9999, 31337}


class NetworkDetector:
    def __init__(self, model_path: str = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def predict_flow(self, flow_dict: dict) -> dict:
        """
        Analyzes a single network flow dictionary.
        Returns verdict (NORMAL / ANOMALY) and confidence/anomaly score.
        """
        signals = []
        dst_port = int(flow_dict.get("dst_port", 80))
        byte_rate = float(flow_dict.get("byte_rate", 0))
        packet_rate = float(flow_dict.get("packet_rate", 0))

        if dst_port in SUSPICIOUS_C2_PORTS:
            signals.append(f"Outbound connection to notorious Trojan/C2 port ({dst_port})")

        if packet_rate > 500:
            signals.append(f"Abnormally high packet rate ({packet_rate:.1f} pkts/s), potential DoS or scan")

        if byte_rate > 1_000_000:
            signals.append(f"Unusual high outbound data burst ({byte_rate/1024/1024:.2f} MB/s), potential data exfiltration")

        vector = np.array([[float(flow_dict.get(k, 0)) for k in FLOW_FEATURE_NAMES]])

        if self.model is not None:
            # IsolationForest: -1 is anomaly, 1 is normal
            pred = self.model.predict(vector)[0]
            score = -self.model.score_samples(vector)[0]  # higher score = more anomalous
            # Normalize score to approximate 0.0 - 1.0 confidence
            anomaly_conf = float(1.0 / (1.0 + np.exp(-3 * (score - 0.5))))
            is_anomaly = (pred == -1) or (anomaly_conf > 0.6)
        else:
            is_anomaly = len(signals) > 0
            anomaly_conf = 0.85 if is_anomaly else 0.15

        if len(signals) > 0:
            is_anomaly = True
            anomaly_conf = max(anomaly_conf, 0.88)

        verdict = "ANOMALY" if is_anomaly else "SAFE"

        return {
            "input_type": "NETWORK_FLOW",
            "input": f"{flow_dict.get('src_ip', '192.168.1.5')}:{flow_dict.get('src_port', 50000)} -> {flow_dict.get('dst_ip', '10.0.0.1')}:{dst_port}",
            "verdict": verdict,
            "confidence": round(anomaly_conf if verdict == "ANOMALY" else 1.0 - anomaly_conf, 3),
            "signals": signals
        }
