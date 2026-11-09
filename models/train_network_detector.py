"""
Training script for Network Flow Anomaly Detector.
Trains an Isolation Forest model on baseline non-malicious traffic flows
and exports the model to models/network_isolation_forest.joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from shieldedge.network_detector import FLOW_FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_OUT = os.path.join(BASE_DIR, "models", "network_isolation_forest.joblib")


def generate_baseline_flows(n_samples=5000):
    """Generates synthetic baseline traffic flows modeled on standard CICIDS normal profiles."""
    np.random.seed(42)
    duration = np.random.exponential(scale=2.5, size=n_samples)
    fwd_pkts = np.random.poisson(lam=10, size=n_samples) + 1
    bwd_pkts = np.random.poisson(lam=12, size=n_samples) + 1
    fwd_bytes = fwd_pkts * np.random.randint(60, 1400, size=n_samples)
    bwd_bytes = bwd_pkts * np.random.randint(60, 1400, size=n_samples)
    pkt_rate = (fwd_pkts + bwd_pkts) / np.maximum(duration, 0.01)
    byte_rate = (fwd_bytes + bwd_bytes) / np.maximum(duration, 0.01)
    # Common ports: 80, 443, 53, 8080
    common_ports = [80, 443, 53, 8080, 22]
    dst_port = np.random.choice(common_ports, size=n_samples, p=[0.2, 0.6, 0.15, 0.03, 0.02])
    is_tcp = np.where(dst_port == 53, 0, 1)
    syn_flags = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.25, 0.05])
    rst_flags = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])

    df = pd.DataFrame({
        "duration": duration,
        "total_fwd_packets": fwd_pkts,
        "total_bwd_packets": bwd_pkts,
        "total_fwd_bytes": fwd_bytes,
        "total_bwd_bytes": bwd_bytes,
        "packet_rate": pkt_rate,
        "byte_rate": byte_rate,
        "dst_port": dst_port,
        "is_tcp": is_tcp,
        "syn_flag_count": syn_flags,
        "rst_flag_count": rst_flags
    })
    return df


def main():
    print("Generating baseline traffic flow training data...")
    df_normal = generate_baseline_flows(n_samples=5000)
    X_train = df_normal[FLOW_FEATURE_NAMES].values

    print(f"Training Isolation Forest on {len(X_train)} baseline network flows...")
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(X_train)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"Network Anomaly Model exported successfully to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
