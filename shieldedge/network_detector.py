"""
Network flow anomaly detector for ShieldEdge.

  * Input: bidirectional flow records as produced by CICFlowMeter (the tool
    behind CIC-IDS-2017) - either a CSV (`predict_csv`) or a single dict
    (`predict_flow`). No packet capture happens in Phase 1.
  * Model: k-nearest-neighbour novelty detection on log1p-scaled,
    standardised flow statistics. The anomaly score is the mean distance to the
    k closest flows in a reference set of *benign* traffic (CIC-IDS-2017 Monday).
    Thresholds are percentiles of that score on held-out benign flows, so no
    attack labels are used for training or tuning. kNN replaced an Isolation
    Forest, which missed DDoS and botnet C2 flows: their individual statistics
    are ordinary, but no normal flow has the same *combination* of them.
  * Rules add plain-language signals (scan-like resets, remote-access ports,
    very high packet rates, large uploads). A known remote-access port can
    raise a flow to SUSPICIOUS but never to ANOMALY on its own.
  * Missing features (e.g. the 3-number `--flow` shortcut) are filled with
    benign medians, not zeros, and reported as imputed.
"""

import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

FLOW_FEATURE_NAMES = [
    "duration", "total_fwd_packets", "total_bwd_packets", "total_fwd_bytes", "total_bwd_bytes",
    "packet_rate", "byte_rate", "dst_port", "is_tcp", "syn_flag_count", "rst_flag_count",
    "fin_flag_count", "ack_flag_count", "psh_flag_count", "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "flow_iat_mean", "fwd_init_win",
]

# CICFlowMeter / CIC-IDS-2017 column names (normalised: lower-case, alphanumerics only) -> feature
CIC_COLUMNS = {
    "flowduration": "duration",
    "totalfwdpacket": "total_fwd_packets", "totalfwdpackets": "total_fwd_packets",
    "totalbwdpackets": "total_bwd_packets", "totalbackwardpackets": "total_bwd_packets",
    "totallengthoffwdpacket": "total_fwd_bytes", "totallengthoffwdpackets": "total_fwd_bytes",
    "totallengthofbwdpacket": "total_bwd_bytes", "totallengthofbwdpackets": "total_bwd_bytes",
    "flowpacketss": "packet_rate", "flowbytess": "byte_rate",
    "dstport": "dst_port", "destinationport": "dst_port",
    "protocol": "protocol",
    "synflagcount": "syn_flag_count", "rstflagcount": "rst_flag_count", "finflagcount": "fin_flag_count",
    "ackflagcount": "ack_flag_count", "pshflagcount": "psh_flag_count",
    "fwdpacketlengthmean": "fwd_pkt_len_mean", "bwdpacketlengthmean": "bwd_pkt_len_mean",
    "flowiatmean": "flow_iat_mean",
    "fwdinitwinbytes": "fwd_init_win", "initwinbytesforward": "fwd_init_win",
    "label": "label",
}
MICROSECOND_COLUMNS = ("duration", "flow_iat_mean")  # CICFlowMeter reports these in microseconds

REMOTE_ACCESS_PORTS = {4444: "Metasploit/remote shell", 1337: "backdoor", 31337: "Back Orifice",
                       6667: "IRC botnet control", 5555: "Android debug bridge", 23: "Telnet", 2323: "Telnet (IoT botnets)"}


def log1p_clip(X):
    """Log-scale heavy-tailed flow statistics (module-level so the model pickles)."""
    return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))


class KNNNovelty(BaseEstimator):
    """Novelty score = minus the mean distance to the k nearest benign reference flows.

    Follows the sklearn convention of score_samples (higher = more normal), so it can be
    the last step of a Pipeline. Fitted on benign data only.
    """

    def __init__(self, n_neighbors: int = 20, n_reference: int = 20_000, random_state: int = 0):
        self.n_neighbors = n_neighbors
        self.n_reference = n_reference
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if len(X) > self.n_reference:
            X = X[np.random.RandomState(self.random_state).choice(len(X), self.n_reference, replace=False)]
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        return self

    def score_samples(self, X):
        return -self.nn_.kneighbors(np.asarray(X, dtype=float))[0].mean(axis=1)


def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def cic_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map a CICFlowMeter-style dataframe to ShieldEdge features (+ 'label' if present)."""
    mapped = {}
    for col in df.columns:
        key = CIC_COLUMNS.get(_norm_col(col))
        if key and key not in mapped:
            mapped[key] = df[col]
    out = pd.DataFrame(mapped)
    for c in MICROSECOND_COLUMNS:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce") / 1e6
    if "protocol" in out:
        out["is_tcp"] = (pd.to_numeric(out["protocol"], errors="coerce") == 6).astype(int)
        out = out.drop(columns="protocol")
    for c in out.columns:
        if c != "label":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def flow_summary(f: dict) -> str:
    proto = "TCP" if f.get("is_tcp", 0) else "UDP/other"
    parts = [f"{proto} to port {int(f.get('dst_port', 0))}"]
    if f.get("duration") is not None and not pd.isna(f.get("duration")):
        parts.append(f"{f['duration']:.3f}s")
    pk = (f.get("total_fwd_packets") or 0) + (f.get("total_bwd_packets") or 0)
    if pk:
        parts.append(f"{int(pk)} pkts")
    return ", ".join(parts)


def rule_signals(f: dict) -> list:
    s = []
    port = int(f.get("dst_port", 0) or 0)
    if port in REMOTE_ACCESS_PORTS:
        s.append(f"Connects to port {port}, often used by {REMOTE_ACCESS_PORTS[port]} tools")
    pk = (f.get("total_fwd_packets") or 0) + (f.get("total_bwd_packets") or 0)
    if f.get("syn_flag_count", 0) and f.get("rst_flag_count", 0) and (f.get("duration") or 1) < 0.01 and pk <= 4:
        s.append("Very short connection that was reset immediately, typical of port scanning")
    if (f.get("packet_rate") or 0) > 20_000 and pk >= 10:
        s.append(f"Extremely high packet rate ({f['packet_rate']:,.0f} packets/s)")
    if (f.get("total_fwd_bytes") or 0) > 20_000_000:
        s.append(f"Large upload from this device ({f['total_fwd_bytes'] / 1e6:.0f} MB)")
    return s


class NetworkDetector:
    def __init__(self, model_path: str = None):
        self.bundle = None
        if model_path and os.path.exists(model_path):
            obj = joblib.load(model_path)
            if isinstance(obj, dict) and "pipeline" in obj:
                self.bundle = obj
        self.medians = (self.bundle or {}).get("medians", {})

    # ------------------------------------------------------------------ scoring
    def _matrix(self, feats: pd.DataFrame):
        X = feats.reindex(columns=FLOW_FEATURE_NAMES)
        imputed = X.isna()
        X = X.fillna(pd.Series(self.medians)).fillna(0.0)
        return X.values.astype(float), imputed

    def scores(self, feats: pd.DataFrame) -> np.ndarray:
        X, _ = self._matrix(feats)
        if self.bundle is None:
            return np.zeros(len(X))
        return -self.bundle["pipeline"].score_samples(X)

    def _verdict(self, score: float, f: dict, imputed_cols=()):
        signals = rule_signals(f)
        if self.bundle is None:
            verdict = "SUSPICIOUS" if signals else "SAFE"
            conf = 0.6
        else:
            t_a, t_s = self.bundle["threshold_anomaly"], self.bundle["threshold_suspicious"]
            if score >= t_a:
                verdict = "ANOMALY"
            elif score >= t_s:
                verdict = "SUSPICIOUS"
            else:
                verdict = "SAFE"
            if verdict == "SAFE" and any("port" in s and "often used" in s for s in signals):
                verdict = "SUSPICIOUS"
            # map score to a confidence-like number relative to the benign score distribution
            lo, hi = self.bundle["score_p50"], self.bundle["score_p999"]
            rel = float(np.clip((score - lo) / max(1e-9, hi - lo), 0, 1))
            conf = 0.5 + 0.5 * rel if verdict != "SAFE" else 1.0 - 0.5 * rel
            if verdict != "SAFE":
                signals.append(f"Traffic pattern is more unusual than {self._pct(score):.1f}% of normal flows")
        if imputed_cols:
            signals.append(f"Only partial flow data was available ({len(FLOW_FEATURE_NAMES) - len(imputed_cols)}/{len(FLOW_FEATURE_NAMES)} features)")
        return verdict, round(conf, 3), signals

    def _pct(self, score: float) -> float:
        ref = self.bundle["benign_score_quantiles"]
        return float(np.searchsorted(ref, score) / len(ref) * 100)

    # ------------------------------------------------------------------ public API
    def predict_flow(self, flow_dict: dict) -> dict:
        feats = pd.DataFrame([{k: flow_dict.get(k) for k in FLOW_FEATURE_NAMES}])
        X, imputed = self._matrix(feats)
        f = dict(zip(FLOW_FEATURE_NAMES, X[0]))
        score = float(self.scores(feats)[0])
        imputed_cols = [c for c in FLOW_FEATURE_NAMES if imputed.iloc[0][c]]
        verdict, conf, signals = self._verdict(score, f, imputed_cols)
        summary = flow_summary(f)
        return {"input_type": "NETWORK_FLOW", "input": summary, "flow_summary": summary, "verdict": verdict,
                "confidence": conf, "score": round(score, 4), "signals": signals}

    def predict_frame(self, feats: pd.DataFrame, source: str = "flows") -> list:
        """Verdicts for a dataframe of ShieldEdge flow features (as returned by cic_to_features)."""
        X, imputed = self._matrix(feats)
        imputed = imputed.values
        sc = self.scores(feats)
        labels = feats["label"].astype(str).values if "label" in feats else None
        results = []
        for i in range(len(X)):
            f = dict(zip(FLOW_FEATURE_NAMES, X[i]))
            imputed_cols = [c for c, miss in zip(FLOW_FEATURE_NAMES, imputed[i]) if miss]
            verdict, conf, signals = self._verdict(float(sc[i]), f, imputed_cols)
            summary = flow_summary(f)
            r = {"row": i + 1, "input_type": "NETWORK_FLOW", "input": f"{source}#{i + 1}: {summary}",
                 "flow_summary": summary, "verdict": verdict, "confidence": conf, "score": round(float(sc[i]), 4),
                 "signals": signals}
            if labels is not None:
                r["label"] = labels[i]
            results.append(r)
        return results

    def predict_csv(self, path: str) -> list:
        feats = cic_to_features(pd.read_csv(path, low_memory=False))
        missing = [c for c in FLOW_FEATURE_NAMES if c not in feats]
        if len(missing) > len(FLOW_FEATURE_NAMES) // 2:
            raise ValueError(f"{path}: not a CICFlowMeter-style flow CSV (missing {', '.join(missing)})")
        return self.predict_frame(feats, os.path.basename(path))
