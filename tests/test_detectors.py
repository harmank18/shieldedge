"""Behavioural tests for the detectors, storage and REST API.  Run: pytest -q"""

import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shieldedge.network_detector import FLOW_FEATURE_NAMES, NetworkDetector, cic_to_features  # noqa: E402
from shieldedge.sms_detector import SMSDetector  # noqa: E402
from shieldedge.storage import AlertLogger, hash_input  # noqa: E402
from shieldedge.ui_cli import NET_MODEL_PATH, SMS_MODEL_PATH, URL_MODEL_PATH  # noqa: E402
from shieldedge.url_detector import URLDetector, normalise_url  # noqa: E402

DEMO_FLOWS = os.path.join(BASE_DIR, "data", "samples", "demo_flows.csv")


@pytest.fixture(scope="module")
def url_det():
    return URLDetector(URL_MODEL_PATH)


@pytest.fixture(scope="module")
def sms_det():
    return SMSDetector(SMS_MODEL_PATH)


# ------------------------------------------------------------------ URL
@pytest.mark.parametrize("bad", ["http://[bad", "", "   ", "javascript:alert(1)", "http://", "%%%%", "a" * 5000])
def test_url_never_crashes(url_det, bad):
    r = url_det.predict(bad)
    assert r["verdict"] in {"SAFE", "SUSPICIOUS", "PHISHING"}


def test_scheme_does_not_change_features():
    assert normalise_url("https://www.example.com/a")["norm"] == normalise_url("example.com/a")["norm"]


@pytest.mark.parametrize("url", [
    "https://netbanking.hdfcbank.com/netbanking/", "https://www.onlinesbi.sbi/", "https://www.incometax.gov.in/iec/foportal/",
    "https://www.google.com", "https://en.wikipedia.org/wiki/Phishing",
])
def test_official_and_popular_sites_are_safe(url_det, url):
    assert url_det.predict(url)["verdict"] == "SAFE"


@pytest.mark.parametrize("url", [
    "http://sbi-net-banking-update.xyz/login", "https://hdfc-kyc-update.weebly.com/", "http://paytm-kyc-pending.shop/verify",
])
def test_obvious_phishing_is_flagged(url_det, url):
    assert url_det.predict(url)["verdict"] == "PHISHING"


def test_shortener_is_suspicious_not_phishing(url_det):
    assert url_det.predict("https://bit.ly/3xYz12")["verdict"] == "SUSPICIOUS"


def test_user_content_on_brand_platform_not_called_impersonation(url_det):
    sig = " ".join(url_det.predict("https://docs.google.com/forms/d/e/abc/viewform")["signals"])
    assert "not an official GOOGLE" not in sig


# ------------------------------------------------------------------ SMS
@pytest.mark.parametrize("text", [
    "Your OTP for SBI NetBanking login is 482913. Do not share it with anyone. -SBI",
    "Rs.2,500.00 debited from A/c XX4521 on 17-Nov-26 to VPA zomato@icici. Not you? Call 18001234 -HDFC Bank",
    "Your Amazon order #405-123 has been shipped. Track at https://amzn.in/d/abc",
    "Meeting moved to 3pm tomorrow, room 204",
])
def test_transactional_and_personal_sms_safe(sms_det, text):
    assert sms_det.predict(text)["verdict"] == "SAFE"


@pytest.mark.parametrize("text", [
    "Dear customer your SBI account will be blocked in 24 hours. Update KYC at http://sbi-kyc-verify.top now",
    "Aapka SBI khata band ho jayega. Turant KYC update karo: http://sbi-kyc-help.xyz",
    "Rs 4,999 credited to your account by mistake. Please return it to 9182736450@ybl immediately.",
])
def test_scam_sms_flagged(sms_det, text):
    assert sms_det.predict(text)["verdict"] in {"SCAM", "SUSPICIOUS"}


def test_phone_numbers_masked_in_signals(sms_det):
    r = sms_det.predict("Your electricity will be disconnected tonight. Call officer 9823012938 immediately")
    assert "9823012938" not in " ".join(r["signals"])


# ------------------------------------------------------------------ Network
def test_cic_csv_mapping_and_batch():
    import pandas as pd

    feats = cic_to_features(pd.read_csv(DEMO_FLOWS))
    assert set(FLOW_FEATURE_NAMES) <= set(feats.columns)
    res = NetworkDetector(NET_MODEL_PATH).predict_csv(DEMO_FLOWS)
    assert len(res) == len(feats)
    scans = [r for r in res if r["label"] == "Portscan"]
    assert scans and all(r["verdict"] != "SAFE" for r in scans)


def test_partial_flow_is_reported():
    r = NetworkDetector(NET_MODEL_PATH).predict_flow({"dst_port": 4444, "byte_rate": 2e6, "packet_rate": 600})
    assert r["verdict"] in {"SUSPICIOUS", "ANOMALY"}
    assert any("partial flow data" in s for s in r["signals"])


# ------------------------------------------------------------------ Storage / privacy
def test_storage_hash_only_mark_and_clear(tmp_path):
    db = AlertLogger(str(tmp_path / "a.db"))
    secret = "OTP 123456 for your account"
    aid = db.log_alert("SMS", secret, "SCAM", 0.9)
    import sqlite3

    dump = "".join(str(r) for r in sqlite3.connect(db.db_path).execute("SELECT * FROM alerts"))
    assert secret not in dump and hash_input(secret) in dump
    assert db.mark_false_positive(aid)
    assert db.feedback_stats()["marked_safe_by_user"] == 1
    assert db.clear() == 1 and db.get_recent_alerts() == []


# ------------------------------------------------------------------ REST API
def test_api_health_and_bad_input(tmp_path, monkeypatch):
    flask = pytest.importorskip("flask")  # noqa: F841
    from shieldedge import api_server

    monkeypatch.setattr(api_server, "_logger", AlertLogger(str(tmp_path / "api.db")))
    c = api_server.app.test_client()
    assert c.get("/health").get_json()["status"] == "ok"
    assert c.post("/analyze/url", json={}).status_code == 400
    r = c.post("/analyze/sms", json={"message": "Your SBI KYC expired, update at http://sbi-kyc.top now"})
    assert r.status_code == 200 and r.get_json()["verdict"] in {"SCAM", "SUSPICIOUS"}
