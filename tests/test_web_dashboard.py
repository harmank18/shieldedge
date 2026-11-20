import pytest

import shieldedge.web_dashboard as wd
from shieldedge.storage import AlertLogger


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "alerts.db"
    orig = wd.AlertLogger
    wd.AlertLogger = lambda: AlertLogger(str(db))  # never touch the real audit log
    try:
        dash = wd.Dashboard(replay=False)
    finally:
        wd.AlertLogger = orig
    return wd.create_app(dash).test_client()


def test_index_and_static(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_analyze_url_and_sms(client):
    r = client.post("/api/analyze", json={"text": "http://sbi-net-banking-update.xyz/login"}).get_json()
    assert r["kind"] == "URL" and r["verdict"] == "PHISHING" and r["alert_id"]
    assert r["steps"] and r["source"] == "you"
    r = client.post("/api/analyze", json={"text": "Rs.2,500.00 debited from A/c XX4521 to VPA zomato@icici. "
                                                   "Not you? Call 18002586161 -HDFC Bank"}).get_json()
    assert r["kind"] == "SMS" and r["verdict"] == "SAFE" and r["alert_id"] is None


def test_analyze_rejects_bad_input(client):
    assert client.post("/api/analyze", json={"text": ""}).status_code == 400
    assert client.post("/api/analyze", json={"text": "x" * 2000}).status_code == 400


def test_evidence_never_sent_to_browser(client):
    r = client.post("/api/analyze", json={"text": "http://paytm-kyc-pending.shop/verify"}).get_json()
    assert "evidence" not in r
    assert "evidence" not in client.get(f"/api/event/{r['id']}").get_json()


def test_explain_refuses_safe_items(client):
    r = client.post("/api/analyze", json={"text": "https://netbanking.hdfcbank.com/netbanking/"}).get_json()
    assert r["verdict"] == "SAFE"
    body = client.get(f"/api/explain/{r['id']}").get_data(as_text=True)
    assert "Only flagged items" in body


def test_stats_device_quality_alerts(client):
    s = client.get("/api/stats").get_json()
    assert s["counts"]["URL"] and len(s["timeline"]) == 120
    d = client.get("/api/device").get_json()
    assert d["external_connections"] == 0 and d["cpu"]
    assert "detection" in client.get("/api/quality").get_json()
    a = client.get("/api/alerts").get_json()
    assert a["alerts"] and client.post(f"/api/alerts/{a['alerts'][0]['id']}/safe").get_json()["ok"]
    assert client.post("/api/alerts/clear").get_json()["deleted"] >= 1
