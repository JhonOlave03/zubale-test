import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_endpoint():
    sample = [{
        "plan_type": "basic",
        "contract_type": "monthly",
        "autopay": "yes",
        "is_promo_user": "no",
        "add_on_count": 1,
        "tenure_months": 12,
        "monthly_usage_gb": 50,
        "avg_latency_ms": 30,
        "support_tickets_30d": 1,
        "discount_pct": 10,
        "payment_failures_90d": 0,
        "downtime_hours_30d": 0.5
    }]

    resp = client.post("/predict", json=sample)
    assert resp.status_code == 200
    result = resp.json()["results"][0]

    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in [0, 1]
