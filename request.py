import requests

url = "http://127.0.0.1:8000/predict"

data = [
    {
        "plan_type": "Basic",
        "contract_type": "Monthly",
        "autopay": "Yes",
        "is_promo_user": "No",
        "add_on_count": 2,
        "tenure_months": 12,
        "monthly_usage_gb": 50,
        "avg_latency_ms": 100,
        "support_tickets_30d": 1,
        "discount_pct": 10,
        "payment_failures_90d": 0,
        "downtime_hours_30d": 2
    }
]



resp = requests.post(url, json=data)
print(resp.json())

