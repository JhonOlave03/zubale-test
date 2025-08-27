import os
import json

def test_artifacts_and_metrics():
    assert os.path.exists("artifacts/model.pkl")
    assert os.path.exists("artifacts/feature_pipeline.pkl")
    assert os.path.exists("artifacts/metrics.json")

    with open("artifacts/metrics.json") as f:
        metrics = json.load(f)

    assert metrics["roc_auc"] >= 0.83
