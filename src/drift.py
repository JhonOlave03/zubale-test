import pandas as pd
import numpy as np
import json
import os
from scipy.stats import ks_2samp
import argparse

def psi(expected, actual, buckets=10):
    """
    Calcula el Population Stability Index (PSI) para columnas categóricas.
    """
    expected_counts = expected.value_counts(normalize=True)
    actual_counts = actual.value_counts(normalize=True)

    # Asegurarse que todas las categorías estén en ambos
    all_cats = set(expected_counts.index) | set(actual_counts.index)
    psi_val = 0.0
    for cat in all_cats:
        e = expected_counts.get(cat, 1e-6)  # evita log(0)
        a = actual_counts.get(cat, 1e-6)
        psi_val += (e - a) * np.log(e / a)
    return psi_val

def calculate_drift(ref_df, new_df):
    """
    Calcula drift por feature: KS para numéricas y PSI para categóricas.
    """
    drift_metrics = {}
    for col in ref_df.columns:
        if np.issubdtype(ref_df[col].dtype, np.number):
            # KS-test
            ks_stat, _ = ks_2samp(ref_df[col], new_df[col])
            drift_metrics[col] = float(ks_stat)
        else:
            # PSI para categóricas
            drift_metrics[col] = float(psi(ref_df[col], new_df[col]))
    return drift_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="CSV de referencia")
    parser.add_argument("--new", required=True, help="CSV de nuevos datos")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.ref)
    new_df = pd.read_csv(args.new)

    drift_metrics = calculate_drift(ref_df, new_df)
    threshold = 0.2
    overall_drift = any(v > threshold for v in drift_metrics.values())

    report = {
        "threshold": threshold,
        "overall_drift": overall_drift,
        "features": drift_metrics
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/drift_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Drift report saved to artifacts/drift_report.json")

if __name__ == "__main__":
    main()
