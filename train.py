import argparse
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from xgboost import XGBClassifier
import subprocess
from datetime import datetime


def main(data_path):
    os.makedirs("artifacts", exist_ok=True)

    # Cargar dataset
    df = pd.read_csv(data_path)

    # Features y target
    target = "churned"
    X = df.drop(columns=[target])
    y = df[target]

    # Columnas categóricas y numéricas
    categorical_features = ["plan_type", "contract_type", "autopay", "is_promo_user"]
    numeric_features = [
        "add_on_count", "tenure_months", "monthly_usage_gb",
        "avg_latency_ms", "support_tickets_30d", "discount_pct",
        "payment_failures_90d", "downtime_hours_30d"
    ]

    # Preprocesamiento
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features)
        ]
    )

    # Modelo
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    # Pipeline completo
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenamiento
    clf.fit(X_train, y_train)

    # Predicciones
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    # Métricas
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_sha = None

    metrics = {
        "roc_auc": roc_auc_score(y_val, y_proba),
        "pr_auc": average_precision_score(y_val, y_proba),
        "accuracy": accuracy_score(y_val, y_pred),
        "timestamp": datetime.utcnow().isoformat(),
        "git_sha": git_sha,
    }

    # Guardar métricas
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Guardar modelo y pipeline
    joblib.dump(clf, "artifacts/model.pkl")
    joblib.dump(preprocessor, "artifacts/feature_pipeline.pkl")

    print("Entrenamiento finalizado. Métricas:")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Ruta al dataset CSV")
    args = parser.parse_args()

    main(args.data)
