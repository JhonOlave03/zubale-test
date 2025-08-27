from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
import os

# Cargar el modelo entrenado y el pipeline de features
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PIPELINE_PATH = os.path.join("artifacts", "feature_pipeline.pkl")

model = joblib.load(MODEL_PATH)
#pipeline = joblib.load(PIPELINE_PATH)

app = FastAPI(title="Churn Prediction API")

# Definir esquema de entrada con Pydantic
class CustomerFeatures(BaseModel):
    plan_type: str
    contract_type: str
    autopay: str
    is_promo_user: str
    add_on_count: float
    tenure_months: float
    monthly_usage_gb: float
    avg_latency_ms: float
    support_tickets_30d: float
    discount_pct: float
    payment_failures_90d: float
    downtime_hours_30d: float

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customers: List[CustomerFeatures]):
    try:
        # Convertir lista de objetos a lista de diccionarios
        data = [c.dict() for c in customers]
        df = pd.DataFrame(data)

        print("Columns in request:", df.columns.tolist())
        print("Shape of df:", df.shape)
        #print("Pipeline expects:", getattr(pipeline, 'feature_names_in_', 'No info'))

        # Transformar con el pipeline
        #X = pipeline.transform(df)
        X = df

        # Probabilidades
        probs = model.predict_proba(X)[:, 1]
        preds = model.predict(X)

        # Respuesta
        results = []
        for i in range(len(df)):
            results.append({
                "probability": float(probs[i]),
                "prediction": int(preds[i])
            })

        return {"results": results}

    except Exception as e:
        import traceback
        print("ERROR TRACEBACK:\n", traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")
