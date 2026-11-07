from pathlib import Path
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"

FEATURES = ["age", "anciennete", "revenu",
            "nb_reclamations", "montant_mensuel", "retards_paiement"]

app = FastAPI(title="Churn Prediction API", version="1.0.0")

model = joblib.load(MODEL_PATH)


class ClientData(BaseModel):
    age: int
    anciennete: int
    revenu: float
    nb_reclamations: int
    montant_mensuel: float
    retards_paiement: int


@app.get("/")
def root():
    return {"application": "Churn Prediction API", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "UP"}


@app.get("/model/info")
def model_info():
    return {"model": "Random Forest", "version": "1.0.0"}


@app.post("/predict")
def predict(data: ClientData):
    start = time.time()

    # DataFrame avec les mêmes noms de colonnes qu'à l'entraînement
    values = pd.DataFrame([data.model_dump()])[FEATURES]

    prediction = model.predict(values)[0]
    probability = model.predict_proba(values)[0][1]

    duration = time.time() - start
    print(f"prediction={prediction}, probability={probability:.4f}, duration={duration:.4f}s")

    return {
        "prediction": "CHURN" if prediction == 1 else "NO_CHURN",
        "probability": round(float(probability), 4),
        "model_version": "1.0.0",
        "duration_ms": round(duration * 1000, 2),
    }


Instrumentator().instrument(app).expose(app)