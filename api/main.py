from pathlib import Path
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from api.logging_config import setup_logging

logger = setup_logging()

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"
MODEL_VERSION = "1.0.0"
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
    return {"model": "Random Forest", "version": MODEL_VERSION}


@app.post("/predict")
def predict(data: ClientData, request: Request):
    start = time.time()
    client_ip = request.client.host if request.client else "unknown"

    logger.info("Prediction request: client=%s model_version=%s", client_ip, MODEL_VERSION,
                extra={"event": "prediction_request", "client_ip": client_ip,
                       "model_version": MODEL_VERSION})
    try:
        values = pd.DataFrame([data.model_dump()])[FEATURES]
        prediction = int(model.predict(values)[0])
        probability = float(model.predict_proba(values)[0][1])
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True,
                     extra={"event": "prediction_error", "client_ip": client_ip,
                            "model_version": MODEL_VERSION, "error": str(exc)})
        raise HTTPException(status_code=500, detail="Erreur lors de la prédiction")

    duration_ms = round((time.time() - start) * 1000, 2)
    label = "CHURN" if prediction == 1 else "NO_CHURN"

    logger.info("Prediction done: result=%s probability=%.4f duration_ms=%s",
                label, probability, duration_ms,
                extra={"event": "prediction", "client_ip": client_ip,
                       "model_version": MODEL_VERSION, "prediction": label,
                       "probability": round(probability, 4),
                       "response_time_ms": duration_ms})

    return {"prediction": label, "probability": round(probability, 4),
            "model_version": MODEL_VERSION, "duration_ms": duration_ms}


Instrumentator().instrument(app).expose(app)