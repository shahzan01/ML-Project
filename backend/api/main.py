import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

MODEL_PATH = os.getenv("MODEL_PATH", "models/cardekho_rf.joblib")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "models/feature_schema.joblib")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

app = FastAPI(title="Car Price Predictor")

origins = [o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first (python train.py).")
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Schema not found at {SCHEMA_PATH}. Train first (python train.py).")
    pipe = joblib.load(MODEL_PATH)
    schema = joblib.load(SCHEMA_PATH)
    if not isinstance(schema, dict) or "numeric" not in schema or "categorical" not in schema:
        raise ValueError("Invalid schema file.")
    return pipe, schema

pipe, schema = load_artifacts()
NUM = schema["numeric"]
CAT = schema["categorical"]

class CarInput(BaseModel):
    year: int = Field(..., ge=1980, le=2100)
    km_driven: int = Field(..., ge=0)
    model: str
    fuel: str
    seller_type: str
    transmission: str
    owner: str

class PredictRequest(BaseModel):
    items: list[CarInput]

class PredictResponse(BaseModel):
    predictions: list[float]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.items:
        raise HTTPException(400, "Empty items")
    rows = [{
        "year": it.year,
        "km_driven": it.km_driven,
        "model": it.model,
        "fuel": it.fuel,
        "seller_type": it.seller_type,
        "transmission": it.transmission,
        "owner": it.owner
    } for it in req.items]

    # Ensure column order matches training
    X = pd.DataFrame(rows, columns=NUM + CAT)
    preds = pipe.predict(X)
    return PredictResponse(predictions=[float(p) for p in preds])

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )

