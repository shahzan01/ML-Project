import os, math
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

DATASET_PATH = os.getenv("DATASET_PATH", "data/cardekho_dataset.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cardekho_linreg.joblib")
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_schema.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

def extract_model(name: str) -> str:
    if not isinstance(name, str): return "unknown"
    parts = name.split()
    if not parts: return "unknown"
    return " ".join(parts[:2]) if len(parts) >= 2 else parts[0]

df["model"] = df["name"].apply(extract_model)
df = df.drop_duplicates().reset_index(drop=True)
df = df.dropna(subset=["selling_price"])
df = df[df["selling_price"] > 0]

if "km_driven" in df.columns:
    cap = df["km_driven"].quantile(0.99)
    df["km_driven"] = np.where(df["km_driven"] > cap, cap, df["km_driven"])

target = "selling_price"
numeric = [c for c in ["year", "km_driven"] if c in df.columns]
categorical = [c for c in ["model", "fuel", "seller_type", "transmission", "owner"] if c in df.columns]

X = df[numeric + categorical].copy()
y = df[target].copy()

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric),
])

pipe = Pipeline([("preprocess", pre), ("model", LinearRegression())])

# Fit full data (simple baseline). If you prefer, fit on train split and evaluate first.
pipe.fit(X, y)

# Save schema and model
joblib.dump({"numeric": numeric, "categorical": categorical}, SCHEMA_PATH)
joblib.dump(pipe, MODEL_PATH)

print(f"Saved model -> {MODEL_PATH}")
print(f"Saved schema -> {SCHEMA_PATH}")
