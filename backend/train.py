import os, math
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Paths
DATASET_PATH = os.getenv("DATASET_PATH", "data/cardekho_dataset.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "cardekho_rf.joblib")
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_schema.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# --- Feature engineering ---
def extract_model(name: str) -> str:
    if not isinstance(name, str): 
        return "unknown"
    parts = name.split()
    if not parts: 
        return "unknown"
    return " ".join(parts[:2]) if len(parts) >= 2 else parts[0]

df["model"] = df["name"].apply(extract_model)
df = df.drop_duplicates().reset_index(drop=True)
df = df.dropna(subset=["selling_price"])
df = df[df["selling_price"] > 0]

if "km_driven" in df.columns:
    cap = df["km_driven"].quantile(0.99)
    df["km_driven"] = np.where(df["km_driven"] > cap, cap, df["km_driven"])

# --- Columns ---
target = "selling_price"
numeric = [c for c in ["year", "km_driven"] if c in df.columns]
categorical = [c for c in ["model", "fuel", "seller_type", "transmission", "owner"] if c in df.columns]

X = df[numeric + categorical].copy()
y = df[target].copy()

# --- Preprocess + Model ---
pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ]
)

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline([
    ("preprocess", pre),
    ("model", rf),
])

# --- Optional: quick CV check (RMSE) ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = -cross_val_score(pipe, X, y, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
print(f"5-fold CV RMSE: mean={rmse_scores.mean():.2f}, std={rmse_scores.std():.2f}")

# --- Fit on full data and persist ---
pipe.fit(X, y)

joblib.dump({"numeric": numeric, "categorical": categorical}, SCHEMA_PATH)
joblib.dump(pipe, MODEL_PATH)

print(f"Saved model -> {MODEL_PATH}")
print(f"Saved schema -> {SCHEMA_PATH}")
