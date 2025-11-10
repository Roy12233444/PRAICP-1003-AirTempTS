# src/agents/utils.py
import numpy as np
import pandas as pd
from typing import List, Optional
import json, os
from joblib import dump, load

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_json(obj, path: str):
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def reindex_features(df: pd.DataFrame, expected: List[str], fill_value=0.0) -> pd.DataFrame:
    # Guarantee ordering and missing columns are created
    out = df.copy()
    for c in expected:
        if c not in out.columns:
            out[c] = fill_value
    return out[expected].astype(float)

def top_k_indices(arr, k):
    idx = np.argsort(np.abs(arr))[::-1]
    return idx[:k]
