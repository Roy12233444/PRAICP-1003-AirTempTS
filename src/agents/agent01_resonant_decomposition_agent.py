# src/agents/resonant_decomposition_agent.py
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from typing import Optional, Dict, Any, List
from agents.agent_base import BaseAgent
from agents.utils import save_json, load_json, ensure_dir

class ResonantDecompositionAgent(BaseAgent):
    """
    - Fit: compute DFT on target, pick top-k spectral peaks (omega_k)
    - Transform: add harmonic features sin(omega*t), cos(omega*t), and STL trend/seasonal/resid
    Math summary implemented:
      Y(omega) = sum_t y_t e^{-i omega t}
      select omega_k where |Y(omega)| max
      h_sin = sin(omega_k * t), h_cos = cos(...)
      STL: y_t = T_t + S_t + R_t
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k = int(self.config.get("k", 3))
        self.period = int(self.config.get("period", 12))
        self.omegas: List[float] = []

    def fit(self, df: pd.DataFrame, target_col: str = "mean_temp") -> "ResonantDecompositionAgent":
        y = df[target_col].astype(float).values
        T = len(y)
        if T < 3:
            raise ValueError("Not enough rows to compute DFT")
        y_demean = y - np.mean(y)
        Y = np.fft.rfft(y_demean)
        freqs = np.fft.rfftfreq(T, d=1.0)  # cycles per timestep (month)
        amps = np.abs(Y)
        # skip DC (index 0) and choose top k nonzero freqs
        candidates = np.argsort(amps)[::-1]
        selected = []
        for idx in candidates:
            if idx == 0:
                continue
            omega = 2.0 * np.pi * freqs[idx]
            if not any(np.isclose(omega, w, atol=1e-8) for w in selected):
                selected.append(omega)
            if len(selected) >= self.k:
                break
        self.omegas = selected
        # store nothing else heavy; STL will be computed per-transform
        return self

    def transform(self, df: pd.DataFrame, target_col: str = "mean_temp") -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        T = len(out)
        t = np.arange(T)
        # harmonics
        for i, omega in enumerate(self.omegas):
            out[f"h_sin_{i+1}"] = np.sin(omega * t)
            out[f"h_cos_{i+1}"] = np.cos(omega * t)
        # STL decomposition; robust True for outliers
        try:
            stl = STL(out[target_col].astype(float), period=self.period, robust=True)
            res = stl.fit()
            out["stl_trend"] = res.trend
            out["stl_seasonal"] = res.seasonal
            out["stl_resid"] = res.resid
        except Exception:
            # When STL fails (tiny series), fill with NaN
            out["stl_trend"] = np.nan
            out["stl_seasonal"] = np.nan
            out["stl_resid"] = np.nan
        return out

    def fit_transform(self, df: pd.DataFrame, target_col: str = "mean_temp") -> pd.DataFrame:
        self.fit(df, target_col=target_col)
        return self.transform(df, target_col=target_col)

    def save(self, path: str):
        ensure_dir(path)
        save_json({"k": self.k, "period": self.period, "omegas": self.omegas}, path + ".json")

    def load(self, path: str):
        state = load_json(path + ".json")
        self.k = state["k"]
        self.period = state["period"]
        self.omegas = state["omegas"]
        return self
