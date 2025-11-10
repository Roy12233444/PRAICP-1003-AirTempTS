# src/agents/wavelet_transient_agent.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from agents.agent_base import BaseAgent

try:
    import pywt
except Exception:
    pywt = None

class WaveletTransientAgent(BaseAgent):
    """
    Compute wavelet coefficients and per-scale energy:
      W(s, tau) = sum_t y_t * psi((t - tau)/s)
      E_s = (1/T) sum_tau |W(s,tau)|^2
    Implementation uses discrete wavelet transform (DWT) or CWT if available.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.wavelet = self.config.get("wavelet", "db4")
        self.scales = self.config.get("scales", [1,2,4,8,16])
        self.mode = self.config.get("mode", "dwt")  # 'dwt' or 'cwt'

    def fit(self, df: pd.DataFrame, target_col: str = "mean_temp"):
        if pywt is None:
            raise ImportError("pywt not installed. pip install pywt")
        # nothing to fit; scales chosen in config
        return self

    def transform(self, df: pd.DataFrame, target_col: str = "mean_temp") -> pd.DataFrame:
        if pywt is None:
            raise ImportError("pywt not installed. pip install pywt")
        y = df[target_col].astype(float).values
        out = df.copy().reset_index(drop=True)
        T = len(y)
        if self.mode == "dwt":
            # produce energy for each level of DWT (approx + details)
            coeffs = pywt.wavedec(y, self.wavelet, mode='periodization', level=None)
            # energy per coeff array
            for i, c in enumerate(coeffs):
                out[f"w_energy_dwt_{i}"] = np.sum(np.square(c)) / max(1, len(c))
        else:
            # CWT path (slower): produce mean energy per scale
            scales = self.scales
            coef, freqs = pywt.cwt(y, scales, self.wavelet)
            # coef shape (len(scales), len(y))
            for i, s in enumerate(scales):
                E_s = np.mean(np.abs(coef[i])**2)
                out[f"w_energy_cwt_{s}"] = E_s
        return out

    def fit_transform(self, df: pd.DataFrame, target_col: str = "mean_temp") -> pd.DataFrame:
        self.fit(df, target_col=target_col)
        return self.transform(df, target_col=target_col)
