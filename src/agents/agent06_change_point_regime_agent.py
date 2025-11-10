# src/agents/change_point_regime_agent.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from agents.agent_base import BaseAgent

try:
    import ruptures as rpt
except Exception:
    rpt = None

class ChangePointRegimeAgent(BaseAgent):
    """
    Detect change-points using ruptures and provide regime labels.
    Uses cost='l2' by default and Pelt or Binseg algorithms.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = self.config.get("model", "pelt")
        self.pen = self.config.get("pen", 10)
        self.breakpoints: List[int] = []

    def fit(self, df: pd.DataFrame, target_col: str = "mean_temp"):
        if rpt is None:
            raise ImportError("ruptures not installed. pip install ruptures")
        y = df[target_col].astype(float).values
        algo = rpt.Pelt(model="l2").fit(y) if self.model == "pelt" else rpt.Binseg(model="l2").fit(y)
        bkps = algo.predict(pen=self.pen)
        # ruptures returns last index as len(y); make breakpoints list
        self.breakpoints = bkps
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        n = len(out)
        labels = np.zeros(n, dtype=int)
        start = 0
        for i, b in enumerate(self.breakpoints):
            end = min(b, n)
            labels[start:end] = i
            start = end
        out["regime"] = labels
        return out
