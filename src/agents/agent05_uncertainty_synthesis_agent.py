# src/agents/uncertainty_synthesis_agent.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from agents.agent_base import BaseAgent

class UncertaintySynthesisAgent(BaseAgent):
    """
    Implements:
      - split-conformal calibration for residual-based intervals
      - bootstrap ensemble intervals
    API:
      - calibrate(y_cal, yhat_cal): compute q for desired alpha
      - predict_interval(yhat_pred, alpha): return intervals using q
      - bootstrap_intervals(train_X, train_y, X_pred, model_fn, n_boot)
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.q_cache = {}  # store quantiles per alpha

    def calibrate(self, y_cal, yhat_cal, alpha=0.05):
        # nonconformity scores
        s = np.abs(np.array(y_cal) - np.array(yhat_cal))
        q = np.quantile(s, 1.0 - alpha)
        self.q_cache[alpha] = float(q)
        return q

    def predict_interval_conformal(self, yhat_pred, alpha=0.05):
        if alpha not in self.q_cache:
            raise ValueError("alpha not calibrated; call calibrate() on a calibration set first")
        q = self.q_cache[alpha]
        yhat = np.array(yhat_pred)
        lower = yhat - q
        upper = yhat + q
        return lower, upper

    def bootstrap_intervals(self, X_train, y_train, X_pred, model_fn, n_boot=100, random_state=42):
        """
        model_fn: function(X_train, y_train) -> fitted_model with predict(X)
        Returns: lower, median, upper arrays for X_pred
        """
        rng = np.random.RandomState(random_state)
        preds = np.zeros((n_boot, len(X_pred)))
        n = len(X_train)
        for i in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            Xs = X_train.iloc[idx]
            ys = y_train.iloc[idx]
            m = model_fn(Xs, ys)
            preds[i, :] = m.predict(X_pred)
        lower = np.percentile(preds, 2.5, axis=0)
        median = np.percentile(preds, 50, axis=0)
        upper = np.percentile(preds, 97.5, axis=0)
        return lower, median, upper
