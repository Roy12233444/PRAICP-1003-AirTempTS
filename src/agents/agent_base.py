# src/agents/agent_base.py
from typing import Any, Dict, Optional, Union
import pandas as pd
import joblib
import os


class BaseAgent:
    """
    Minimal BaseAgent interface for all agents.

    Concrete agents SHOULD implement:
      - fit(self, df: pd.DataFrame, **kwargs) -> "BaseAgent"
      - transform(self, df: pd.DataFrame) -> pd.DataFrame
    Optional:
      - predict(self, df: pd.DataFrame) -> Union[float, Dict[str, Any], list]
      - save / load are implemented here using joblib

    Helpers:
      - fit_transform: convenience wrapper
      - predict_with_fallback: attempts predict -> transform -> naive mean
      - to_dict / from_dict: simple serialization helpers for config/state
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config.copy() if config else {}
        # optional internal state placeholder
        self._state: Dict[str, Any] = {}
        # Initialize with default values to prevent NoneType errors
        self.model = None
        self._is_fitted = False

    # -------------------------
    # Required interface methods
    # -------------------------
    def fit(self, df: pd.DataFrame, **kwargs) -> "BaseAgent":
        """
        Train or adapt agent on df (expects pandas DataFrame).
        Must be implemented by child classes. Return self.
        """
        raise NotImplementedError("fit() must be implemented by the concrete agent")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produces features or DataFrame with additional columns.
        Must be implemented by child classes.
        """
        raise NotImplementedError("transform() must be implemented by the concrete agent")

    # -------------------------
    # Optional / convenience
    # -------------------------
    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Convenience: fit then transform."""
        self.fit(df, **kwargs)
        return self.transform(df)

    def predict(self, df: pd.DataFrame) -> Union[float, Dict[str, Any], list]:
        """
        Optional forecasting/prediction entrypoint.
        If not implemented, users can call predict_with_fallback which will
        attempt to use transform() or a naive strategy.
        """
        raise NotImplementedError("predict() not implemented for this agent")

    # -------------------------
    # Helpers: robust predict
    # -------------------------
    def predict_with_fallback(self, df: pd.DataFrame, target_col: str = "mean_temp") -> Dict[str, Any]:
        """
        Try predict(); if missing, try transform() and extract likely prediction columns.
        If neither exists, fallback to naive mean of last 12 entries (if available).
        Returns dict: {'mu': float, 'var': float, 'mode': str}
        """
        # 1) try predict()
        try:
            out = self.predict(df)
            if isinstance(out, dict):
                mu = float(out.get("mu", float("nan")))
                var = float(out.get("var", out.get("sigma", 0.0) ** 2 if out.get("sigma") else 0.0) or 0.0)
                return {"mu": mu, "var": var, "mode": "predict_dict", "raw": out}
            if isinstance(out, (list, tuple)):
                return {"mu": float(out[0]), "var": 0.0, "mode": "predict_array", "raw": out}
            return {"mu": float(out), "var": 0.0, "mode": "predict_scalar", "raw": out}
        except NotImplementedError:
            pass
        except Exception:
            # swallow and try transform fallback
            pass

        # 2) try transform()
        try:
            df_out = self.transform(df)
            if isinstance(df_out, pd.DataFrame):
                # common prediction column names to check
                for col in ("mu", "pred", "prediction", "forecast", "mean_temp", "stl_trend"):
                    if col in df_out.columns:
                        val = df_out[col].iloc[-1]
                        return {"mu": float(val), "var": 0.0, "mode": f"transform_column:{col}"}
                # if seasonal/trend available, combine
                if "stl_trend" in df_out.columns:
                    trend = float(df_out["stl_trend"].iloc[-1])
                    season = float(df_out.get("stl_seasonal", pd.Series([0.0])).iloc[-1]) if "stl_seasonal" in df_out.columns else 0.0
                    return {"mu": float(trend + season), "var": 0.0, "mode": "stl_combination"}
        except NotImplementedError:
            pass
        except Exception:
            pass

        # 3) naive fallback: mean of last 12 values if target exists
        if target_col in df.columns:
            try:
                arr = df[target_col].dropna().astype(float)
                window = min(12, len(arr))
                if window > 0:
                    mu = float(arr.iloc[-window:].mean())
                    var = float(arr.iloc[-window:].var() if window > 1 else 0.0)
                    return {"mu": mu, "var": var, "mode": "naive_mean", "window": window}
            except Exception:
                pass

        # final fallback
        return {"mu": float("nan"), "var": float("nan"), "mode": "fail"}

    # -------------------------
    # Persistence helpers
    # -------------------------
    def save(self, path: str, overwrite: bool = True) -> None:
        """
        Save agent (joblib). Creates parent dirs if needed.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if (not overwrite) and os.path.exists(path):
            raise FileExistsError(f"File exists and overwrite=False: {path}")
        joblib.dump(self, path)

    def load(self, path: str) -> "BaseAgent":
        """
        Load agent state from joblib and update self.
        Returns self for chaining.
        """
        loaded = joblib.load(path)
        if hasattr(loaded, "__dict__"):
            self.__dict__.update(loaded.__dict__)
        return self

    # -------------------------
    # Serialization helpers
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a lightweight dict representation (config + non-callable state).
        Agents can override if they have large non-serializable state.
        """
        state = {k: v for k, v in self.__dict__.items() if not callable(v) and k not in ("_state",)}
        return {"config": self.config, "state": state}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseAgent":
        inst = cls(d.get("config", {}))
        state = d.get("state", {})
        for k, v in state.items():
            setattr(inst, k, v)
        return inst

    # -------------------------
    # Representation
    # -------------------------
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} config={self.config}>"
