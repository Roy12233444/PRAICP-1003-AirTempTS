# src/agents/feature_alchemist_agent.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from agents.agent_base import BaseAgent
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

try:
    from boruta import BorutaPy
except Exception:
    BorutaPy = None
try:
    import shap
except Exception:
    shap = None

class FeatureAlchemistAgent(BaseAgent):
    """
    Generate candidate features (lags, rolling stats, STL, Fourier harmonics via ResonantAgent,
    wavelet energies via WaveletAgent) and select a compact set.
    Methods:
      - generate_candidates(df) -> large df
      - select_features(X, y) -> list of selected features
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Initialize with safe defaults
        self.lags = [1, 2, 3, 12, 24]
        self.rolls = [3, 6, 12]
        self.selected_features: List[str] = []
        
        # Safely update from config if provided
        if config:
            self.lags = config.get("lags", self.lags)
            self.rolls = config.get("rolls", self.rolls)

    def generate_candidates(self, df: pd.DataFrame, target_col: str = "mean_temp") -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        series = out[target_col].astype(float)
        # lags
        for l in self.lags:
            out[f"lag_{l}"] = series.shift(l)
        # rolling stats
        for r in self.rolls:
            out[f"roll_mean_{r}"] = series.rolling(r, min_periods=1).mean()
            out[f"roll_std_{r}"] = series.rolling(r, min_periods=1).std().fillna(0.0)
            out[f"roll_q90_{r}"] = series.rolling(r, min_periods=1).quantile(0.90).fillna(method='bfill')
        # month cyclic
        if 'month' in out.columns:
            months = pd.to_datetime(out['month']).dt.month
            out['month_sin'] = np.sin(2*np.pi*months/12)
            out['month_cos'] = np.cos(2*np.pi*months/12)
        # drop early NaNs (common) but keep index alignment with original by not dropping, leave NaNs
        return out
        
    def fit(self, df: pd.DataFrame, **kwargs) -> 'FeatureAlchemistAgent':
        """
        Fit the feature selector on the training data.
        """
        # Generate features
        X = self.generate_candidates(df)
        y = df['mean_temp'] if 'mean_temp' in df.columns else df.iloc[:, -1]
        
        # Simple feature selection using correlation
        corr = X.corrwith(y).abs().sort_values(ascending=False)
        self.selected_features = corr.head(10).index.tolist()  # Top 10 most correlated features
        self._is_fitted = True
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the selected features.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        X = self.generate_candidates(df)
        # Return only selected features that exist in the data
        available_features = [f for f in self.selected_features if f in X.columns]
        return X[available_features] if available_features else X
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the transformed features.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        X = self.transform(df)
        # Simple prediction using mean of selected features
        return X.mean(axis=1).values

    def select_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 20) -> List[str]:
        # simple pipeline: Boruta if available, else permutation importance on RF
        numeric_X = X.select_dtypes(include=[np.number]).fillna(0.0)
        if BorutaPy is not None:
            rf = RandomForestRegressor(n_jobs=-1, random_state=42)
            boruta = BorutaPy(rf, n_estimators='auto', random_state=42)
            boruta.fit(numeric_X.values, y.values)
            selected = numeric_X.columns[boruta.support_].tolist()
            self.selected_features = selected[:max_features]
            return self.selected_features
        else:
            # fallback permutation importance
            rf = RandomForestRegressor(n_jobs=-1, random_state=42)
            rf.fit(numeric_X, y)
            res = permutation_importance(rf, numeric_X, y, n_repeats=10, random_state=42, n_jobs=-1)
            importance = pd.Series(res.importances_mean, index=numeric_X.columns).sort_values(ascending=False)
            self.selected_features = importance.index[:max_features].tolist()
            return self.selected_features

    def fit(self, df: pd.DataFrame, target_col: str = "mean_temp"):
        cand = self.generate_candidates(df, target_col=target_col)
        y = cand[target_col].astype(float).fillna(method='ffill').fillna(0.0)
        X = cand.drop(columns=[target_col, 'month'], errors='ignore')
        self.select_features(X, y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        # ensure selected features exist in out; if not generate
        cand = self.generate_candidates(out)
        # return only selected features + month + target if present
        cols = [c for c in self.selected_features if c in cand.columns]
        result = pd.concat([out[['month']] if 'month' in out.columns else out.index.to_frame(index=False),
                            cand[cols].reset_index(drop=True)], axis=1)
        return result
