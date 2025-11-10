import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind
from sklearn.ensemble import IsolationForest
from agents.agent_base import BaseAgent
from agents.constants import DRIFT_SIGNIFICANCE_LEVEL, DRIFT_WINDOW_SIZE

class DriftDetectorAgent(BaseAgent):
    """
    Advanced drift detection agent for time series data.
    Implements statistical tests (KS, t-test) and ML-based detection (Isolation Forest).
    Monitors data distribution changes over time.
    """

    def __init__(self, method='statistical', significance_level=DRIFT_SIGNIFICANCE_LEVEL,
                 window_size=DRIFT_WINDOW_SIZE, contamination=0.1):
        super().__init__()
        self.method = method  # 'statistical', 'ml', or 'hybrid'
        self.significance_level = significance_level
        self.window_size = window_size
        self.contamination = contamination
        self.reference_data = None
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.fitted = False
        self.drift_history = []

    def fit(self, X, y=None):
        """
        Fit the drift detector on reference data.

        Args:
            X (np.ndarray or pd.DataFrame): Reference time series data
            y (np.ndarray, optional): Not used
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        self.reference_data = X.copy()

        if self.method in ['ml', 'hybrid']:
            # Fit isolation forest on reference data
            self.isolation_forest.fit(X)

        self.fitted = True
        return self

    def predict(self, X):
        """
        Detect drift in new data compared to reference.

        Args:
            X (np.ndarray or pd.DataFrame): New time series data to check

        Returns:
            dict: Drift detection results
        """
        if not self.fitted:
            raise ValueError("Agent must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'method_used': self.method,
            'details': {}
        }

        if self.method == 'statistical':
            drift_score, details = self._statistical_drift_detection(X)
        elif self.method == 'ml':
            drift_score, details = self._ml_drift_detection(X)
        else:  # hybrid
            stat_score, stat_details = self._statistical_drift_detection(X)
            ml_score, ml_details = self._ml_drift_detection(X)
            drift_score = (stat_score + ml_score) / 2
            details = {**stat_details, **ml_details}

        results['drift_score'] = drift_score
        results['drift_detected'] = drift_score > self.significance_level
        results['details'] = details

        # Update history
        self.drift_history.append(results)

        return results

    def _statistical_drift_detection(self, X):
        """Perform statistical drift detection using KS test and t-test."""
        details = {}
        scores = []

        # Flatten data for statistical tests
        ref_flat = self.reference_data.flatten()
        new_flat = X.flatten()

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = ks_2samp(ref_flat, new_flat)
        details['ks_statistic'] = ks_stat
        details['ks_p_value'] = ks_p
        scores.append(1 - ks_p)  # Higher score = more drift

        # t-test
        try:
            t_stat, t_p = ttest_ind(ref_flat, new_flat)
            details['t_statistic'] = t_stat
            details['t_p_value'] = t_p
            scores.append(1 - t_p)
        except:
            details['t_test_error'] = "Could not perform t-test"
            scores.append(0)

        # Mean and std differences
        mean_diff = abs(np.mean(ref_flat) - np.mean(new_flat))
        std_diff = abs(np.std(ref_flat) - np.std(new_flat))
        details['mean_difference'] = mean_diff
        details['std_difference'] = std_diff
        scores.append(min(mean_diff / np.std(ref_flat), 1))  # Normalized
        scores.append(min(std_diff / np.std(ref_flat), 1))

        overall_score = np.mean(scores)
        return overall_score, details

    def _ml_drift_detection(self, X):
        """ML-based drift detection using Isolation Forest."""
        # Predict anomaly scores for new data
        scores = self.isolation_forest.decision_function(X)
        anomaly_score = -np.mean(scores)  # Higher = more anomalous

        # Compare to reference scores
        ref_scores = self.isolation_forest.decision_function(self.reference_data)
        ref_anomaly_score = -np.mean(ref_scores)

        drift_score = max(0, anomaly_score - ref_anomaly_score)

        details = {
            'new_anomaly_score': anomaly_score,
            'reference_anomaly_score': ref_anomaly_score,
            'drift_score': drift_score
        }

        return drift_score, details