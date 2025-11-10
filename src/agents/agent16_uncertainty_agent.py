import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from agents.agent_base import BaseAgent
from agents.constants import NUM_BOOTSTRAP_SAMPLES, CONFIDENCE_LEVEL

class UncertaintyAgent(BaseAgent):
    """
    Advanced uncertainty estimation agent for time series predictions.
    Implements bootstrap sampling, ensemble methods, and conformal prediction
    to quantify prediction uncertainty.
    """

    def __init__(self, method='bootstrap', n_estimators=10,
                 confidence_level=CONFIDENCE_LEVEL, n_bootstrap=NUM_BOOTSTRAP_SAMPLES):
        super().__init__()
        self.method = method  # 'bootstrap', 'ensemble', 'conformal'
        self.n_estimators = n_estimators
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.models = []
        self.calibration_data = None
        self.fitted = False

    def fit(self, X, y):
        """
        Fit the uncertainty estimation models.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X, y = np.array(X), np.array(y)

        if self.method == 'bootstrap':
            self._fit_bootstrap(X, y)
        elif self.method == 'ensemble':
            self._fit_ensemble(X, y)
        elif self.method == 'conformal':
            self._fit_conformal(X, y)

        self.fitted = True
        return self

    def predict(self, X, return_uncertainty=True):
        """
        Make predictions with uncertainty estimates.

        Args:
            X (np.ndarray): Feature matrix
            return_uncertainty (bool): Whether to return uncertainty estimates

        Returns:
            dict: Predictions and uncertainty measures
        """
        if not self.fitted:
            raise ValueError("Agent must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.method == 'bootstrap':
            return self._predict_bootstrap(X, return_uncertainty)
        elif self.method == 'ensemble':
            return self._predict_ensemble(X, return_uncertainty)
        elif self.method == 'conformal':
            return self._predict_conformal(X, return_uncertainty)

    def _fit_bootstrap(self, X, y):
        """Fit bootstrap models."""
        n_samples = len(X)
        self.models = []

        for _ in range(self.n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            # Train model on bootstrap sample
            model = LinearRegression()
            model.fit(X_boot, y_boot)
            self.models.append(model)

    def _predict_bootstrap(self, X, return_uncertainty):
        """Make predictions with bootstrap uncertainty."""
        predictions = np.array([model.predict(X) for model in self.models])

        result = {'predictions': np.mean(predictions, axis=0)}

        if return_uncertainty:
            result['uncertainty'] = {
                'std': np.std(predictions, axis=0),
                'confidence_interval': self._compute_confidence_interval(predictions),
                'prediction_interval': self._compute_prediction_interval(predictions)
            }

        return result

    def _fit_ensemble(self, X, y):
        """Fit ensemble model."""
        base_model = LinearRegression()
        self.ensemble_model = BaggingRegressor(
            base_estimator=base_model,
            n_estimators=self.n_estimators,
            random_state=42
        )
        self.ensemble_model.fit(X, y)

    def _predict_ensemble(self, X, return_uncertainty):
        """Make predictions with ensemble uncertainty."""
        # Get predictions from all estimators
        all_predictions = np.array([estimator.predict(X) for estimator in self.ensemble_model.estimators_])

        result = {'predictions': self.ensemble_model.predict(X)}

        if return_uncertainty:
            result['uncertainty'] = {
                'std': np.std(all_predictions, axis=0),
                'confidence_interval': self._compute_confidence_interval(all_predictions)
            }

        return result

    def _fit_conformal(self, X, y):
        """Fit conformal prediction model."""
        # Split data for calibration
        n_cal = int(0.2 * len(X))
        indices = np.random.permutation(len(X))
        cal_indices, train_indices = indices[:n_cal], indices[n_cal:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_cal, y_cal = X[cal_indices], y[cal_indices]

        # Train base model
        self.base_model = LinearRegression()
        self.base_model.fit(X_train, y_train)

        # Compute nonconformity scores on calibration set
        cal_predictions = self.base_model.predict(X_cal)
        self.cal_scores = np.abs(y_cal - cal_predictions)
        self.cal_scores.sort()

    def _predict_conformal(self, X, return_uncertainty):
        """Make predictions with conformal uncertainty."""
        predictions = self.base_model.predict(X)

        if return_uncertainty:
            # Compute prediction intervals using conformal prediction
            alpha = 1 - self.confidence_level
            q_level = np.ceil((len(self.cal_scores) + 1) * (1 - alpha)) / len(self.cal_scores)
            q_index = int(q_level * len(self.cal_scores)) - 1
            q_score = self.cal_scores[min(q_index, len(self.cal_scores) - 1)]

            lower_bounds = predictions - q_score
            upper_bounds = predictions + q_score

            result = {
                'predictions': predictions,
                'uncertainty': {
                    'conformal_interval': (lower_bounds, upper_bounds),
                    'q_score': q_score
                }
            }
        else:
            result = {'predictions': predictions}

        return result

    def _compute_confidence_interval(self, predictions):
        """Compute confidence interval from prediction distribution."""
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Assuming normal distribution
        z_score = 1.96  # 95% confidence
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred

        return (lower, upper)

    def _compute_prediction_interval(self, predictions):
        """Compute prediction interval (wider than confidence interval)."""
        # For simplicity, use wider bounds
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        z_score = 2.576  # 99% confidence for prediction interval
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred

        return (lower, upper)