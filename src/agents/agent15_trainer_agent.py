import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from agents.agent_base import BaseAgent
from agents.constants import DEFAULT_EPOCHS, LEARNING_RATE, VALIDATION_SPLIT

class TrainerAgent(BaseAgent):
    """
    Advanced training agent for time series forecasting models.
    Supports multiple algorithms with hyperparameter tuning and cross-validation.
    Implements model selection, training, and evaluation.
    """

    def __init__(self, model_type='rf', cv_folds=5, tune_hyperparams=True):
        super().__init__()
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.tune_hyperparams = tune_hyperparams
        self.model = None
        self.best_params = None
        self.training_history = []
        self.fitted = False

    def fit(self, X, y):
        """
        Train the forecasting model.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X, y = np.array(X), np.array(y)

        # Select and initialize model
        self.model = self._get_model()

        if self.tune_hyperparams:
            # Hyperparameter tuning with time series cross-validation
            param_grid = self._get_param_grid()
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)

            grid_search = GridSearchCV(
                self.model, param_grid, cv=tscv,
                scoring='neg_mean_absolute_error', n_jobs=-1
            )
            grid_search.fit(X, y)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            # Store CV results
            self.training_history = grid_search.cv_results_
        else:
            # Simple training
            self.model.fit(X, y)

        self.fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.fitted:
            raise ValueError("Agent must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict(X)

    def _get_model(self):
        """Get the appropriate model based on type."""
        if self.model_type == 'linear':
            return LinearRegression()
        elif self.model_type == 'rf':
            return RandomForestRegressor(random_state=42, n_estimators=100)
        elif self.model_type == 'svm':
            return SVR(kernel='rbf')
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _get_param_grid(self):
        """Get hyperparameter grid for tuning."""
        if self.model_type == 'linear':
            return {}  # No hyperparameters to tune
        elif self.model_type == 'rf':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'svm':
            return {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'epsilon': [0.01, 0.1, 0.2]
            }
        return {}

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets

        Returns:
            dict: Evaluation metrics
        """
        if not self.fitted:
            raise ValueError("Agent must be fitted before evaluation")

        predictions = self.predict(X_test)

        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }

        return metrics

    def get_feature_importance(self):
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        else:
            return None