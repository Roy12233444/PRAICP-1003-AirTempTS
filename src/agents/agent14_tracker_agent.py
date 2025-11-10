import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from agents.agent_base import BaseAgent
from agents.constants import LOGS_DIR, LOG_LEVEL

class TrackerAgent(BaseAgent):
    """
    Advanced tracking agent for monitoring system performance and data quality.
    Tracks metrics, data statistics, model performance, and system health over time.
    Provides logging, alerting, and trend analysis capabilities.
    """

    def __init__(self, log_dir=LOGS_DIR, metrics_to_track=None, alert_thresholds=None):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics_to_track = metrics_to_track or [
            'mae', 'mse', 'rmse', 'mape', 'data_mean', 'data_std', 'data_missing_rate'
        ]
        self.alert_thresholds = alert_thresholds or {
            'mae': 10.0,
            'data_missing_rate': 0.05
        }

        self.history = defaultdict(list)
        self.alerts = []
        self.fitted = True  # Always fitted

    def fit(self, X=None, y=None):
        """
        Initialize tracking (no actual fitting needed).

        Args:
            X, y: Not used
        """
        return self

    def predict(self, data=None, predictions=None, targets=None, metadata=None):
        """
        Track metrics and data statistics.

        Args:
            data (np.ndarray): Input data
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth targets
            metadata (dict): Additional metadata

        Returns:
            dict: Tracking results and alerts
        """
        timestamp = datetime.now().isoformat()
        results = {'timestamp': timestamp, 'metrics': {}, 'alerts': []}

        # Track data statistics
        if data is not None:
            results['metrics'].update(self._track_data_stats(data))

        # Track prediction performance
        if predictions is not None and targets is not None:
            results['metrics'].update(self._track_performance(predictions, targets))

        # Track custom metadata
        if metadata:
            results['metrics'].update(metadata)

        # Check for alerts
        alerts = self._check_alerts(results['metrics'])
        results['alerts'] = alerts

        # Update history
        for key, value in results['metrics'].items():
            self.history[key].append((timestamp, value))

        # Log results
        self._log_results(results)

        return results

    def _track_data_stats(self, data):
        """Track data quality statistics."""
        if isinstance(data, pd.DataFrame):
            data = data.values

        stats = {}
        stats['data_shape'] = data.shape
        stats['data_mean'] = float(np.mean(data))
        stats['data_std'] = float(np.std(data))
        stats['data_min'] = float(np.min(data))
        stats['data_max'] = float(np.max(data))
        stats['data_missing_rate'] = float(np.isnan(data).sum() / data.size)

        # Additional time series stats
        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            series = data.flatten()
            stats['data_autocorr_lag1'] = float(np.corrcoef(series[:-1], series[1:])[0, 1])
            stats['data_trend'] = float(np.polyfit(range(len(series)), series, 1)[0])

        return stats

    def _track_performance(self, predictions, targets):
        """Track prediction performance metrics."""
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()

        metrics = {}
        errors = predictions - targets

        metrics['mae'] = float(np.mean(np.abs(errors)))
        metrics['mse'] = float(np.mean(errors ** 2))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))

        # MAPE (avoid division by zero)
        mask = targets != 0
        if mask.any():
            metrics['mape'] = float(np.mean(np.abs(errors[mask] / targets[mask])) * 100)
        else:
            metrics['mape'] = 0.0

        # Additional metrics
        metrics['max_error'] = float(np.max(np.abs(errors)))
        metrics['r_squared'] = float(1 - np.sum(errors**2) / np.sum((targets - np.mean(targets))**2))

        return metrics

    def _check_alerts(self, metrics):
        """Check metrics against alert thresholds."""
        alerts = []
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                alert = {
                    'metric': metric,
                    'value': metrics[metric],
                    'threshold': threshold,
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
                self.alerts.append(alert)

        return alerts

    def _log_results(self, results):
        """Log tracking results to file."""
        log_file = os.path.join(self.log_dir, 'tracker_log.jsonl')

        with open(log_file, 'a') as f:
            json.dump(results, f)
            f.write('\n')

    def get_history(self, metric=None, limit=None):
        """Retrieve tracking history."""
        if metric:
            return self.history[metric][-limit:] if limit else self.history[metric]
        else:
            return dict(self.history)

    def get_alerts(self, limit=None):
        """Retrieve recent alerts."""
        return self.alerts[-limit:] if limit else self.alerts