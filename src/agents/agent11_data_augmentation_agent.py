import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Use absolute import for better compatibility
from src.agents.agent_base import BaseAgent
from src.agents.constants import (
    AUGMENTATION_NOISE_FACTOR, 
    AUGMENTATION_TIME_WARP_FACTOR, 
    AUGMENTATION_SCALING_FACTOR
)

class DataAugmentationAgent(BaseAgent):
    """
    Advanced data augmentation agent for time series data.
    Implements noise injection, time warping, scaling, and dropout augmentations.
    Learns data statistics during fit and applies augmentations during predict.
    """

    def __init__(self, noise_factor=AUGMENTATION_NOISE_FACTOR,
                 time_warp_factor=AUGMENTATION_TIME_WARP_FACTOR,
                 scaling_factor=AUGMENTATION_SCALING_FACTOR,
                 dropout_prob=0.1):
        super().__init__()
        self.noise_factor = noise_factor
        self.time_warp_factor = time_warp_factor
        self.scaling_factor = scaling_factor
        self.dropout_prob = dropout_prob
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X, y=None):
        """
        Fit the augmentation agent by learning data statistics.

        Args:
            X (np.ndarray or pd.DataFrame): Time series data
            y (np.ndarray, optional): Target values (not used)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        # Learn scaling parameters
        self.scaler.fit(X.reshape(-1, X.shape[-1]) if X.ndim > 2 else X)

        # Store data statistics
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.fitted = True

        return self

    def predict(self, X, augmentation_type='random', num_augmentations=1):
        """
        Apply augmentations to the input data.

        Args:
            X (np.ndarray or pd.DataFrame): Input time series data
            augmentation_type (str): Type of augmentation ('noise', 'warp', 'scale', 'dropout', 'random')
            num_augmentations (int): Number of augmented versions to generate

        Returns:
            list: List of augmented data arrays
        """
        if not self.fitted:
            raise ValueError("Agent must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)

        augmented_data = []

        for _ in range(num_augmentations):
            X_aug = X.copy()

            if augmentation_type == 'noise' or (augmentation_type == 'random' and np.random.rand() < 0.25):
                X_aug = self._add_noise(X_aug)
            elif augmentation_type == 'warp' or (augmentation_type == 'random' and np.random.rand() < 0.5):
                X_aug = self._time_warp(X_aug)
            elif augmentation_type == 'scale' or (augmentation_type == 'random' and np.random.rand() < 0.75):
                X_aug = self._scale(X_aug)
            elif augmentation_type == 'dropout' or augmentation_type == 'random':
                X_aug = self._dropout(X_aug)

            augmented_data.append(X_aug)

        return augmented_data

    def _add_noise(self, X):
        """Add Gaussian noise to the data."""
        noise = np.random.normal(0, self.noise_factor * self.std, X.shape)
        return X + noise

    def _time_warp(self, X):
        """Apply time warping augmentation."""
        if X.ndim == 1:
            # For 1D series
            warp_factor = 1 + np.random.uniform(-self.time_warp_factor, self.time_warp_factor)
            new_length = int(len(X) * warp_factor)
            indices = np.linspace(0, len(X) - 1, new_length)
            return np.interp(indices, np.arange(len(X)), X)
        else:
            # For multi-dimensional
            warped = []
            for i in range(X.shape[-1]):
                warp_factor = 1 + np.random.uniform(-self.time_warp_factor, self.time_warp_factor)
                new_length = int(X.shape[0] * warp_factor)
                indices = np.linspace(0, X.shape[0] - 1, new_length)
                warped.append(np.interp(indices, np.arange(X.shape[0]), X[:, i]))
            return np.column_stack(warped)

    def _scale(self, X):
        """Apply scaling augmentation."""
        scale_factor = 1 + np.random.uniform(-self.scaling_factor, self.scaling_factor)
        return X * scale_factor

    def _dropout(self, X):
        """Apply dropout augmentation (randomly set values to zero)."""
        mask = np.random.rand(*X.shape) > self.dropout_prob
        return X * mask