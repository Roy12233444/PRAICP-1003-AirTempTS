import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from agents.agent_base import BaseAgent
from agents.constants import DATA_DIR, DEFAULT_DATA_PATH

class DataIngestAgent(BaseAgent):
    """
    Advanced data ingestion agent for time series data.
    Handles loading from various formats, preprocessing, missing value imputation,
    normalization, and feature engineering.
    """

    def __init__(self, data_path=None, imputation_strategy='mean',
                 normalization='standard', feature_engineering=True):
        super().__init__()
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.imputation_strategy = imputation_strategy
        self.normalization = normalization
        self.feature_engineering = feature_engineering
        self.imputer = SimpleImputer(strategy=imputation_strategy)
        self.scaler = StandardScaler() if normalization == 'standard' else MinMaxScaler()
        self.fitted = False
        self.feature_names = None

    def fit(self, X=None, y=None):
        """
        Fit the ingestion agent on the data.

        Args:
            X (pd.DataFrame or str): Data or path to data file
            y (np.ndarray, optional): Not used
        """
        if isinstance(X, str):
            self.data_path = X
            data = self._load_data()
        elif isinstance(X, pd.DataFrame):
            data = X
        else:
            raise ValueError("X must be a DataFrame or path string")

        # Preprocess data
        processed_data = self._preprocess_data(data)

        # Fit imputer and scaler
        self.imputer.fit(processed_data)
        self.scaler.fit(processed_data)

        self.fitted = True
        return self

    def predict(self, X=None):
        """
        Ingest and preprocess new data.

        Args:
            X (pd.DataFrame, str, or None): New data or path, or None to load from fitted path

        Returns:
            dict: Processed data and metadata
        """
        if not self.fitted:
            raise ValueError("Agent must be fitted before prediction")

        if X is None:
            data = self._load_data()
        elif isinstance(X, str):
            data = self._load_data(X)
        elif isinstance(X, pd.DataFrame):
            data = X
        else:
            raise ValueError("X must be a DataFrame, path string, or None")

        processed_data = self._preprocess_data(data)
        imputed_data = self.imputer.transform(processed_data)
        normalized_data = self.scaler.transform(imputed_data)

        result = {
            'data': normalized_data,
            'original_shape': data.shape,
            'processed_shape': normalized_data.shape,
            'feature_names': self.feature_names
        }

        if self.feature_engineering:
            result['engineered_features'] = self._engineer_features(normalized_data)

        return result

    def _load_data(self, path=None):
        """Load data from file."""
        path = path or self.data_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        file_ext = os.path.splitext(path)[1].lower()
        if file_ext == '.csv':
            data = pd.read_csv(path, index_col=0, parse_dates=True)
        elif file_ext == '.json':
            data = pd.read_json(path)
        elif file_ext in ['.xlsx', '.xls']:
            data = pd.read_excel(path, index_col=0)
        elif file_ext == '.parquet':
            data = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return data

    def _preprocess_data(self, data):
        """Basic preprocessing: handle dates, convert to numeric."""
        # Ensure datetime index if applicable
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                pass  # Keep original index

        # Convert to numeric, handle non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found in data")

        self.feature_names = numeric_data.columns.tolist()
        return numeric_data.values

    def _engineer_features(self, data):
        """Engineer additional time series features."""
        features = {}

        # Rolling statistics
        df = pd.DataFrame(data, columns=self.feature_names)
        for col in self.feature_names:
            features[f'{col}_rolling_mean'] = df[col].rolling(window=10).mean()
            features[f'{col}_rolling_std'] = df[col].rolling(window=10).std()
            features[f'{col}_diff'] = df[col].diff()

        # Combine into array
        engineered = pd.DataFrame(features).fillna(0).values
        return engineered