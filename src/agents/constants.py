import os

# Project directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Model file extensions
MODEL_EXT = ".joblib"

# Time series constants
DEFAULT_SEQUENCE_LENGTH = 100
DEFAULT_FORECAST_HORIZON = 24
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100

# Data augmentation parameters
AUGMENTATION_NOISE_FACTOR = 0.1
AUGMENTATION_TIME_WARP_FACTOR = 0.05
AUGMENTATION_SCALING_FACTOR = 0.1

# Drift detection thresholds
DRIFT_SIGNIFICANCE_LEVEL = 0.05
DRIFT_WINDOW_SIZE = 1000

# Training parameters
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Uncertainty estimation
NUM_BOOTSTRAP_SAMPLES = 100
CONFIDENCE_LEVEL = 0.95

# File paths
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "default_model.joblib")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "airtemp_data.csv")

# Agent names
AGENT_NAMES = {
    "augmentation": "data_augmentation_agent",
    "ingest": "data_ingest_agent",
    "drift": "drift_detector_agent",
    "tracker": "tracker_agent",
    "trainer": "trainer_agent",
    "uncertainty": "uncertainty_agent"
}

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"