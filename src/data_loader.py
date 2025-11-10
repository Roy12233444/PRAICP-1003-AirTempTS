"""
src/data_loader.py

Loader for PRAICP-1003 AirTempTS using the project's dataset location.

Default CSV:
 E:\AI-Engineering-Capstone-Projects\AirTempTs\PRAICP-1003-AirTempTS\data\surface-air-temperature-monthly-mean.csv

Default results dir:
 E:\AI-Engineering-Capstone-Projects\AirTempTs\PRAICP-1003-AirTempTS\results
"""

from typing import Tuple
import pandas as pd
import os

# Project-specific default paths (use raw strings on Windows)
DEFAULT_CSV = r"E:\AI-Engineering-Capstone-Projects\AirTempTs\PRAICP-1003-AirTempTS\data\surface-air-temperature-monthly-mean.csv"
DEFAULT_RESULTS_DIR = r"E:\AI-Engineering-Capstone-Projects\AirTempTs\PRAICP-1003-AirTempTS\results"
DEFAULT_FIGURES_DIR = os.path.join(DEFAULT_RESULTS_DIR, "figures")


def load_airtemp(csv_path: str = DEFAULT_CSV,
                 parse_dates: bool = True,
                 date_col: str = "month",
                 value_col: str = "mean_temp") -> pd.DataFrame:
    """
    Load the air temperature CSV and return a cleaned DataFrame.

    Returns a DataFrame with columns:
        - month (datetime)
        - mean_temp (float)
    """
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {date_col, value_col}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}. Found: {list(df.columns)}")

    if parse_dates:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce")

    # Drop rows where date or value couldn't be parsed
    df = df.dropna(subset=[date_col, value_col]).copy()

    # Ensure numeric
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).copy()

    df = df.sort_values(by=date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col].dt.to_period("M").dt.to_timestamp())

    df = df.rename(columns={date_col: "month", value_col: "mean_temp"})
    return df


def train_test_split_by_time(df: pd.DataFrame,
                             test_periods: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test by time (last `test_periods` rows are test).
    """
    if "month" not in df.columns:
        raise ValueError("DataFrame must contain 'month' column")
    if test_periods <= 0:
        raise ValueError("test_periods must be > 0")
    if len(df) <= test_periods:
        raise ValueError(f"DataFrame length ({len(df)}) must be > test_periods ({test_periods})")

    train = df.iloc[:-test_periods].reset_index(drop=True)
    test = df.iloc[-test_periods:].reset_index(drop=True)
    return train, test


def ensure_results_dirs(results_dir: str = DEFAULT_RESULTS_DIR,
                        figures_dir: str = DEFAULT_FIGURES_DIR):
    """
    Ensure results and figures directories exist.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return os.path.abspath(results_dir), os.path.abspath(figures_dir)


if __name__ == "__main__":
    # quick local test
    try:
        df = load_airtemp()
        print(f"Loaded {len(df)} rows from {df['month'].min().date()} to {df['month'].max().date()}")
        tr, te = train_test_split_by_time(df, test_periods=12)
        print(f"Train: {len(tr)} rows; Test: {len(te)} rows")
        rd, fd = ensure_results_dirs()
        print("Results dir:", rd)
        print("Figures dir:", fd)
    except Exception as e:
        print("Quick test failed:", e)
