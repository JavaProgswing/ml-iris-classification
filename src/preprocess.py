"""Load, clean, and preprocess the iris dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Iris.csv"
TARGET_COLUMN = "Species"

FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
EXPECTED_COLUMNS = FEATURES + [TARGET_COLUMN]


def load_clean_data(
    data_path: Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, int]:
    """Load the raw CSV and remove exact duplicate rows."""
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find the dataset at {data_path}")

    data = pd.read_csv(data_path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    duplicate_count = int(data.duplicated().sum())
    clean_data = data.drop_duplicates().reset_index(drop=True)
    return clean_data, duplicate_count


def split_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the flower measurements X and species target y."""
    X = data[FEATURES].copy()
    y = data[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> StandardScaler:
    """Standardize the four measurement columns."""
    return StandardScaler()
