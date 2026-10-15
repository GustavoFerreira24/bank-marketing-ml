from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank_marketing.csv"

LEAKAGE_FEATURES = [
    "duration",
    "campaign",
    "contact",
    "day_of_week",
    "month"
]


def load_data():
    """Load the local dataset and prepare features and target."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run src/download_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["subscribed", *LEAKAGE_FEATURES]).copy()
    y = df["subscribed"]

    # -1 indicates no previous contact
    X["pdays"] = X["pdays"].replace(-1, float("nan"))

    return X, y


def split_data(X, y):
    """Create stratified training and test sets."""

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
