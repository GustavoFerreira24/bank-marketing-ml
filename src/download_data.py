
from pathlib import Path

from ucimlrepo import fetch_ucirepo
import pandas as pd


# Define the local dataset path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank_marketing.csv"


def download_data():
    # Avoid downloading the dataset if it already exists
    if DATA_PATH.exists():
        print(f"Dataset already exists at: {DATA_PATH}")
        return

    print("Downloading Bank Marketing dataset...")

    dataset = fetch_ucirepo(id=222)

    X = dataset.data.features
    y = dataset.data.targets

    # Combine features and target into a single DataFrame
    df = X.copy()
    df["subscribed"] = y.iloc[:, 0]

    # Save the dataset locally
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    print(f"Dataset saved to: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")


if __name__ == "__main__":
    download_data()
