from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Define dataset path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank_marketing.csv"

# Load local dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop(columns=["subscribed"])
y = df["subscribed"]

# Remove features unavailable before the first campaign contact
leakage_features = [
    "duration",
    "campaign",
    "contact",
    "day_of_week",
    "month"
]

X = X.drop(columns=leakage_features)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

print(f"Training positive rate: {(y_train == 'yes').mean():.2%}")
print(f"Test positive rate: {(y_test == 'yes').mean():.2%}")


# Identify numerical and categorical features
numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)