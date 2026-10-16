
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"


def predict_client(client, model, threshold):
    """Predict whether a new client will subscribe."""

    # Convert client data into a DataFrame
    client_df = pd.DataFrame([client])

    # Get probability of the positive class
    positive_index = list(model.classes_).index("yes")

    probability = model.predict_proba(client_df)[
        0, positive_index
    ]

    # Apply the selected classification threshold
    prediction = (
        "yes" if probability >= threshold else "no"
    )

    return prediction, probability


def main():
    # Load the trained model and threshold
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    threshold = artifact["threshold"]

    # Example of a new client
    client = {
        "age": 35,
        "job": "technician",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "pdays": float("nan"),
        "previous": 0,
        "poutcome": "unknown"
    }

    # Generate prediction
    prediction, probability = predict_client(
        client, model, threshold
    )

    print("\nClient Prediction:")
    print(f"Estimated probability: {probability:.2%}")
    print(f"Classification threshold: {threshold:.4f}")
    print(f"Predicted subscription: {prediction}")


if __name__ == "__main__":
    main()
