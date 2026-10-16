from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score
)

from data_loader import load_data, split_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    # 1. Load the saved model and threshold
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    threshold = artifact["threshold"]

    # 2. Recreate the original test split
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)

    # 3. Generate predictions
    positive_index = list(model.classes_).index("yes")

    probabilities = model.predict_proba(X_test)[
        :, positive_index
    ]

    predictions = np.where(
        probabilities >= threshold,
        "yes",
        "no"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=["no", "yes"],
        display_labels=["No", "Yes"],
        values_format="d",
        colorbar=False,
        ax=ax
    )

    ax.set_title("Random Forest - Confusion Matrix")
    fig.tight_layout()

    confusion_path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=180)
    plt.close(fig)

    # 5. Plot Precision-Recall curve
    fig, ax = plt.subplots(figsize=(7, 5))

    y_binary = (y_test == "yes").astype(int)

    PrecisionRecallDisplay.from_predictions(
        y_binary,
        probabilities,
        name="Random Forest",
        ax=ax
    )

    # Baseline: proportion of positive examples
    positive_rate = y_binary.mean()

    ax.axhline(
        positive_rate,
        linestyle="--",
        label=f"Random baseline ({positive_rate:.1%})"
    )

    # Mark the selected threshold on the curve
    selected_precision = precision_score(
        y_test, predictions, pos_label="yes"
    )

    selected_recall = recall_score(
        y_test, predictions, pos_label="yes"
    )

    ax.scatter(
        selected_recall,
        selected_precision,
        s=70,
        zorder=5,
        label=f"Selected threshold ({threshold:.3f})"
    )

    ax.set_title("Random Forest - Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()

    pr_curve_path = FIGURES_DIR / "precision_recall_curve.png"
    fig.savefig(pr_curve_path, dpi=180)
    plt.close(fig)

    print("Evaluation plots saved successfully!")
    print(f"Confusion matrix: {confusion_path}")
    print(f"Precision-Recall curve: {pr_curve_path}")


if __name__ == "__main__":
    main()
