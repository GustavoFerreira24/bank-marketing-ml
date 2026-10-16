import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)


# Evaluation metrics
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score, pos_label="yes", zero_division=0
    ),
    "recall": make_scorer(
        recall_score, pos_label="yes", zero_division=0
    ),
    "f1": make_scorer(
        f1_score, pos_label="yes", zero_division=0
    ),
    "roc_auc": "roc_auc",
    "pr_auc": make_scorer(
        average_precision_score,
        response_method="predict_proba",
        pos_label="yes"
    )
}


def evaluate_model(name, model, X_train, y_train, cv):
    """Evaluate a model using stratified cross-validation."""

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING
    )

    results = {
        metric: scores[f"test_{metric}"].mean()
        for metric in SCORING
    }

    print(f"\n{name} Results:")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall: {results['recall']:.2%}")
    print(f"F1-score: {results['f1']:.2%}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"PR-AUC: {results['pr_auc']:.4f}")

    return results


def get_oof_probabilities(model, X_train, y_train, cv):
    """Generate out-of-fold probabilities for the positive class."""

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba"
    )

    # scikit-learn orders the classes alphabetically: no, yes
    positive_class_index = sorted(y_train.unique()).index("yes")

    return probabilities[:, positive_class_index]


def compare_thresholds(
    y_true,
    probabilities,
    thresholds=(0.2, 0.3, 0.4, 0.5)
):
    """Compare classification metrics across different thresholds."""

    results = []

    for threshold in thresholds:
        predictions = np.where(
            probabilities >= threshold,
            "yes",
            "no"
        )

        results.append({
            "Threshold": threshold,
            "Precision": precision_score(
                y_true, predictions, pos_label="yes", zero_division=0
            ),
            "Recall": recall_score(
                y_true, predictions, pos_label="yes", zero_division=0
            ),
            "F1": f1_score(
                y_true, predictions, pos_label="yes", zero_division=0
            )
        })

    return pd.DataFrame(results)


def find_best_threshold(y_true, probabilities):
    """Find the threshold that maximizes the F1-score."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
        pos_label="yes"
    )

    # Calculate F1 for thresholds with corresponding P/R values
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = f1_scores.argmax()

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_scores[best_index])
    }




def evaluate_test_set(model, X_test, y_test, threshold):
    """Evaluate the final model on the held-out test set."""

    # Obtain probabilities for the positive class
    positive_index = list(model.classes_).index("yes")

    probabilities = model.predict_proba(X_test)[
        :, positive_index
    ]

    # Convert probabilities into class predictions
    predictions = np.where(
        probabilities >= threshold,
        "yes",
        "no"
    )

    # Binary target for probability-based metrics
    y_binary = (y_test == "yes").astype(int)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, pos_label="yes", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, pos_label="yes", zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, pos_label="yes", zero_division=0
        ),
        "roc_auc": roc_auc_score(y_binary, probabilities),
        "pr_auc": average_precision_score(
            y_binary, probabilities
        )
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=["no", "yes"]
    ).ravel()

    print("\nFinal Test Set Results:")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall: {results['recall']:.2%}")
    print(f"F1-score: {results['f1']:.2%}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"PR-AUC: {results['pr_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")

    return results
