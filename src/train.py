from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from data_loader import load_data, split_data
from preprocessing import build_preprocessor
from evaluation import (
    evaluate_model,
    get_oof_probabilities,
    compare_thresholds,
    find_best_threshold
)


def main():
    # 1. Load and split dataset
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)

    print(f"Training positive rate: {(y_train == 'yes').mean():.2%}")
    print(f"Test positive rate: {(y_test == 'yes').mean():.2%}")

    # 2. Configure cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # 3. Define models
    models = {
        "Baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),

        "Random Forest": Pipeline(steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    # 4. Evaluate all models
    model_results = {}

    for name, model in models.items():
        model_results[name] = evaluate_model(
            name, model, X_train, y_train, cv
        )

    # 5. Compare classification thresholds
    for name in ["Logistic Regression", "Random Forest"]:
        model = models[name]

        probabilities = get_oof_probabilities(
            model, X_train, y_train, cv
        )

        threshold_results = compare_thresholds(
            y_train, probabilities
        )

        print(f"\n{name} Threshold Comparison:")
        print(threshold_results.round(3).to_string(index=False))

        # 6. Optimize the Random Forest threshold
        if name == "Random Forest":
            best = find_best_threshold(y_train, probabilities)

            print("\nBest Random Forest Threshold:")
            print(f"Threshold: {best['threshold']:.4f}")
            print(f"Precision: {best['precision']:.2%}")
            print(f"Recall: {best['recall']:.2%}")
            print(f"F1-score: {best['f1']:.2%}")


if __name__ == "__main__":
    main()
