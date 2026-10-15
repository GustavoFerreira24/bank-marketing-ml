from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

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

# Treat -1 as no previous contact
X["pdays"] = X["pdays"].replace(-1, float("nan"))

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


# Numerical preprocessing pipeline
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler())
])

# Categorical preprocessing pipeline
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Apply different transformations to each feature type
preprocessor = ColumnTransformer(transformers=[
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nPreprocessing pipeline created successfully!")




# Fit preprocessing on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations to test data
X_test_processed = preprocessor.transform(X_test)

print("\nBefore preprocessing:")
print("Training:", X_train.shape)
print("Test:", X_test.shape)

print("\nAfter preprocessing:")
print("Training:", X_train_processed.shape)
print("Test:", X_test_processed.shape)

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

print("\nProcessed features:")
for feature in feature_names:
    print(feature)



# Complete machine learning pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

print("\nLogistic Regression trained successfully!")






# Baseline model: always predict the majority class
baseline = DummyClassifier(strategy="most_frequent")

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate baseline using training data only
baseline_scores = cross_validate(
    baseline,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "recall": make_scorer(recall_score, pos_label="yes"),
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(
            average_precision_score,
            response_method="predict_proba",
            pos_label="yes"
        )
    }

)

print("\nBaseline Results:")
print(f"Accuracy: {baseline_scores['test_accuracy'].mean():.2%}")
print(f"Recall: {baseline_scores['test_recall'].mean():.2%}")
print(f"ROC-AUC: {baseline_scores['test_roc_auc'].mean():.4f}")
print(f"PR-AUC: {baseline_scores['test_pr_auc'].mean():.4f}")




# Evaluate Logistic Regression using cross-validation
logreg_scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score, pos_label="yes", zero_division=0
        ),
        "recall": make_scorer(
            recall_score, pos_label="yes"
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
)

print("\nLogistic Regression Results:")
print(f"Accuracy: {logreg_scores['test_accuracy'].mean():.2%}")
print(f"Precision: {logreg_scores['test_precision'].mean():.2%}")
print(f"Recall: {logreg_scores['test_recall'].mean():.2%}")
print(f"F1-score: {logreg_scores['test_f1'].mean():.2%}")
print(f"ROC-AUC: {logreg_scores['test_roc_auc'].mean():.4f}")
print(f"PR-AUC: {logreg_scores['test_pr_auc'].mean():.4f}")



# Obtain out-of-fold probabilities using cross-validation
oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]  # Probability of class "yes"

# Compare different classification thresholds
thresholds = [0.2, 0.3, 0.4, 0.5]

results = []

for threshold in thresholds:
    predictions = [
        "yes" if prob >= threshold else "no"
        for prob in oof_probabilities
    ]

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_train, predictions, pos_label="yes", zero_division=0
        ),
        "Recall": recall_score(
            y_train, predictions, pos_label="yes"
        ),
        "F1": f1_score(
            y_train, predictions, pos_label="yes", zero_division=0
        )
    })

threshold_results = pd.DataFrame(results)

print("\nThreshold Comparison:")
print(threshold_results.round(3).to_string(index=False))


# Random Forest pipeline
random_forest = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

# Evaluate Random Forest using cross-validation
rf_scores = cross_validate(
    random_forest,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(
            average_precision_score,
            response_method="predict_proba",
            pos_label="yes"
        )
    }
)

print("\nRandom Forest Results:")
print(f"ROC-AUC: {rf_scores['test_roc_auc'].mean():.4f}")
print(f"PR-AUC: {rf_scores['test_pr_auc'].mean():.4f}")


# Obtain out-of-fold probabilities for Random Forest
rf_oof_probabilities = cross_val_predict(
    random_forest,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]  # Probability of class "yes"

# Compare classification thresholds
rf_threshold_results = []

for threshold in [0.2, 0.3, 0.4, 0.5]:
    predictions = [
        "yes" if prob >= threshold else "no"
        for prob in rf_oof_probabilities
    ]

    rf_threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_train, predictions, pos_label="yes", zero_division=0
        ),
        "Recall": recall_score(
            y_train, predictions, pos_label="yes"
        ),
        "F1": f1_score(
            y_train, predictions, pos_label="yes", zero_division=0
        )
    })

rf_threshold_df = pd.DataFrame(rf_threshold_results)

print("\nRandom Forest Threshold Comparison:")
print(rf_threshold_df.round(3).to_string(index=False))



# Calculate Precision and Recall across thresholds
precision, recall, thresholds = precision_recall_curve(
    y_train,
    rf_oof_probabilities,
    pos_label="yes"
)

# Calculate F1-score for each threshold
f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

# Find the threshold with the highest F1-score
best_index = f1_scores.argmax()
best_threshold = thresholds[best_index]

print("\nBest Random Forest Threshold:")
print(f"Threshold: {best_threshold:.4f}")
print(f"Precision: {precision[best_index]:.2%}")
print(f"Recall: {recall[best_index]:.2%}")
print(f"F1-score: {f1_scores[best_index]:.2%}")