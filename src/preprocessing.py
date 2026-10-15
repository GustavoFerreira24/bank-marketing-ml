from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(X_train):
    """Build preprocessing pipelines for numerical and categorical features."""

    # Identify feature types
    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="unknown"
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor
