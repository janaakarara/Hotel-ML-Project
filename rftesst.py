from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import ARTIFACTS_DIR, DATA_FILE, RANDOM_STATE, TARGET_CLASSIFICATION


EXCLUDED_FEATURES = ["ReservationStatus", "ReservationStatusDate"]
BINARY_FEATURE = "PreviousBookingsNotCanceled"
TEST_SIZE = 0.2


def main() -> None:
    data = pd.read_csv(DATA_FILE)
    required_columns = {TARGET_CLASSIFICATION, BINARY_FEATURE, *EXCLUDED_FEATURES}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    features = data.drop(columns=[TARGET_CLASSIFICATION, *EXCLUDED_FEATURES])
    features[BINARY_FEATURE] = (features[BINARY_FEATURE] > 0).astype("int8")
    target = data[TARGET_CLASSIFICATION]
    if target.isna().any():
        raise ValueError(f"Target column {TARGET_CLASSIFICATION!r} contains missing values")

    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=1000,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Training rows: {len(X_train):,}; test rows: {len(X_test):,}")
    print(f"Predictor features: {len(features.columns)}")
    print(f"{BINARY_FEATURE}: converted to 1 when > 0, otherwise 0")
    print(classification_report(y_test, predictions, zero_division=0))

    fitted_preprocessor = model.named_steps["preprocessor"]
    encoded_importances = model.named_steps["classifier"].feature_importances_
    feature_importances: dict[str, float] = {}

    numeric_slice = fitted_preprocessor.output_indices_["numeric"]
    for name, importance in zip(
        numeric_features, encoded_importances[numeric_slice], strict=True
    ):
        feature_importances[name] = float(importance)

    categorical_slice = fitted_preprocessor.output_indices_["categorical"]
    categorical_importances = encoded_importances[categorical_slice]
    encoder = (
        fitted_preprocessor.named_transformers_["categorical"]
        .named_steps["onehot"]
    )
    offset = 0
    for name, categories in zip(
        categorical_features, encoder.categories_, strict=True
    ):
        width = len(categories)
        feature_importances[name] = float(
            categorical_importances[offset : offset + width].sum()
        )
        offset += width

    importance_table = (
        pd.DataFrame(
            {
                "feature": list(feature_importances),
                "importance": list(feature_importances.values()),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    output_dir = Path(ARTIFACTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = (
        output_dir
        / "random_forest_cancellation_binary_previous_bookings.joblib"
    )
    importance_path = (
        output_dir
        / "random_forest_feature_importance_binary_previous_bookings.csv"
    )
    plot_path = (
        output_dir
        / "random_forest_feature_importance_binary_previous_bookings.png"
    )

    joblib.dump(model, model_path)
    importance_table.to_csv(importance_path, index=False)

    plot_data = importance_table.sort_values("importance", ascending=True)
    figure, axis = plt.subplots(
        figsize=(12, max(8, 0.32 * len(plot_data)))
    )
    axis.barh(plot_data["feature"], plot_data["importance"], color="steelblue")
    axis.set_title(
        "Random Forest Feature Importance "
        "(PreviousBookingsNotCanceled Binarized)"
    )
    axis.set_xlabel("Mean decrease in impurity")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", linestyle="--", alpha=0.4)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Model saved to: {model_path}")
    print(f"Feature importances saved to: {importance_path}")
    print(f"Feature importance graph saved to: {plot_path}")


if __name__ == "__main__":
    main()