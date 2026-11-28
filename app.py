from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from config import TARGET_CLASSIFICATION
from pipeline.inference import predict_uploaded_data


ROOT = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "Random Forest (optimized)": {
        "model": ROOT
        / "artifacts"
        / "classification"
        / "Randomforrestoptimized"
        / "random_forest_classifier_optimized_0.3.joblib",
        "metrics": ROOT
        / "artifacts"
        / "classification"
        / "Randomforrestoptimized"
        / "random_forest_classifier_optimized_0.3_metrics.json",
    },
    "MLP": {
        "model": ROOT
        / "artifacts"
        / "classification"
        / "MLP"
        / "MLP_1.1.joblib",
        "metrics": ROOT
        / "artifacts"
        / "classification"
        / "MLP"
        / "MLP_1.1_metrics.json",
    },
}


@st.cache_resource
def load_model(model_path: str) -> Any:
    return joblib.load(model_path)


@st.cache_data
def load_metrics(metrics_path: str) -> dict[str, Any]:
    return json.loads(Path(metrics_path).read_text(encoding="utf-8"))


def show_metrics(metrics: dict[str, Any], title: str) -> None:
    st.subheader(title)
    columns = st.columns(5)
    for column, (key, label) in zip(
        columns,
        (
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1 score"),
            ("roc_auc", "ROC AUC"),
        ),
    ):
        value = metrics.get(key)
        column.metric(label, f"{value:.1%}" if value is not None else "N/A")

    if "pr_auc" in metrics:
        st.metric("PR AUC", f"{metrics['pr_auc']:.1%}")

    if "confusion_matrix" in metrics:
        st.caption("Confusion matrix (rows: actual, columns: predicted)")
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=["Not canceled", "Canceled"],
                columns=["Predicted not canceled", "Predicted canceled"],
            ),
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Hotel Cancellation Predictor",
        page_icon="🏨",
        layout="wide",
    )
    st.title("Hotel Cancellation Predictor")
    st.write(
        "Upload a hotel bookings CSV, choose a trained model, and predict "
        "which bookings are likely to be canceled."
    )

    model_name = st.selectbox("Choose a model", list(MODEL_OPTIONS))
    uploaded_file = st.file_uploader("Upload booking data (CSV)", type=["csv"])

    if uploaded_file is None:
        st.info(
            "Upload a CSV with the booking columns used by the project. "
            f"The {TARGET_CLASSIFICATION} column is optional."
        )
        return

    csv_bytes = uploaded_file.getvalue()
    result_key = (
        hashlib.sha256(csv_bytes).hexdigest(),
        model_name,
    )
    if st.session_state.get("result_key") != result_key:
        st.session_state["result_key"] = result_key
        st.session_state.pop("prediction_result", None)
        st.session_state["screen"] = "predictions"

    selected_paths = MODEL_OPTIONS[model_name]
    model_path = selected_paths["model"]
    metrics_path = selected_paths["metrics"]
    missing_artifacts = [
        path for path in (model_path, metrics_path) if not path.is_file()
    ]
    if missing_artifacts:
        st.error(
            "Required model artifacts are missing: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing_artifacts)
        )
        return

    try:
        raw = pd.read_csv(io.BytesIO(csv_bytes))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        return

    with st.expander("Preview uploaded data"):
        st.dataframe(raw.head(10), use_container_width=True)

    if st.button("Predict cancellations", type="primary"):
        with st.spinner("Preprocessing the data and generating predictions..."):
            try:
                model = load_model(str(model_path))
                model_metrics = load_metrics(str(metrics_path))
                threshold = model_metrics.get("decision_threshold", {}).get(
                    "value", 0.5
                )
                predictions, uploaded_metrics, summary = predict_uploaded_data(
                    raw, model, threshold
                )
            except (TypeError, ValueError, KeyError) as exc:
                st.error(f"Prediction could not be completed: {exc}")
                return

        st.session_state["prediction_result"] = {
            "model_name": model_name,
            "threshold": threshold,
            "predictions": predictions,
            "uploaded_metrics": uploaded_metrics,
            "summary": summary,
            "model_metrics": model_metrics,
        }
        st.session_state["screen"] = "predictions"

    result = st.session_state.get("prediction_result")
    if result is None:
        return

    if st.session_state.get("screen") == "metrics":
        st.header(f"{result['model_name']} metrics")
        if st.button("Back to predictions"):
            st.session_state["screen"] = "predictions"
            st.rerun()

        saved_metrics = result["model_metrics"]
        st.caption(
            "The following scores are from this model's held-out test set."
        )
        show_metrics(saved_metrics["test_set_performance"], "Held-out test set")

        if result["uploaded_metrics"] is not None:
            show_metrics(
                result["uploaded_metrics"],
                "Uploaded data (measured against its IsCanceled labels)",
            )
        else:
            st.info(
                f"To measure performance on this upload, include a valid "
                f"{TARGET_CLASSIFICATION} column."
            )

        cross_validation = saved_metrics.get("cross_validation")
        if cross_validation and cross_validation.get("metrics"):
            with st.expander("Cross-validation scores"):
                st.json(cross_validation)
        return

    predictions = result["predictions"]
    summary = result["summary"]
    st.subheader("Prediction results")
    st.caption(
        f"{summary['output']['rows']:,} rows predicted; "
        f"{summary['undefined_distribution_channel_rows_removed']:,} rows "
        "with an undefined distribution channel were excluded by preprocessing. "
        f"Decision threshold: {result['threshold']:.3f}."
    )
    st.dataframe(predictions, use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="hotel_cancellation_predictions.csv",
        mime="text/csv",
    )
    if st.button("See the metrics of this model"):
        st.session_state["screen"] = "metrics"
        st.rerun()


if __name__ == "__main__":
    main()
