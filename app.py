from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from runtime import load_runtime_bundle, predict_batch, predict_record  # noqa: E402


@st.cache_resource
def get_bundle() -> dict:
    return load_runtime_bundle()


def format_metric(value: float) -> str:
    return f"{value:.4f}"


st.set_page_config(
    page_title="Fraud Risk Studio",
    page_icon=None,
    layout="wide",
)

bundle = get_bundle()
threshold = bundle["threshold"]
feature_schema = bundle.get("feature_schema") or {}
input_columns = bundle["input_columns"]

st.markdown(
    """
    <style>
    .app-shell {
        background: linear-gradient(180deg, #f8f4ef 0%, #fffdf8 100%);
    }
    .card {
        background: white;
        border: 1px solid rgba(25, 25, 25, 0.08);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 12px 30px rgba(18, 18, 18, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fraud Risk Studio")
st.caption("Streamlit app backed by the saved XGBoost model artifacts in outputs/model/.")

left, right, far_right = st.columns(3)
left.metric("Decision threshold", f"{threshold:.2f}")
right.metric("Model features", len(bundle["feature_columns"]))
far_right.metric("Risk tiers", len(bundle["risk_tiers"]))

st.sidebar.header("Bundle")
st.sidebar.write(f"Model directory: {bundle['model_dir']}")
st.sidebar.write(f"Saved threshold: {threshold:.4f}")
st.sidebar.write(f"Feature count: {len(bundle['feature_columns'])}")

tab_single, tab_batch, tab_info = st.tabs(["Single transaction", "Batch CSV", "Model info"])

with tab_single:
    st.markdown("### Score one transaction")
    with st.form("single_transaction_form"):
        columns = st.columns(2)
        values: dict[str, float] = {}
        for index, column in enumerate(input_columns):
            with columns[index % 2]:
                if column in {"Time", "Amount"}:
                    values[column] = st.number_input(column, value=0.0, format="%.6f" if column == "Time" else "%.2f")
                else:
                    values[column] = st.number_input(column, value=0.0, format="%.6f")

        submitted = st.form_submit_button("Run prediction")

    if submitted:
        result = predict_record(values, bundle, include_explanations=True)

        score_col, tier_col = st.columns([2, 1])
        with score_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Prediction")
            st.metric("Fraud probability", format_metric(result["fraud_probability"]))
            st.metric("Predicted fraud", "Yes" if result["predicted_fraud"] else "No")
            st.metric("Risk tier", result["risk_tier"])
            st.markdown("</div>", unsafe_allow_html=True)

        with tier_col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Top SHAP drivers")
            shap_rows = result.get("top_shap_features", [])
            if shap_rows:
                st.dataframe(pd.DataFrame(shap_rows), use_container_width=True, hide_index=True)
            else:
                st.info("SHAP background sample not available.")
            st.markdown("</div>", unsafe_allow_html=True)

with tab_batch:
    st.markdown("### Score a CSV upload")
    uploaded = st.file_uploader("Upload a CSV with Time, Amount, and V1-V28", type=["csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        try:
            scored = predict_batch(frame, bundle, include_explanations=False)
            st.success(f"Scored {len(scored):,} rows")
            st.dataframe(scored.head(50), use_container_width=True)
            st.download_button(
                "Download scored CSV",
                scored.to_csv(index=False).encode("utf-8"),
                file_name="scored_transactions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(str(exc))

with tab_info:
    st.markdown("### Saved artifacts")
    st.write(bundle.get("artifact_manifest") or {"model": "xgboost_fraud_model.json"})
    st.write(bundle.get("threshold_config"))
    st.write(bundle.get("training_metadata"))
    st.write(feature_schema)
