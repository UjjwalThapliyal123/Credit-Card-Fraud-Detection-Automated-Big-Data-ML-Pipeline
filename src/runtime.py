"""
runtime.py
-----------
Shared inference helpers for the FastAPI backend and Streamlit frontend.

This module loads the saved artifact bundle from outputs/model/ and
reconstructs the model input feature frame from raw transaction values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from explain import build_explainer, get_shap_values, top_k_features
from features import FEATURE_COLUMNS, engineer_features
from model import load_artifacts

DEFAULT_MODEL_DIR = ROOT_DIR / "outputs" / "model"
DEFAULT_THRESHOLD = 0.5
DEFAULT_RISK_TIERS = {
    "LOW": (0.00, 0.35),
    "MEDIUM": (0.35, 0.65),
    "HIGH": (0.65, 1.01),
}


def assign_risk_tier(probability: float, tiers: dict[str, tuple[float, float]] | None = None) -> str:
    tiers = tiers or DEFAULT_RISK_TIERS
    for tier, (lower, upper) in tiers.items():
        if lower <= probability < upper:
            return tier
    return "HIGH"


def load_runtime_bundle(model_dir: str | Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """Load model, threshold config, and runtime metadata from disk."""
    artifacts = load_artifacts(model_dir)

    feature_schema = artifacts.get("feature_schema") or {}
    threshold_config = artifacts.get("threshold_config") or {}

    feature_columns = feature_schema.get("engineered_feature_columns") or list(FEATURE_COLUMNS)
    raw_input_columns = feature_schema.get("raw_input_columns") or [
        "Time",
        "Amount",
        *[f"V{i}" for i in range(1, 29)],
        "Class",
    ]
    inference_input_columns = [column for column in raw_input_columns if column != "Class"]

    return {
        **artifacts,
        "model_dir": Path(model_dir),
        "feature_columns": feature_columns,
        "input_columns": inference_input_columns,
        "threshold": float(threshold_config.get("threshold", DEFAULT_THRESHOLD)),
        "risk_tiers": feature_schema.get("risk_tiers") or [
            {"name": name, "min": bounds[0], "max": bounds[1]}
            for name, bounds in DEFAULT_RISK_TIERS.items()
        ],
    }


def _coerce_input_frame(payload: dict[str, Any] | pd.DataFrame, input_columns: list[str]) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        frame = pd.DataFrame([payload])

    if "Class" in frame.columns:
        frame = frame.drop(columns=["Class"])

    missing_columns = [column for column in input_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")

    frame = frame[input_columns].copy()
    for column in input_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame.isnull().any().any():
        raise ValueError("Input contains non-numeric or missing values.")

    return frame


def prepare_feature_frame(payload: dict[str, Any] | pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Convert raw transaction values into the exact model feature frame."""
    raw_frame = _coerce_input_frame(payload, bundle["input_columns"])
    engineered = engineer_features(raw_frame)

    feature_columns = bundle["feature_columns"]
    for column in feature_columns:
        if column not in engineered.columns:
            engineered[column] = 0.0

    return engineered[feature_columns].fillna(0.0)


def explain_single_prediction(
    bundle: dict[str, Any],
    feature_frame: pd.DataFrame,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Return the top-k SHAP features for one scored row."""
    background = bundle.get("shap_background_sample")
    if background is None or background.empty:
        return []

    feature_columns = bundle["feature_columns"]
    background = background.reindex(columns=feature_columns, fill_value=0.0)
    explainer = build_explainer(bundle["model"], background)
    shap_values = get_shap_values(explainer, feature_frame)
    return top_k_features(shap_values[0], feature_columns, k=k)


def predict_record(
    payload: dict[str, Any],
    bundle: dict[str, Any],
    include_explanations: bool = True,
) -> dict[str, Any]:
    """Score a single transaction and return probability, label, and explanations."""
    feature_frame = prepare_feature_frame(payload, bundle)
    probability = float(bundle["model"].predict_proba(feature_frame)[:, 1][0])
    threshold = bundle["threshold"]

    result = {
        "fraud_probability": round(probability, 6),
        "threshold": round(float(threshold), 6),
        "predicted_fraud": int(probability >= threshold),
        "risk_tier": assign_risk_tier(probability, bundle.get("risk_tiers")),
    }

    if include_explanations:
        result["top_shap_features"] = explain_single_prediction(bundle, feature_frame)

    return result


def predict_batch(
    payload: pd.DataFrame,
    bundle: dict[str, Any],
    include_explanations: bool = False,
) -> pd.DataFrame:
    """Score a batch of transactions."""
    feature_frame = prepare_feature_frame(payload, bundle)
    probabilities = bundle["model"].predict_proba(feature_frame)[:, 1]
    threshold = bundle["threshold"]

    results = payload.copy()
    if "Class" in results.columns:
        results = results.drop(columns=["Class"])

    results["fraud_probability"] = np.round(probabilities, 6)
    results["threshold"] = float(threshold)
    results["predicted_fraud"] = (probabilities >= threshold).astype(int)
    results["risk_tier"] = [assign_risk_tier(probability, bundle.get("risk_tiers")) for probability in probabilities]

    if include_explanations:
        background = bundle.get("shap_background_sample")
        if background is not None and not background.empty:
            feature_columns = bundle["feature_columns"]
            background = background.reindex(columns=feature_columns, fill_value=0.0)
            explainer = build_explainer(bundle["model"], background)
            shap_values = get_shap_values(explainer, feature_frame)
            results["top_shap_features"] = [
                top_k_features(row, feature_columns, k=3) for row in shap_values
            ]

    return results
