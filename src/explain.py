"""
explain.py
----------
SHAP-based explanation utilities for the fraud detection pipeline.

Why SHAP?
  A fraud analyst receiving a flag needs to know *why* a transaction was
  flagged — not just a probability score. SHAP values provide a consistent,
  game-theoretic attribution of each feature's contribution to the prediction.
  Without this, the model is a black box that no operations team will trust.

Outputs:
  - Per-transaction top-3 feature explanations (written to scored CSV)
  - Global SHAP summary plot (bar + beeswarm)
  - Individual SHAP waterfall plot for a single transaction
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# SHAP explainer setup
# ---------------------------------------------------------------------------

def build_explainer(model: XGBClassifier, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for an XGBoost model.

    TreeExplainer is exact (not approximate) for tree-based models and
    runs in O(TLD) time where T=trees, L=leaves, D=depth.

    Parameters
    ----------
    X_background : a representative sample of the training data used to
                   compute expected (base) SHAP values.  100–500 rows is
                   usually sufficient; too many just slows computation.
    """
    return shap.TreeExplainer(model, data=X_background, feature_perturbation="interventional")


# ---------------------------------------------------------------------------
# Per-transaction explanations
# ---------------------------------------------------------------------------

def get_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Compute SHAP values for every row in X.

    Returns
    -------
    2-D array of shape (n_samples, n_features).
    Positive values push toward fraud (class 1); negative toward legit.
    """
    sv = explainer.shap_values(X)
    # TreeExplainer may return a list [class0_vals, class1_vals]; take class 1
    if isinstance(sv, list):
        return sv[1]
    return sv


def top_k_features(
    shap_row: np.ndarray,
    feature_names: list[str],
    k: int = 3,
) -> list[dict]:
    """
    Return the top-k features by absolute SHAP magnitude for a single row.

    Returns
    -------
    List of dicts: [{"feature": name, "shap": value, "direction": "+"/"-"}, ...]
    sorted by descending |SHAP|.
    """
    indices = np.argsort(np.abs(shap_row))[::-1][:k]
    return [
        {
            "feature":   feature_names[i],
            "shap":      round(float(shap_row[i]), 4),
            "direction": "+" if shap_row[i] > 0 else "-",
        }
        for i in indices
    ]


def add_shap_explanations(
    df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    k: int = 3,
) -> pd.DataFrame:
    """
    Append top-k SHAP explanation columns to a scored DataFrame.

    Adds columns: shap_feature_1, shap_value_1, shap_feature_2, ...
    """
    df = df.copy()
    for rank in range(1, k + 1):
        df[f"shap_feature_{rank}"] = ""
        df[f"shap_value_{rank}"]   = np.nan

    for i, row_shap in enumerate(shap_values):
        top = top_k_features(row_shap, feature_names, k=k)
        for rank, entry in enumerate(top, start=1):
            df.at[df.index[i], f"shap_feature_{rank}"] = entry["feature"]
            df.at[df.index[i], f"shap_value_{rank}"]   = entry["shap"]

    return df


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    plot_type: str = "bar",
    max_display: int = 20,
) -> None:
    """
    Global SHAP summary plot.

    plot_type = 'bar'     → mean absolute SHAP (feature importance)
    plot_type = 'beeswarm'→ distribution of SHAP values per feature
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))

    if plot_type == "bar":
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
        plt.title("Mean |SHAP| — Global Feature Importance", fontsize=13)
    else:
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        plt.title("SHAP Value Distribution (beeswarm)", fontsize=13)

    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"  SHAP summary plot saved → {output_path}")
    else:
        plt.show()

    plt.close()


def plot_shap_waterfall(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "SHAP Waterfall — Single Transaction",
) -> None:
    """
    Waterfall plot for a single transaction showing how each feature
    pushes the model output from the base value to the final score.

    X_row must be a single-row DataFrame (the transaction to explain).
    """
    import matplotlib.pyplot as plt

    explanation = explainer(X_row)
    # Take class-1 explanation if multi-output
    if explanation.values.ndim == 3:
        shap_explanation = shap.Explanation(
            values=explanation.values[0, :, 1],
            base_values=explanation.base_values[0, 1],
            data=explanation.data[0],
            feature_names=X_row.columns.tolist(),
        )
    else:
        shap_explanation = shap.Explanation(
            values=explanation.values[0],
            base_values=explanation.base_values[0] if np.ndim(explanation.base_values) > 0
                        else explanation.base_values,
            data=explanation.data[0],
            feature_names=X_row.columns.tolist(),
        )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation, show=False)
    plt.title(title, fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"  Waterfall plot saved → {output_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# Convenience wrapper for pipeline
# ---------------------------------------------------------------------------

def explain_predictions(
    model: XGBClassifier,
    X_train_sample: pd.DataFrame,
    X_score: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
    generate_plots: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Full explanation pass:
      1. Build SHAP explainer on training sample
      2. Compute SHAP values for X_score
      3. Optionally save global summary + waterfall for first fraud example
      4. Return (shap_values, X_score_with_explanations)

    Parameters
    ----------
    X_train_sample : 200–500 row sample of training data for SHAP background
    X_score        : full set of transactions to explain
    output_dir     : if provided, save plots here
    """
    print("  Building SHAP explainer…")
    explainer = build_explainer(model, X_train_sample)

    print(f"  Computing SHAP values for {len(X_score):,} transactions…")
    shap_values = get_shap_values(explainer, X_score)

    X_with_shap = add_shap_explanations(X_score, shap_values, X_score.columns.tolist())

    if generate_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_shap_summary(
            shap_values, X_score,
            output_path=output_dir / "shap_summary_bar.png",
            plot_type="bar",
        )
        plot_shap_summary(
            shap_values, X_score,
            output_path=output_dir / "shap_summary_beeswarm.png",
            plot_type="beeswarm",
        )

    return shap_values, X_with_shap
