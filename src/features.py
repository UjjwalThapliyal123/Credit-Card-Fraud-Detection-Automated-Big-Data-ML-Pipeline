"""
features.py
-----------
Feature engineering functions for the fraud detection pipeline.

Transformations applied:
  - log1p on Amount  (removes right-skew)
  - Cyclical encoding of Time  (sin/cos of hour-of-day)
  - Top pairwise V-feature interaction products
"""

import numpy as np
import pandas as pd
from itertools import combinations


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def log_transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to Amount column to compress right-skewed distribution."""
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    return df


def cyclical_encode_time(df: pd.DataFrame, seconds_in_day: int = 86400) -> pd.DataFrame:
    """
    Encode Time as cyclical sin/cos features representing the hour-of-day.

    The dataset spans ~48 hours; we fold Time into a 24-hour cycle so that
    23:59 and 00:01 are close together rather than at opposite ends of a
    linear axis.
    """
    df = df.copy()
    time_of_day = df["Time"] % seconds_in_day          # seconds within day
    angle = 2 * np.pi * time_of_day / seconds_in_day
    df["Time_sin"] = np.sin(angle)
    df["Time_cos"] = np.cos(angle)
    return df


def add_v_interactions(
    df: pd.DataFrame,
    top_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Add pairwise product interaction features for selected V-columns.

    Parameters
    ----------
    df : DataFrame with V1-V28 columns present
    top_pairs : list of (col_a, col_b) tuples.  If None, uses the five
                pairs that consistently rank as top SHAP contributors in
                the fraud detection literature.

    Returns
    -------
    DataFrame with new interaction columns appended.
    """
    if top_pairs is None:
        top_pairs = [
            ("V14", "V17"),
            ("V12", "V14"),
            ("V10", "V14"),
            ("V3",  "V7"),
            ("V11", "V4"),
        ]

    df = df.copy()
    for col_a, col_b in top_pairs:
        if col_a in df.columns and col_b in df.columns:
            interaction_name = f"{col_a}_x_{col_b}"
            df[interaction_name] = df[col_a] * df[col_b]

    return df


# ---------------------------------------------------------------------------
# Composite feature engineering step
# ---------------------------------------------------------------------------

INTERACTION_PAIRS = [
    ("V14", "V17"),
    ("V12", "V14"),
    ("V10", "V14"),
    ("V3",  "V7"),
    ("V11", "V4"),
]

FEATURE_COLUMNS = (
    [f"V{i}" for i in range(1, 29)]
    + ["Amount_log", "Time_sin", "Time_cos"]
    + [f"{a}_x_{b}" for a, b in INTERACTION_PAIRS]
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pass:
      1. log-transform Amount
      2. Cyclical-encode Time
      3. Add V-feature interaction products

    Original Amount and Time columns are kept for reference but are not
    included in the model feature set (see FEATURE_COLUMNS).
    """
    df = log_transform_amount(df)
    df = cyclical_encode_time(df)
    df = add_v_interactions(df, top_pairs=INTERACTION_PAIRS)
    return df


# ---------------------------------------------------------------------------
# Derive top interaction pairs from data (used in EDA notebook)
# ---------------------------------------------------------------------------

def compute_top_v_pairs(
    df: pd.DataFrame,
    label_col: str = "Class",
    top_n: int = 5,
) -> list[tuple[str, str]]:
    """
    Identify the top-N V-feature pairs by absolute Pearson correlation
    of their product with the fraud label.

    This is an exploratory utility — the results validate the hard-coded
    INTERACTION_PAIRS above and can be used to discover new ones.
    """
    v_cols = [c for c in df.columns if c.startswith("V")]
    pairs = list(combinations(v_cols, 2))

    correlations = {}
    for col_a, col_b in pairs:
        product = df[col_a] * df[col_b]
        corr = product.corr(df[label_col])
        correlations[(col_a, col_b)] = abs(corr)

    sorted_pairs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    return [pair for pair, _ in sorted_pairs[:top_n]]
