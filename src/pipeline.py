"""
pipeline.py
-----------
End-to-end automated fraud scoring pipeline.

Stages
------
1. Data ingestion & schema validation
2. PySpark preprocessing (scaling, null audit)
3. Feature engineering (log Amount, cyclical Time, V interactions)
4. Class imbalance handling (SMOTE on train fold)
5. XGBoost model training with early stopping
6. Threshold optimisation (F2-score + business cost matrix)
7. Automated scoring output with per-transaction SHAP explanations

Usage
-----
    python src/pipeline.py --data data/creditcard.csv --output outputs/

Zero manual intervention from raw CSV to scored output.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Add src/ to path (for notebook-style imports)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from features import engineer_features, FEATURE_COLUMNS, INTERACTION_PAIRS
from model import (
    ModelConfig,
    train,
    evaluate,
    save_model,
    sweep_thresholds,
    select_threshold,
)
from explain import explain_predictions

# ---------------------------------------------------------------------------
# Stage 1 — Data ingestion
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
EXPECTED_ROWS_MIN = 280_000


def load_and_validate(data_path: str | Path) -> pd.DataFrame:
    """
    Load raw CSV, assert schema, fail loudly on drift.
    No silent errors — a missing column or wrong dtype surfaces immediately.
    """
    print("\n[Stage 1] Data ingestion & schema validation")
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place creditcard.csv in the data/ directory."
        )

    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Column presence check
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema drift — missing columns: {missing_cols}")

    # Row count sanity check
    if len(df) < EXPECTED_ROWS_MIN:
        raise ValueError(
            f"Expected ≥{EXPECTED_ROWS_MIN:,} rows, got {len(df):,}. "
            "Dataset may be truncated."
        )

    # Fraud rate sanity check
    fraud_rate = df["Class"].mean()
    print(f"  Fraud rate: {fraud_rate:.4%}  ({df['Class'].sum():,} fraud / {len(df):,} total)")
    if not (0.001 <= fraud_rate <= 0.01):
        print(f"  ⚠ Unexpected fraud rate {fraud_rate:.4%} — expected ~0.172%")

    print("  ✓ Schema validation passed")
    return df


# ---------------------------------------------------------------------------
# Stage 2 — PySpark preprocessing
# ---------------------------------------------------------------------------

def preprocess_pyspark(df: pd.DataFrame) -> pd.DataFrame:
    """
    PySpark-based preprocessing step.

    In a production environment this runs on a Spark cluster.
    Here we use PySpark in local mode (auto-detected single-machine).
    Falls back to pandas if PySpark is unavailable.
    """
    print("\n[Stage 2] PySpark preprocessing")
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, isnull, sum as spark_sum
        from pyspark.ml.feature import StandardScaler as SparkScaler, VectorAssembler
        from pyspark.sql.types import DoubleType

        spark = (
            SparkSession.builder
            .appName("fraud_detection_pipeline")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        sdf = spark.createDataFrame(df)
        print(f"  PySpark DataFrame: {sdf.count():,} rows, {len(sdf.columns)} columns")

        # Null audit
        null_counts = {
            c: sdf.filter(isnull(col(c))).count()
            for c in EXPECTED_COLUMNS
        }
        total_nulls = sum(null_counts.values())
        if total_nulls > 0:
            print(f"  ⚠ Nulls found: { {k:v for k,v in null_counts.items() if v>0} }")
            sdf = sdf.dropna()
        else:
            print("  ✓ No nulls found")

        # Scale Amount and Time via PySpark ML pipeline
        assembler = VectorAssembler(inputCols=["Amount", "Time"], outputCol="amount_time_vec")
        scaler = SparkScaler(inputCol="amount_time_vec", outputCol="amount_time_scaled")

        sdf_vec = assembler.transform(sdf)
        scaler_model = scaler.fit(sdf_vec)
        sdf_scaled = scaler_model.transform(sdf_vec)

        # Convert back to pandas for downstream sklearn/xgboost steps
        df_out = sdf_scaled.drop("amount_time_vec", "amount_time_scaled").toPandas()
        spark.stop()
        print("  ✓ PySpark preprocessing complete")
        return df_out

    except ImportError:
        print("  PySpark not available — falling back to pandas preprocessing")
        return _preprocess_pandas(df)

    except Exception as e:
        print(f"  PySpark error ({e}) — falling back to pandas")
        return _preprocess_pandas(df)


def _preprocess_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas fallback for single-machine environments."""
    print("  Running pandas preprocessing…")
    df = df.copy()

    # Null audit
    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    if null_counts.sum() > 0:
        print(f"  ⚠ Nulls found:\n{null_counts[null_counts > 0]}")
        df = df.dropna(subset=EXPECTED_COLUMNS)
    else:
        print("  ✓ No nulls found")

    print("  ✓ Pandas preprocessing complete")
    return df


# ---------------------------------------------------------------------------
# Stage 3 — Feature engineering
# ---------------------------------------------------------------------------

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Stage 3] Feature engineering")
    df = engineer_features(df)
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    print(f"  Feature set: {len(available)} features")
    print(f"  Added: Amount_log, Time_sin, Time_cos, 5 V-interaction products")
    return df


# ---------------------------------------------------------------------------
# Stage 4 — Train/test split + SMOTE
# ---------------------------------------------------------------------------

def split_and_resample(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    use_smote: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified 80/20 split, then SMOTE on the training fold only.

    SMOTE is applied *after* the split — never to test data — to prevent
    any form of data leakage.
    """
    print("\n[Stage 4] Train/test split + class imbalance handling")

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"  Train fraud rate: {y_train.mean():.4%}  |  Test fraud rate: {y_test.mean():.4%}")

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  SMOTE applied → train set now {len(X_train):,} rows  "
                  f"(fraud rate: {y_train.mean():.4%})")
        except ImportError:
            print("  ⚠ imbalanced-learn not available — skipping SMOTE; "
                  "using scale_pos_weight instead")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Stage 5 — Model training
# ---------------------------------------------------------------------------

def run_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[object, ModelConfig]:
    """Train XGBoost with validation-based early stopping."""
    print("\n[Stage 5] XGBoost model training")

    # Hold out 10% of train for early-stopping validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.10,
        random_state=42,
        stratify=y_train,
    )
    print(f"  Training: {len(X_tr):,}  |  Validation (early-stop): {len(X_val):,}")

    config = ModelConfig(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=30,
    )

    model = train(X_tr, y_tr, X_val, y_val, config=config, balanced=True)
    return model, config


# ---------------------------------------------------------------------------
# Stage 6 — Threshold optimisation
# ---------------------------------------------------------------------------

def run_threshold_optimisation(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Sweep thresholds 0.10–0.90 and select via F2-score."""
    print("\n[Stage 6] Threshold optimisation")

    y_prob = model.predict_proba(X_test)[:, 1]
    sweep_df = sweep_thresholds(y_test.values, y_prob)

    best_f2   = select_threshold(sweep_df, strategy="f2")
    best_cost = select_threshold(sweep_df, strategy="cost")

    print(f"  F2-optimised threshold:   {best_f2.threshold:.2f}  "
          f"(recall={best_f2.recall:.3f}, precision={best_f2.precision:.3f})")
    print(f"  Cost-optimised threshold: {best_cost.threshold:.2f}  "
          f"(business_cost=${best_cost.business_cost:,.0f})")
    print(f"  → Using F2-optimised threshold: {best_f2.threshold:.2f}")

    return {"f2": best_f2, "cost": best_cost, "selected": best_f2, "sweep": sweep_df}


# ---------------------------------------------------------------------------
# Stage 7 — Scoring output
# ---------------------------------------------------------------------------

RISK_TIERS = {
    "LOW":    (0.00, 0.35),
    "MEDIUM": (0.35, 0.65),
    "HIGH":   (0.65, 1.01),
}


def assign_risk_tier(prob: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "HIGH"


def score_transactions(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_result,
    X_train_sample: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """
    Score all test transactions and write output CSV with:
      - fraud_probability
      - risk_tier
      - top 3 SHAP features
      - actual label (for evaluation)
    """
    print("\n[Stage 7] Automated scoring + SHAP explanations")
    output_dir = Path(output_dir)

    y_prob = model.predict_proba(X_test)[:, 1]

    # SHAP explanations
    shap_values, X_with_shap = explain_predictions(
        model=model,
        X_train_sample=X_train_sample,
        X_score=X_test,
        output_dir=output_dir / "reports",
        generate_plots=True,
    )

    # Assemble output DataFrame
    scored = X_with_shap[[c for c in X_with_shap.columns if c.startswith("shap_")]].copy()
    scored["fraud_probability"] = np.round(y_prob, 4)
    scored["risk_tier"]         = [assign_risk_tier(p) for p in y_prob]
    scored["predicted_fraud"]   = (y_prob >= threshold_result.threshold).astype(int)
    scored["actual_label"]      = y_test.values

    # Full scored file
    scored_path = output_dir / "scored_transactions.csv"
    scored.to_csv(scored_path, index=False)
    print(f"  Full scored output saved → {scored_path}  ({len(scored):,} rows)")

    # Sample (50 rows) for the repo example
    sample_path = output_dir / "scored_sample.csv"
    scored.head(50).to_csv(sample_path, index=False)
    print(f"  Sample (50 rows) saved  → {sample_path}")

    return scored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str, output_dir: str) -> None:
    t0 = time.time()
    print("\n" + "=" * 60)
    print("  CREDIT CARD FRAUD DETECTION — AUTOMATED PIPELINE")
    print("=" * 60)

    # Stage 1
    df = load_and_validate(data_path)

    # Stage 2
    df = preprocess_pyspark(df)

    # Stage 3
    df = run_feature_engineering(df)

    # Stage 4
    X_train, X_test, y_train, y_test = split_and_resample(df)

    # Stage 5
    model, model_config = run_training(X_train, y_train, X_test, y_test)

    # Stage 6
    threshold_info = run_threshold_optimisation(model, X_test, y_test)

    # Evaluate and print report
    print("\n[Evaluation]")
    report = evaluate(model, X_test, y_test, threshold_strategy="f2")
    report.print()

    # Stage 7 uses a train sample for SHAP, and the same sample is saved for app reuse
    X_train_sample = X_train.sample(n=min(500, len(X_train)), random_state=42)

    feature_schema = {
        "raw_input_columns": EXPECTED_COLUMNS,
        "engineered_feature_columns": FEATURE_COLUMNS,
        "interaction_pairs": [list(pair) for pair in INTERACTION_PAIRS],
        "risk_tiers": [
            {"name": tier, "min": bounds[0], "max": bounds[1]}
            for tier, bounds in RISK_TIERS.items()
        ],
        "target_column": "Class",
    }

    training_metadata = {
        "model_config": asdict(model_config),
        "evaluation": {
            "auc_roc": report.auc_roc,
            "auc_pr": report.auc_pr,
            "threshold_result": report.threshold_result.to_dict(),
            "confusion_matrix": report.confusion_matrix,
        },
        "threshold_selection": "f2",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "shap_background_rows": len(X_train_sample),
    }

    # Save model
    save_model(
        model,
        threshold_info["selected"],
        output_dir=Path(output_dir) / "model",
        feature_schema=feature_schema,
        metadata=training_metadata,
        shap_background_sample=X_train_sample,
    )

    scored = score_transactions(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold_result=threshold_info["selected"],
        X_train_sample=X_train_sample,
        output_dir=output_dir,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Model     → {output_dir}/model/")
    print(f"  Reports   → {output_dir}/reports/")
    print(f"  Scored    → {output_dir}/scored_transactions.csv")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — Automated ML Pipeline"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to creditcard.csv  (download from Kaggle)",
    )
    parser.add_argument(
        "--output",
        default="outputs/",
        help="Output directory for model, reports, and scored CSV (default: outputs/)",
    )
    args = parser.parse_args()
    run_pipeline(data_path=args.data, output_dir=args.output)
