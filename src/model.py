"""
model.py
--------
XGBoost model training, evaluation, and threshold optimisation.

Design decisions documented here:
  - eval_metric = 'aucpr'  (AUC-PR is the correct metric under class imbalance)
  - F2-score drives threshold selection (recall weighted 2× over precision)
  - Business cost matrix as an alternative threshold strategy
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 1.0        # overridden when class_weight='balanced'
    eval_metric: str = "aucpr"
    early_stopping_rounds: int = 30
    random_state: int = 42
    n_jobs: int = -1

    def to_xgb_params(self) -> dict:
        return {
            "n_estimators":        self.n_estimators,
            "max_depth":           self.max_depth,
            "learning_rate":       self.learning_rate,
            "subsample":           self.subsample,
            "colsample_bytree":    self.colsample_bytree,
            "scale_pos_weight":    self.scale_pos_weight,
            "eval_metric":         self.eval_metric,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state":        self.random_state,
            "n_jobs":              self.n_jobs,
            "use_label_encoder":   False,
        }


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    f2: float
    false_positive_rate: float
    business_cost: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    auc_roc: float
    auc_pr: float
    threshold_result: ThresholdResult
    confusion_matrix: list[list[int]]

    def print(self) -> None:
        print("\n" + "=" * 55)
        print("  FRAUD DETECTION — EVALUATION REPORT")
        print("=" * 55)
        print(f"  AUC-ROC          : {self.auc_roc:.4f}")
        print(f"  AUC-PR           : {self.auc_pr:.4f}")
        print(f"\n  — Threshold = {self.threshold_result.threshold:.2f} (F2-optimised) —")
        print(f"  Precision        : {self.threshold_result.precision:.4f}")
        print(f"  Recall           : {self.threshold_result.recall:.4f}")
        print(f"  F1               : {self.threshold_result.f1:.4f}")
        print(f"  F2               : {self.threshold_result.f2:.4f}")
        print(f"  False-positive % : {self.threshold_result.false_positive_rate * 100:.3f}%")
        print("\n  Confusion matrix (rows=actual, cols=predicted):")
        cm = self.confusion_matrix
        print(f"                Pred 0    Pred 1")
        print(f"  Actual 0 :  {cm[0][0]:>8,}  {cm[0][1]:>8,}")
        print(f"  Actual 1 :  {cm[1][0]:>8,}  {cm[1][1]:>8,}")
        print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_model(config: ModelConfig | None = None) -> XGBClassifier:
    """Instantiate an XGBClassifier with the given (or default) config."""
    cfg = config or ModelConfig()
    return XGBClassifier(**cfg.to_xgb_params())


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: ModelConfig | None = None,
    balanced: bool = True,
) -> XGBClassifier:
    """
    Train XGBoost with early stopping on the validation AUC-PR.

    Parameters
    ----------
    balanced : if True, sets scale_pos_weight = neg_count / pos_count,
               making the model weight fraud misclassification more heavily.
    """
    cfg = config or ModelConfig()

    if balanced:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        cfg.scale_pos_weight = neg / pos
        print(f"  scale_pos_weight set to {cfg.scale_pos_weight:.1f} (neg/pos ratio)")

    model = build_model(cfg)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iteration = model.best_iteration
    print(f"  Training complete — best iteration: {best_iteration}")
    return model


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------

def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 200.0,
    cost_fp: float = 5.0,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Sweep decision thresholds from 0.1 to 0.9 and compute metrics at each.

    Parameters
    ----------
    cost_fn : business cost of a false negative (missed fraud), in dollars
    cost_fp : business cost of a false positive (blocked legit tx), in dollars

    Returns
    -------
    DataFrame with one row per threshold, sorted ascending by threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.01)

    rows = []
    neg_total = (y_true == 0).sum()

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = f1_score(y_true, y_pred, zero_division=0)
        f2    = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        fpr   = fp / neg_total if neg_total > 0 else 0.0
        cost  = fn * cost_fn + fp * cost_fp

        rows.append({
            "threshold":           round(float(t), 2),
            "precision":           round(prec,  4),
            "recall":              round(rec,   4),
            "f1":                  round(f1,    4),
            "f2":                  round(f2,    4),
            "false_positive_rate": round(fpr,   6),
            "business_cost":       round(cost,  2),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_threshold(
    sweep_df: pd.DataFrame,
    strategy: str = "f2",
) -> ThresholdResult:
    """
    Select the best threshold from the sweep DataFrame.

    Strategies
    ----------
    'f2'   : maximise F2-score (recall weighted 2×)
    'cost' : minimise total business cost
    'f1'   : maximise F1 (balanced precision-recall)
    """
    strategy_map = {
        "f2":   ("f2",             "max"),
        "cost": ("business_cost",  "min"),
        "f1":   ("f1",             "max"),
    }
    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(strategy_map)}")

    col, direction = strategy_map[strategy]
    if direction == "max":
        best_row = sweep_df.loc[sweep_df[col].idxmax()]
    else:
        best_row = sweep_df.loc[sweep_df[col].idxmin()]

    return ThresholdResult(
        threshold=          best_row["threshold"],
        precision=          best_row["precision"],
        recall=             best_row["recall"],
        f1=                 best_row["f1"],
        f2=                 best_row["f2"],
        false_positive_rate=best_row["false_positive_rate"],
        business_cost=      best_row.get("business_cost"),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_strategy: str = "f2",
) -> EvaluationReport:
    """
    Full evaluation: threshold sweep + AUC metrics + confusion matrix.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr  = average_precision_score(y_test, y_prob)

    sweep_df = sweep_thresholds(y_test.values, y_prob)
    best     = select_threshold(sweep_df, strategy=threshold_strategy)

    y_pred = (y_prob >= best.threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    return EvaluationReport(
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        threshold_result=best,
        confusion_matrix=cm,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    model: XGBClassifier,
    threshold_result: ThresholdResult,
    output_dir: str | Path,
    feature_schema: dict | None = None,
    metadata: dict | None = None,
    shap_background_sample: pd.DataFrame | None = None,
) -> None:
    """Save the XGBoost model plus runtime artifacts for app inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgboost_fraud_model.json"
    model.save_model(str(model_path))
    print(f"  Model saved → {model_path}")

    config_path = output_dir / "threshold_config.json"
    with open(config_path, "w") as f:
        json.dump(threshold_result.to_dict(), f, indent=2)
    print(f"  Threshold config saved → {config_path}")

    if feature_schema is not None:
        feature_schema_path = output_dir / "feature_schema.json"
        with open(feature_schema_path, "w") as f:
            json.dump(feature_schema, f, indent=2)
        print(f"  Feature schema saved → {feature_schema_path}")

    if metadata is not None:
        metadata_path = output_dir / "training_metadata.json"
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        with open(metadata_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Training metadata saved → {metadata_path}")

    if shap_background_sample is not None:
        background_path = output_dir / "shap_background_sample.csv"
        shap_background_sample.to_csv(background_path, index=False)
        print(f"  SHAP background sample saved → {background_path}")

    manifest = {
        "model": model_path.name,
        "threshold_config": config_path.name,
    }
    if feature_schema is not None:
        manifest["feature_schema"] = "feature_schema.json"
    if metadata is not None:
        manifest["training_metadata"] = "training_metadata.json"
    if shap_background_sample is not None:
        manifest["shap_background_sample"] = "shap_background_sample.csv"

    manifest_path = output_dir / "artifact_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Artifact manifest saved → {manifest_path}")


def load_model(model_dir: str | Path) -> tuple[XGBClassifier, dict]:
    """Load model and threshold config from disk."""
    model_dir = Path(model_dir)

    model = XGBClassifier()
    model.load_model(str(model_dir / "xgboost_fraud_model.json"))

    with open(model_dir / "threshold_config.json") as f:
        threshold_config = json.load(f)

    return model, threshold_config


def load_artifacts(model_dir: str | Path) -> dict:
    """Load the model, threshold config, and optional runtime artifacts."""
    model_dir = Path(model_dir)

    model, threshold_config = load_model(model_dir)

    feature_schema = None
    feature_schema_path = model_dir / "feature_schema.json"
    if feature_schema_path.exists():
        with open(feature_schema_path) as f:
            feature_schema = json.load(f)

    training_metadata = None
    training_metadata_path = model_dir / "training_metadata.json"
    if training_metadata_path.exists():
        with open(training_metadata_path) as f:
            training_metadata = json.load(f)

    shap_background_sample = None
    background_path = model_dir / "shap_background_sample.csv"
    if background_path.exists():
        shap_background_sample = pd.read_csv(background_path)

    artifact_manifest = None
    manifest_path = model_dir / "artifact_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            artifact_manifest = json.load(f)

    return {
        "model": model,
        "threshold_config": threshold_config,
        "feature_schema": feature_schema,
        "training_metadata": training_metadata,
        "shap_background_sample": shap_background_sample,
        "artifact_manifest": artifact_manifest,
    }
