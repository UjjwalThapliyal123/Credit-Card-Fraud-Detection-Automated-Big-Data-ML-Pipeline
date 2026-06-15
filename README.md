# Credit Card Fraud Detection — Automated ML Pipeline

> End-to-end automated fraud scoring pipeline built on 284K+ real transaction records.  
> Zero manual intervention from raw data to per-transaction risk output.  
> Stack: **Python · PySpark · XGBoost · SMOTE · SHAP · scikit-learn**

---

## Why this project exists

Credit card fraud detection is not a classification tutorial — it is a **business cost optimisation problem** under extreme class imbalance (0.172% fraud rate). A model that predicts "not fraud" for every transaction achieves 99.8% accuracy and catches zero fraud. This project treats it the way a production team would: design the pipeline first, then make deliberate decisions at every stage — class weighting, threshold selection, and cost-aware evaluation.

The pipeline runs fully automated from raw CSV to scored output. No manual relabelling, no hand-tuned thresholds, no notebook cells that require human decisions mid-run.

---

## Dataset

**Source:** [ULB Machine Learning Group — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Records:** 284,807 transactions · September 2013 · European cardholders  
**Fraud rate:** 492 frauds / 284,807 total = **0.172%**  
**Features:** V1–V28 (PCA-anonymised) · `Time` · `Amount` · `Class` (0 = legit, 1 = fraud)

> The dataset is real transaction data, not synthetic. The features are PCA-transformed for privacy — this is exactly how financial data arrives in practice.

---

## Project structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv              ← Kaggle download (not committed to git)
│
├── notebooks/
│   ├── 01_eda.ipynb                ← Distribution analysis, imbalance visualisation
│   ├── 02_preprocessing.ipynb      ← PySpark pipeline, feature engineering
│   ├── 03_modelling.ipynb          ← SMOTE, XGBoost, threshold tuning
│   └── 04_evaluation.ipynb         ← SHAP, business cost analysis, final report
│
├── src/
│   ├── pipeline.py                 ← End-to-end automated pipeline script
│   ├── features.py                 ← Feature engineering functions
│   ├── model.py                    ← Training, evaluation, threshold selection
│   └── explain.py                  ← SHAP explanation utilities
│
├── outputs/
│   ├── model/                      ← Saved XGBoost model + threshold config
│   ├── reports/                    ← Evaluation charts, confusion matrices
│   └── scored_sample.csv           ← Example scored output (50 rows)
│
├── requirements.txt
└── README.md
```

---

## Pipeline stages (fully automated)

### Stage 1 — Data ingestion
- Load raw CSV with schema validation
- Assert expected row count, column names, data types
- Fail loudly if schema drifts — no silent errors

### Stage 2 — PySpark preprocessing
- Load into PySpark DataFrame for scalable processing
- Null check and imputation audit
- `StandardScaler` on `Amount` and `Time` (V1–V28 already PCA-scaled)
- Stratified 80/20 train-test split, preserving fraud ratio

### Stage 3 — Feature engineering
- **`Amount`**: log1p transform — removes right-skew, compresses outliers
- **`Time`**: cyclical encoding (`sin`/`cos` of hour within day) — captures time-of-day fraud patterns without a linear assumption
- **V-feature interactions**: top 5 pairwise products from correlation analysis (e.g. `V14 × V17`) — consistently top SHAP contributors in fraud literature

### Stage 4 — Class imbalance handling
Two strategies, compared:
- `SMOTE` (Synthetic Minority Oversampling): generates synthetic fraud cases in feature space — applied to training set only, never to test set
- `class_weight='balanced'` in XGBoost: penalises misclassified fraud more heavily

> Why both? SMOTE changes the data. Class weighting changes the loss function. They address imbalance differently and their interaction is worth documenting — this is the kind of design decision that appears in a real ML review.

### Stage 5 — XGBoost model training
- `XGBClassifier` with `eval_metric='aucpr'` — area under precision-recall curve, the correct metric under imbalance (AUC-ROC is misleading when negatives dominate)
- Early stopping on validation set to prevent overfitting
- Logged: AUC-ROC, AUC-PR, F1, precision, recall at default threshold

### Stage 6 — Threshold optimisation
Default threshold (0.5) is wrong for fraud — catching fraud costs less than missing it. Pipeline sweeps thresholds 0.1–0.9 and selects based on:
- **F2-score** (weighs recall 2× over precision) — conservative, maximises fraud catch rate
- **Business cost matrix**: assign cost to false negatives (missed fraud) vs false positives (blocked legit transactions)
- Final threshold logged with its precision/recall trade-off documented

### Stage 7 — Automated scoring output
For each transaction, the pipeline outputs:
- `fraud_probability` (float, 0–1)
- `risk_tier` (`LOW` / `MEDIUM` / `HIGH` based on threshold bands)
- `top_3_shap_features` — which features drove this score, for auditability

---

## Key results

| Metric | Value |
|--------|-------|
| AUC-ROC | ~0.98 |
| AUC-PR | ~0.85 |
| Recall (fraud) | ~0.88 |
| Precision (fraud) | ~0.82 |
| False positive rate | ~0.03% |
| Threshold selected | 0.35 (F2-optimised) |

> Note: exact figures from your run — fill these in after running the pipeline.

---

## What makes this non-tutorial

Most fraud detection notebooks on Kaggle do three things: load data, run `RandomForest.fit()`, report 99.9% accuracy, done. This pipeline makes **deliberate documented decisions** at each stage:

1. **Metric choice**: why AUC-PR over AUC-ROC under imbalance
2. **Threshold as a business decision**: the threshold is not 0.5, it is derived from a cost function
3. **SHAP explanations**: every prediction has a human-readable reason — this is what a real fraud ops team needs to act on it
4. **PySpark for scalability**: the preprocessing stage is written in PySpark, not pandas — it scales to 10× or 100× the data without code changes
5. **No data leakage**: SMOTE applied only to training fold, scaler fitted only on training fold

---

## How to run

```bash
# 1. Download dataset from Kaggle
# Place creditcard.csv in data/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full automated pipeline
python src/pipeline.py --data data/creditcard.csv --output outputs/

# Pipeline runs all 7 stages, saves model + scored output, prints evaluation report
```

### Runtime artifacts for FastAPI / Streamlit

After the pipeline runs, `outputs/model/` contains the bundle your backend and frontend can load directly:

- `xgboost_fraud_model.json` — trained XGBoost model
- `threshold_config.json` — selected decision threshold and its metrics
- `feature_schema.json` — raw input columns, engineered feature order, interaction pairs, risk tiers
- `training_metadata.json` — model config and evaluation summary
- `shap_background_sample.csv` — reusable SHAP background sample for explanations
- `artifact_manifest.json` — file map for the runtime bundle

For inference code, load the bundle from `src/model.py::load_artifacts()`.

**requirements.txt**
```
pyspark>=3.4.0
xgboost>=1.7.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
shap>=0.43.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Notebooks walkthrough

| Notebook | What to read here |
|----------|-------------------|
| `01_eda.ipynb` | Class imbalance visualisation, Amount/Time distributions, fraud vs legit feature distributions for V1–V28 |
| `02_preprocessing.ipynb` | PySpark DataFrame operations, scaling decisions, feature engineering rationale |
| `03_modelling.ipynb` | SMOTE vs class_weight comparison, XGBoost training, threshold sweep with cost matrix |
| `04_evaluation.ipynb` | SHAP summary plot, SHAP waterfall for individual predictions, final confusion matrix, business impact estimate |

---

## Business framing (for interviews)

> "This isn't just a model — it's an automated decision pipeline. The most interesting decisions aren't in the model itself, they're in stage 4 (how you handle 0.172% fraud rate without the model ignoring all fraud) and stage 6 (how you set the threshold based on what false negatives actually cost the business versus false positives). SHAP output in stage 7 is what allows a fraud analyst to act on a flag — without it, the model is a black box that no operations team will trust."

---

## Relevance to Sales Operations automation

This project demonstrates:
- **Automated pipeline design** with zero manual intervention (direct JD requirement)
- **PySpark** for scalable data processing in big data environments  
- **Business cost-aware modelling** — not just accuracy, but operational impact
- **Explainable outputs** — SHAP explanations translate model signals into actionable decisions for non-technical stakeholders
- **Production thinking**: schema validation, train-test hygiene, logged outputs

---

*Dataset: [Kaggle MLG-ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | License: DbCL*
