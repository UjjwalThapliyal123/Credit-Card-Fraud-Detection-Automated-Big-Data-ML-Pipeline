from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, create_model

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from runtime import load_runtime_bundle, predict_batch, predict_record  # noqa: E402


def _build_transaction_model() -> type[BaseModel]:
    fields: dict[str, tuple[type, Field]] = {
        "Time": (float, Field(..., description="Transaction time in seconds from the first record")),
        "Amount": (float, Field(..., description="Transaction amount")),
    }
    for index in range(1, 29):
        fields[f"V{index}"] = (float, Field(..., description=f"PCA-transformed feature V{index}"))

    return create_model(
        "TransactionInput",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


TransactionInput = _build_transaction_model()

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime_bundle = load_runtime_bundle()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": runtime_bundle.get("model") is not None,
        "threshold": runtime_bundle.get("threshold"),
        "feature_count": len(runtime_bundle["feature_columns"]),
    }


@app.get("/metadata")
def metadata() -> dict:
    return {
        "threshold_config": runtime_bundle.get("threshold_config"),
        "feature_schema": runtime_bundle.get("feature_schema"),
        "training_metadata": runtime_bundle.get("training_metadata"),
        "artifact_manifest": runtime_bundle.get("artifact_manifest"),
    }


@app.post("/predict")
def predict(transaction: TransactionInput) -> dict:
    try:
        return predict_record(transaction.model_dump(), runtime_bundle, include_explanations=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/batch-predict")
def batch_predict(transactions: list[TransactionInput]) -> dict:
    try:
        frame = predict_batch(payload=pd.DataFrame([item.model_dump() for item in transactions]), bundle=runtime_bundle, include_explanations=False)
        return {"rows": len(frame), "results": frame.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
