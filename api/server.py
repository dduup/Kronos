"""Kronos Prediction API — FastAPI service with confidence intervals."""

import sys
import os
import time
from typing import Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import Kronos, KronosTokenizer, KronosPredictor

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
predictor: Optional[KronosPredictor] = None
MODEL_INFO = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class OhlcvData(BaseModel):
    """OHLCV time-series data passed as JSON arrays."""

    timestamps: List[str]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: Optional[List[float]] = None
    amount: Optional[List[float]] = None


class PredictRequest(BaseModel):
    """Prediction request — accepts either inline JSON data or a CSV file path."""

    data: Optional[OhlcvData] = None
    file_path: Optional[str] = None
    pred_len: int = Field(default=120, ge=1, le=512)
    lookback: Optional[int] = Field(default=None, ge=10, description="Number of historical data points to use. Defaults to all data minus pred_len.")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    sample_count: int = Field(default=10, ge=1, le=50)


class PredictionPoint(BaseModel):
    timestamp: str
    values: dict
    std: dict
    ci_95: dict


class PredictResponse(BaseModel):
    success: bool
    model: str
    pred_len: int
    predictions: List[PredictionPoint]


class HealthResponse(BaseModel):
    status: str
    model: dict


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kronos Prediction API",
    description="Kronos financial K-line foundation model — prediction with distribution statistics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup — load model
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_model():
    global predictor, MODEL_INFO
    print("Loading Kronos-base model and tokenizer...")
    t0 = time.time()

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    elapsed = time.time() - t0
    MODEL_INFO = {
        "name": "Kronos-base",
        "params": "102.3M",
        "context_length": 512,
        "device": str(predictor.device),
        "load_time": f"{elapsed:.1f}s",
    }
    print(f"Model loaded in {elapsed:.1f}s on {predictor.device}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return HealthResponse(
        status="ok",
        model=MODEL_INFO,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # --- Parse input data ---
    df, timestamps = _parse_input(req)

    if len(df) < 10:
        raise HTTPException(status_code=400, detail=f"Need at least 10 data points, got {len(df)}")

    total_rows = len(df)
    if req.lookback is not None:
        lookback = req.lookback
        if lookback + req.pred_len > total_rows:
            raise HTTPException(
                status_code=400,
                detail=f"Data has {total_rows} rows, but need lookback({lookback}) + pred_len({req.pred_len})",
            )
    else:
        lookback = total_rows - req.pred_len
        if lookback < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Data has {total_rows} rows, need at least pred_len({req.pred_len}) + 10 rows",
            )

    x_df = df.iloc[:lookback]
    x_timestamp = pd.Series(timestamps[:lookback])
    y_timestamp = pd.Series(timestamps[lookback : lookback + req.pred_len])

    # --- Run prediction ---
    try:
        pred_df, stats_df = predictor.predict_with_stats(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=req.pred_len,
            sample_count=req.sample_count,
            T=req.temperature,
            top_p=req.top_p,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # --- Build response ---
    feature_cols = ["open", "high", "low", "close", "volume", "amount"]
    predictions = []
    for idx in pred_df.index:
        values = {col: round(float(pred_df.loc[idx, col]), 6) for col in feature_cols}
        std = {col: round(float(stats_df.loc[idx, f"{col}_std"]), 6) for col in feature_cols}
        ci_95 = {
            f"{col}_lower": round(float(stats_df.loc[idx, f"{col}_ci_lower"]), 6)
            for col in feature_cols
        }
        ci_95.update(
            {
                f"{col}_upper": round(float(stats_df.loc[idx, f"{col}_ci_upper"]), 6)
                for col in feature_cols
            }
        )
        predictions.append(
            PredictionPoint(
                timestamp=str(idx),
                values=values,
                std=std,
                ci_95=ci_95,
            )
        )

    return PredictResponse(
        success=True,
        model=f"{MODEL_INFO['name']} ({MODEL_INFO['params']})",
        pred_len=req.pred_len,
        predictions=predictions,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_input(req: PredictRequest):
    """Parse request into (df, list_of_timestamps). Raises 400 on bad input."""

    if req.data is None and req.file_path is None:
        raise HTTPException(status_code=400, detail="Must provide either 'data' or 'file_path'")

    if req.data is not None and req.file_path is not None:
        raise HTTPException(status_code=400, detail="Provide either 'data' or 'file_path', not both")

    # --- JSON data ---
    if req.data is not None:
        d = req.data
        n = len(d.timestamps)
        for name in ("open", "high", "low", "close"):
            if len(getattr(d, name)) != n:
                raise HTTPException(
                    status_code=400,
                    detail=f"timestamps has {n} rows but {name} has {len(getattr(d, name))}",
                )

        df = pd.DataFrame(
            {
                "timestamps": pd.to_datetime(d.timestamps),
                "open": d.open,
                "high": d.high,
                "low": d.low,
                "close": d.close,
                "volume": d.volume if d.volume is not None else [0.0] * n,
                "amount": d.amount if d.amount is not None else [0.0] * n,
            }
        )
        timestamps = df["timestamps"].tolist()
        df = df.drop(columns=["timestamps"])
        return df, timestamps

    # --- CSV file path ---
    file_path = req.file_path
    PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    allowed_root = os.path.realpath(PROJECT_ROOT)
    real_path = os.path.realpath(file_path)
    if not real_path.startswith(allowed_root + os.sep) and real_path != allowed_root:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: file_path must be within the project directory",
        )
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    try:
        df = pd.read_csv(real_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Detect timestamp column
    ts_col = None
    for candidate in ("timestamps", "timestamp", "date", "datetime", "time"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise HTTPException(
            status_code=400,
            detail=f"No timestamp column found. Expected one of: timestamps, timestamp, date, datetime, time. Got columns: {list(df.columns)}",
        )

    df["timestamps"] = pd.to_datetime(df[ts_col])
    timestamps = df["timestamps"].tolist()
    df = df.drop(columns=[ts_col])
    return df, timestamps
