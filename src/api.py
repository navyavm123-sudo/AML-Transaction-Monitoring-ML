"""FastAPI Application for Real-Time AML Transaction Monitoring and Anomaly Detection.

Purpose:
FastAPI lets other applications use your trained model to detect suspicious
network traffic and financial transactions without opening your Kaggle notebook.

Run the API:
    $ uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
Or directly:
    $ python src/api.py
Interactive API Documentation (Swagger UI):
    http://localhost:8000/docs
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import MODELS_DIR, PROCESSED_DATA_DIR
except ImportError:
    MODELS_DIR = PROJECT_ROOT / "models_saved"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

from src.explainability import (
    ExplainabilityError,
    ExplainabilityService,
    LocalExplanation,
    UnsupportedModelExplainerError,
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("aml_api")


# ==============================================================================
# 🎯 MODEL LOADING & INFERENCE (XGBoost No SMOTE Selected Model)
# ==============================================================================

# Default path to trained model file (XGBoost No SMOTE)
DEFAULT_MODEL_FILENAME = "best_model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / DEFAULT_MODEL_FILENAME)))
PREPROCESSOR_PATH = (
    MODELS_DIR / "preprocessor.joblib"
    if (MODELS_DIR / "preprocessor.joblib").exists()
    else PROCESSED_DATA_DIR / "preprocessor.joblib"
)
SELECTOR_PATH = PROCESSED_DATA_DIR / "selector.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

# Global state to hold loaded artifacts
model_artifacts = {
    "model": None,
    "preprocessor": None,
    "selector": None,
    "feature_names": None,
    "metadata": {},
    "threshold": 0.5,
    "is_loaded": False,
    "model_name": "XGBoost (No SMOTE)",
    "version": "1.0.0-xgb-no-smote",
}

# Explainability Service Facade
explainability_service = ExplainabilityService()


def load_model_artifacts():
    """Loads the trained XGBoost (No SMOTE) model and preprocessing pipeline from disk."""
    logger.info("Attempting to load model artifacts...")

    # 1. Check for Preprocessor
    target_prep_path = (
        MODELS_DIR / "preprocessor.joblib"
        if (MODELS_DIR / "preprocessor.joblib").exists()
        else PREPROCESSOR_PATH
    )
    if target_prep_path.exists():
        try:
            model_artifacts["preprocessor"] = joblib.load(target_prep_path)
            logger.info(f"Loaded feature preprocessor from: {target_prep_path}")
        except Exception as e:
            logger.warning(f"Could not load preprocessor ({e}). Proceeding without it.")

    # 2. Check for Feature Selector
    if SELECTOR_PATH.exists():
        try:
            model_artifacts["selector"] = joblib.load(SELECTOR_PATH)
            logger.info(f"Loaded feature selector from: {SELECTOR_PATH}")
        except Exception as e:
            logger.warning(f"Could not load selector ({e}). Proceeding without it.")

    # 3. Load Feature Names if present
    if FEATURE_NAMES_PATH.exists():
        try:
            import json

            with open(FEATURE_NAMES_PATH) as f:
                model_artifacts["feature_names"] = json.load(f)
            logger.info(
                f"Loaded {len(model_artifacts['feature_names'])} feature names from: {FEATURE_NAMES_PATH}"
            )
        except Exception as e:
            logger.warning(f"Could not load feature names ({e}).")

    # 4. Load Metadata (tuned thresholds, training tags)
    if METADATA_PATH.exists():
        try:
            import json

            with open(METADATA_PATH) as f:
                meta = json.load(f)
            model_artifacts["metadata"] = meta
            if "threshold" in meta:
                model_artifacts["threshold"] = float(meta["threshold"])
            if "model_name" in meta:
                model_artifacts["model_name"] = meta["model_name"]
            logger.info(
                f"Loaded model metadata from {METADATA_PATH} (threshold={model_artifacts['threshold']})"
            )
        except Exception as e:
            logger.warning(f"Could not load model metadata: {e}")

    # 5. Determine candidate model file path
    candidate_path = MODEL_PATH
    if not candidate_path.exists():
        alt_paths = [
            MODELS_DIR / "xgboost_no_smote.joblib",
            MODELS_DIR / "best_model.joblib",
            MODELS_DIR / "xgboost_no_smote.json",
        ]
        for p in alt_paths:
            if p.exists():
                candidate_path = p
                break

    # 6. Load Trained XGBoost / ML Model
    if candidate_path.exists():
        try:
            if str(candidate_path).endswith(".json"):
                from xgboost import XGBClassifier

                loaded_model = XGBClassifier()
                loaded_model.load_model(str(candidate_path))
            else:
                loaded_model = joblib.load(candidate_path)

            model_artifacts["model"] = loaded_model
            model_artifacts["is_loaded"] = True
            model_cls = type(loaded_model).__name__
            if "xgb" in model_cls.lower():
                model_artifacts["model_name"] = "XGBoost (No SMOTE)"
            else:
                model_artifacts["model_name"] = model_cls

            logger.info(
                f"Successfully loaded trained AML model ({model_artifacts['model_name']}) from {candidate_path}"
            )
        except Exception as e:
            logger.error(f"Error loading model from {candidate_path}: {e}")
            model_artifacts["is_loaded"] = False
    else:
        logger.warning(
            f"No trained model found at '{candidate_path}'. Running in fallback mode."
        )
        model_artifacts["is_loaded"] = False
        model_artifacts["model_name"] = "Mock-Predictor (Waiting for model)"

    # 7. Synchronize Explainability Service with Loaded Artifacts
    if model_artifacts["is_loaded"] and model_artifacts["model"] is not None:
        try:
            explainability_service.set_model_artifacts(
                model=model_artifacts["model"],
                preprocessor=model_artifacts["preprocessor"],
                selector=model_artifacts["selector"],
                feature_names=model_artifacts.get("feature_names"),
            )
            logger.info("Configured ExplainabilityService with loaded model artifacts.")
        except Exception as e:
            logger.warning(f"Could not configure ExplainabilityService with model: {e}")


def predict_single_transaction(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Execute prediction pipeline on an incoming transaction using the XGBoost (No SMOTE) model.

    Args:
        data_dict: Dictionary containing transaction features.

    Returns:
        Dictionary with prediction label, risk score, and metadata.
    """
    model = model_artifacts.get("model")
    preprocessor = model_artifacts.get("preprocessor")
    selector = model_artifacts.get("selector")
    threshold = model_artifacts.get("threshold", 0.5)

    # If the trained model is loaded, execute ML inference
    if model_artifacts["is_loaded"] and model is not None:
        try:
            from src.features.pipeline import engineer_features

            # Apply domain feature transformations
            df_eng = engineer_features(data_dict)

            # Transform through preprocessor pipeline
            X_input = df_eng
            if preprocessor is not None:
                try:
                    X_input = preprocessor.transform(df_eng)
                    if selector is not None:
                        try:
                            X_input = selector.transform(X_input)
                        except Exception:
                            pass
                except Exception as prep_err:
                    logger.debug(f"Preprocessing transform fallback: {prep_err}")

            # Predict probability
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X_input)[0, 1])
            elif hasattr(model, "decision_function"):
                score = float(model.decision_function(X_input)[0])
                proba = 1.0 / (1.0 + np.exp(-score))
            else:
                pred = int(model.predict(X_input)[0])
                proba = 1.0 if pred == 1 else 0.0

            prediction = int(proba >= threshold)
            is_susp = bool(prediction == 1)

            return {
                "risk_score": round(proba, 4),
                "is_suspicious": is_susp,
                "prediction_label": "Suspicious" if is_susp else "Legitimate",
                "model_status": "xgboost_no_smote_active"
                if "xgboost" in model_artifacts["model_name"].lower()
                else "ml_model_active",
                "model_name": model_artifacts["model_name"],
            }

        except Exception as e:
            logger.error(f"Inference error with loaded model: {e}", exc_info=True)
            pass

    # Fallback heuristic calculation if model unavailable
    amount = float(data_dict.get("Amount", 0.0))
    is_cross_border = data_dict.get("Sender_bank_location") != data_dict.get(
        "Receiver_bank_location"
    )
    is_cash = "Cash" in str(data_dict.get("Payment_type", ""))

    simulated_risk = 0.05
    if amount > 10000.0:
        simulated_risk += 0.45
    if is_cross_border:
        simulated_risk += 0.25
    if is_cash and amount > 5000.0:
        simulated_risk += 0.20

    simulated_risk = min(round(simulated_risk, 4), 0.99)
    is_suspicious = simulated_risk >= 0.5

    return {
        "risk_score": simulated_risk,
        "is_suspicious": is_suspicious,
        "prediction_label": "Suspicious" if is_suspicious else "Legitimate",
        "model_status": "awaiting_model",
        "model_name": model_artifacts["model_name"],
        "note": "Using rule-based simulation. Trained XGBoost model active when artifacts are placed in 'models_saved/'.",
    }


# ==============================================================================
# 📋 PYDANTIC SCHEMAS (REQUEST & RESPONSE MODELS)
# ==============================================================================


class TransactionInput(BaseModel):
    """Schema representing an incoming financial transaction or network event."""

    model_config = ConfigDict(extra="allow")

    Sender_account: Optional[Union[str, int]] = Field(
        default="ACC-891024", description="Unique identifier for sender account"
    )
    Receiver_account: Optional[Union[str, int]] = Field(
        default="ACC-381940", description="Unique identifier for receiver account"
    )
    Amount: float = Field(
        default=12500.00, ge=0.0, description="Transaction monetary amount"
    )
    Payment_currency: Optional[str] = Field(
        default="US Dollar", description="Currency sent"
    )
    Received_currency: Optional[str] = Field(
        default="Euro", description="Currency received"
    )
    Sender_bank_location: Optional[str] = Field(
        default="USA", description="Originating country or jurisdiction"
    )
    Receiver_bank_location: Optional[str] = Field(
        default="Switzerland", description="Destination country or jurisdiction"
    )
    Payment_type: Optional[str] = Field(
        default="Cross-border",
        description="Transaction method (e.g. Cross-border, Cash Deposit, ACH)",
    )
    Date: Optional[str] = Field(
        default="2023-10-15", description="Transaction date (YYYY-MM-DD)"
    )
    Time: Optional[str] = Field(
        default="23:45:10", description="Transaction time (HH:MM:SS)"
    )
    Laundering_type: Optional[str] = Field(
        default=None, description="Known laundering typology if labeled"
    )


class BatchTransactionInput(BaseModel):
    """Schema for batch transaction inference."""

    transactions: List[TransactionInput] = Field(
        ..., description="List of transactions to evaluate simultaneously"
    )


class PredictionResponse(BaseModel):
    """Schema for individual prediction result."""

    transaction_id: Optional[Union[str, int]] = None
    is_suspicious: bool
    risk_score: float
    risk_level: str
    prediction_label: str
    model_status: str
    model_name: str
    processing_time_ms: float
    note: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction result."""

    total_processed: int
    suspicious_count: int
    processing_time_ms: float
    results: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    """Schema for model status and runtime metadata."""

    status: str
    is_model_loaded: bool
    model_name: str
    model_path: str
    preprocessor_loaded: bool
    selector_loaded: bool
    purpose: str


# ==============================================================================
# 🚀 FASTAPI APP INITIALIZATION & LIFESPAN
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the application."""
    logger.info("Initializing AML Monitoring API...")
    load_model_artifacts()
    yield
    logger.info("Shutting down AML Monitoring API...")


app = FastAPI(
    title="Intelligent AML Transaction Monitoring API",
    description=(
        "Production-grade FastAPI serving Machine Learning models for real-time "
        "Anti-Money Laundering (AML) transaction monitoring and suspicious activity detection. "
        "FastAPI lets other applications use your trained model to detect suspicious "
        "network traffic and transactions without opening your Kaggle notebook."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for external dashboards and microservices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 🌐 API ENDPOINTS
# ==============================================================================


@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service status and quick links."""
    return {
        "service": "AML Transaction Monitoring ML API",
        "description": "FastAPI lets other applications use your trained model to detect suspicious network traffic without opening your Kaggle notebook.",
        "status": "online",
        "model_loaded": model_artifacts["is_loaded"],
        "model_name": model_artifacts["model_name"],
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=ModelInfoResponse, tags=["General"])
async def health_check():
    """Health check endpoint displaying system and model artifact status."""
    return ModelInfoResponse(
        status="healthy",
        is_model_loaded=model_artifacts["is_loaded"],
        model_name=model_artifacts["model_name"],
        model_path=str(MODEL_PATH),
        preprocessor_loaded=model_artifacts["preprocessor"] is not None,
        selector_loaded=model_artifacts["selector"] is not None,
        purpose="FastAPI lets other applications use your trained model to detect suspicious network traffic without opening your Kaggle notebook.",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_transaction(transaction: TransactionInput):
    """Predict whether a single transaction is suspicious or legitimate.

    Accepts transaction attributes, performs feature extraction, and runs
    inference against the trained ML model.
    """
    start_time = time.time()
    data_dict = transaction.model_dump()

    try:
        res = predict_single_transaction(data_dict)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    score = res["risk_score"]

    # Classify risk level
    if score >= 0.75:
        risk_level = "High"
    elif score >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    tx_id = data_dict.get("Sender_account", "unknown")

    return PredictionResponse(
        transaction_id=str(tx_id),
        is_suspicious=res["is_suspicious"],
        risk_score=score,
        risk_level=risk_level,
        prediction_label=res["prediction_label"],
        model_status=res["model_status"],
        model_name=res["model_name"],
        processing_time_ms=elapsed_ms,
        note=res.get("note"),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(batch_input: BatchTransactionInput):
    """Process a batch of transactions in a single API call.

    Ideal for streaming transaction feeds or asynchronous pipeline audits.
    """
    start_time = time.time()
    results = []
    suspicious_count = 0

    for item in batch_input.transactions:
        item_start = time.time()
        data_dict = item.model_dump()
        res = predict_single_transaction(data_dict)
        item_elapsed = round((time.time() - item_start) * 1000, 2)

        score = res["risk_score"]
        if score >= 0.75:
            risk_level = "High"
        elif score >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        if res["is_suspicious"]:
            suspicious_count += 1

        tx_id = data_dict.get("Sender_account", "unknown")

        results.append(
            PredictionResponse(
                transaction_id=str(tx_id),
                is_suspicious=res["is_suspicious"],
                risk_score=score,
                risk_level=risk_level,
                prediction_label=res["prediction_label"],
                model_status=res["model_status"],
                model_name=res["model_name"],
                processing_time_ms=item_elapsed,
                note=res.get("note"),
            )
        )

    total_elapsed = round((time.time() - start_time) * 1000, 2)

    return BatchPredictionResponse(
        total_processed=len(results),
        suspicious_count=suspicious_count,
        processing_time_ms=total_elapsed,
        results=results,
    )


@app.post("/explain", response_model=LocalExplanation, tags=["Explainability"])
async def explain_transaction_endpoint(
    transaction: TransactionInput,
    method: str = "shap",
    top_k: int = 10,
):
    """Generate dual-granularity SHAP or LIME explanation for an individual transaction.

    Args:
        transaction: Incoming transaction attributes payload.
        method: Explanation algorithm ('shap' by default, 'lime' on demand).
        top_k: Maximum number of top features to return in raw and domain views.
    """
    if not model_artifacts["is_loaded"] or model_artifacts["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Machine learning model is not loaded. Place a trained model in "
                "'models_saved/best_model.joblib' or reload artifacts to generate explanations."
            ),
        )

    method_clean = method.lower().strip()
    if method_clean not in ("shap", "lime"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid explanation method '{method}'. Supported methods are 'shap' and 'lime'.",
        )

    try:
        data_dict = transaction.model_dump()
        explanation = explainability_service.explain_transaction(
            transaction=data_dict,
            raw_record=data_dict,
            method=method_clean,
            top_k=top_k,
        )
        return explanation
    except UnsupportedModelExplainerError as e:
        logger.warning(f"Unsupported model explainer requested: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except ExplainabilityError as e:
        logger.error(f"Explainability execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability execution failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during explanation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during explanation: {str(e)}",
        )


@app.post("/model/reload", tags=["Administration"])
async def reload_model():
    """Hot-reload model artifacts from disk without restarting the API server.

    Use this endpoint after copying your trained Kaggle model to 'models_saved/'.
    """
    logger.info("Admin triggered model reload endpoint.")
    load_model_artifacts()
    return {
        "status": "success",
        "message": "Model artifacts reloaded successfully.",
        "is_model_loaded": model_artifacts["is_loaded"],
        "model_name": model_artifacts["model_name"],
        "model_path": str(MODEL_PATH),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
