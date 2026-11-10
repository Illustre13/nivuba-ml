# --- Imports ---
from pathlib import Path
from typing import Any
import logging
import logging.config
import json
import uuid
import time
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from huggingface_hub import hf_hub_download
import geoip2.database
from geoip2.errors import AddressNotFoundError

from backend.schemas import (
    PipelineInput,
    PredictionEnum,
    PredictedProbabilities,
    PredictionResult,
    PredictionResponse,
    NivubaInput,
    NivubaResponse,
)
from src.custom_transformers import (
    MissingValueChecker,
    MissingValueStandardizer,
    RobustSimpleImputer,
    SnakeCaseFormatter,
    BooleanColumnTransformer,
    JobStabilityTransformer,
    CityTierTransformer,
    StateDefaultRateTargetEncoder,
    RobustStandardScaler,
    RobustOneHotEncoder,
    RobustOrdinalEncoder,
    FeatureSelector
)
from src.global_constants import COLUMNS_TO_KEEP

# --- Logging ---
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "monitoring": {
            "format": "%(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "monitoring_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "monitoring",
            "filename": str(log_dir / "prediction_logs.jsonl"),
            "maxBytes": 10485760,
            "backupCount": 3,
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "INFO",
        },
        "monitoring": {
            "handlers": ["monitoring_file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
monitoring_logger = logging.getLogger("monitoring")


# --- Helper Functions ---
def get_batch_metadata(
    pipeline_input_dict_ls: list[dict[str, Any]],
    request: Request,
    geoip_reader: geoip2.database.Reader | None
) -> dict[str, Any]:
    user_agent = pipeline_input_dict_ls[0].get("user_agent", None)
    if user_agent is None:
        user_agent = request.headers.get("user-agent", "unknown")
    client_ip = pipeline_input_dict_ls[0].get("client_ip", None)
    if client_ip is None:
        x_forwarded_for = request.headers.get("x-forwarded-for")
        client_ip = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else request.client.host

    client_country = "unknown"
    if geoip_reader and client_ip:
        try:
            response = geoip_reader.country(client_ip)
            client_country = response.country.name
        except AddressNotFoundError:
            logger.debug("IP address not found in GeoLite2 country database.")

    return {
        "batch_id": str(uuid.uuid4()),
        "batch_size": len(pipeline_input_dict_ls),
        "batch_timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION_TAG,
        "client_country": client_country,
        "user_agent": user_agent,
    }


def load_pipeline_from_local(path: str | Path) -> Pipeline:
    if not isinstance(path, (str, Path)):
        raise TypeError(f"'path' must be a string or Path object, got {type(path).__name__}")
    path_str = str(path)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at '{path_str}'")
    try:
        logger.info(f"Loading pipeline from '{path_str}'...")
        pipeline = joblib.load(path_str)
        logger.info("Successfully loaded pipeline.")
    except Exception as e:
        raise RuntimeError(f"Error loading pipeline from '{path_str}'") from e
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Loaded object is not a scikit-learn Pipeline")
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Loaded pipeline does not have a .predict_proba() method")
    return pipeline


def load_pipeline_from_huggingface(repo_id: str, filename: str, revision: str) -> Pipeline:
    try:
        logger.info(
            f"Downloading pipeline '{filename}' (tag '{revision}') from '{repo_id}'. "
            "If already cached, will use local copy."
        )
        pipeline_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return load_pipeline_from_local(pipeline_path)
    except Exception as e:
        raise RuntimeError(f"Error loading pipeline '{filename}' from '{repo_id}': {e}") from e


# --- Geolocation Database ---
GEO_DB_PATH = Path("geoip_db/GeoLite2-Country.mmdb")
try:
    geoip_reader = geoip2.database.Reader(GEO_DB_PATH)
    logger.info(f"Loaded GeoLite2 country database from '{GEO_DB_PATH}'")
except FileNotFoundError:
    logger.warning(f"GeoLite2 database not found at '{GEO_DB_PATH}'. Client country will not be logged.")
    geoip_reader = None

# --- ML Pipeline ---
PIPELINE_VERSION_TAG = "v1.0"
PIPELINE_LOCAL_PATH = Path(os.getenv("PIPELINE_LOCAL_PATH", "models/loan_default_rf_pipeline.joblib"))

# Prefer the artifact baked into the image (see Dockerfile); download only if it is missing.
if PIPELINE_LOCAL_PATH.exists():
    pipeline = load_pipeline_from_local(PIPELINE_LOCAL_PATH)
else:
    logger.info("No local pipeline at '%s'; downloading from Hugging Face Hub.", PIPELINE_LOCAL_PATH)
    pipeline = load_pipeline_from_huggingface(
        repo_id="JensBender/loan-default-prediction-pipeline",
        filename="loan_default_rf_pipeline.joblib",
        revision=PIPELINE_VERSION_TAG,
    )

# --- FastAPI App ---
app = FastAPI(title="Nivuba ML Service")


# --- Health ---
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# --- Original pipeline endpoint (full feature set) ---
@app.post("/api/predict", response_model=PredictionResponse)
def predict(pipeline_input: PipelineInput | list[PipelineInput], request: Request) -> PredictionResponse:
    batch_metadata = None
    pipeline_input_dict_ls = None
    try:
        if isinstance(pipeline_input, list):
            if pipeline_input == []:
                return PredictionResponse(results=[])
            pipeline_input_dict_ls = [inp.model_dump() for inp in pipeline_input]
        else:
            pipeline_input_dict_ls = [pipeline_input.model_dump()]

        batch_metadata = get_batch_metadata(pipeline_input_dict_ls, request, geoip_reader)

        pipeline_input_cleaned = [
            {k: v for k, v in d.items() if k not in {"client_ip", "user_agent"}}
            for d in pipeline_input_dict_ls
        ]

        pipeline_input_df: pd.DataFrame = pd.DataFrame(pipeline_input_cleaned)

        start_time = time.perf_counter()
        predicted_probabilities: np.ndarray = pipeline.predict_proba(pipeline_input_df)
        pipeline_prediction_latency_ms = round((time.perf_counter() - start_time) * 1000)

        optimized_threshold: float = 0.29
        predictions: np.ndarray = (predicted_probabilities[:, 1] >= optimized_threshold)

        batch_metadata.update({
            "batch_latency_ms": pipeline_prediction_latency_ms,
            "avg_prediction_latency_ms": round(pipeline_prediction_latency_ms / len(pipeline_input_dict_ls)),
        })

        results: list[PredictionResult] = []
        for i, (pred, pred_proba) in enumerate(zip(predictions, predicted_probabilities)):
            prediction_enum = PredictionEnum.DEFAULT if pred else PredictionEnum.NO_DEFAULT
            prediction_result = PredictionResult(
                prediction=prediction_enum,
                probabilities=PredictedProbabilities(
                    default=float(pred_proba[1]),
                    no_default=float(pred_proba[0])
                )
            )
            results.append(prediction_result)

            monitoring_logger.info(json.dumps({
                **batch_metadata,
                "prediction_id": str(uuid.uuid4()),
                "inputs": pipeline_input_cleaned[i],
                "prediction": prediction_enum.value,
                "probabilities": {"default": float(pred_proba[1]), "no_default": float(pred_proba[0])},
            }))

        return PredictionResponse(results=results)

    except Exception as e:
        logger.error("Error during /api/predict: %s", e, exc_info=True)
        if pipeline_input_dict_ls:
            if batch_metadata is None:
                batch_metadata = {
                    "batch_id": str(uuid.uuid4()),
                    "batch_size": len(pipeline_input_dict_ls),
                    "batch_timestamp": datetime.now(timezone.utc).isoformat(),
                    "pipeline_version": PIPELINE_VERSION_TAG,
                    "client_country": None,
                    "user_agent": None,
                }
            for inp in pipeline_input_dict_ls:
                monitoring_logger.error(json.dumps({
                    **batch_metadata,
                    "prediction_id": str(uuid.uuid4()),
                    "inputs": inp,
                    "prediction": None,
                    "probabilities": None,
                    "error_message": str(e),
                }))
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


# --- Nivuba adapter endpoint ---
# Consumed by @nivuba-core PredictionsService.
# Maps the NestJS payload to pipeline features using safe defaults for fields
# the backend does not collect, then converts P(default) to a 0-1000 score
# (score = round((1 - P(default)) * 1000); higher is safer). The approval cut-off
# is owned by nivuba-core (DECISION_MIN_SCORE, default 700 == P(default) <= 0.30).
@app.post("/predict", response_model=NivubaResponse)
def predict_nivuba(payload: NivubaInput) -> NivubaResponse:
    try:
        annual_income = max(0, int(round(float(payload.salary) * 12)))
        experience = min(int(payload.termMonths // 12), 20)

        pipeline_row = {
            "income": annual_income,
            "age": 35,
            "experience": experience,
            "married": None,
            "house_ownership": None,
            "car_ownership": None,
            "profession": "analyst",
            "city": "mumbai",
            "state": "maharashtra",
            "current_job_yrs": min(experience, 14),
            "current_house_yrs": 10,
        }

        df = pd.DataFrame([pipeline_row])
        probabilities: np.ndarray = pipeline.predict_proba(df)
        p_default = float(probabilities[0, 1])

        score = max(0, min(1000, round((1.0 - p_default) * 1000)))

        feature_names = list(COLUMNS_TO_KEEP) if hasattr(pipeline[-1], "feature_importances_") else []
        shap_values: dict[str, float] = {}
        if feature_names:
            shap_values = {
                name: round(float(imp), 6)
                for name, imp in zip(feature_names, pipeline[-1].feature_importances_)
            }

        logger.info(
            "Nivuba predict | loanId=%s userId=%s salary=%s score=%d p_default=%.3f",
            payload.loanApplicationId, payload.userId, payload.salary, score, p_default,
        )
        return NivubaResponse(score=score, shap_values=shap_values)

    except Exception as e:
        logger.error("Error in /predict (nivuba): %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")
