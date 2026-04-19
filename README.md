# @nivuba-ml

FastAPI ML service for the **Nivuba** loan prediction platform. Exposes the Random Forest loan-default pipeline as a REST API consumed by `@nivuba-core`.

Part of the `nivuba` monorepo alongside `@nivuba-core` (NestJS backend) and `@nivuba-ui` (React frontend).

---

## Model

Random Forest Classifier trained on 252,000 loan applications.

| Metric | Value |
|---|---|
| AUC-PR | 0.59 |
| Recall (defaulters) | 0.79 |
| Precision (defaulters) | 0.51 |
| Decision threshold | 0.29 |

Pipeline sourced from [Hugging Face Hub](https://huggingface.co/JensBender/loan-default-prediction-pipeline) at startup.

---

## Stack

| Concern | Technology |
|---|---|
| API framework | FastAPI 0.115 + Uvicorn |
| Data validation | Pydantic v2 |
| ML pipeline | scikit-learn 1.6 + joblib |
| Geolocation logging | MaxMind GeoLite2 (optional) |
| Container | Docker (python:3.10-slim-bookworm) |

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/predict` | **Nivuba adapter** — consumed by `@nivuba-core` |
| `POST` | `/api/predict` | Full pipeline endpoint (all 11 features required) |

### `POST /predict` — Nivuba adapter

Accepts the NestJS `PredictRequestDto` payload and maps it to pipeline features using safe defaults for fields not collected by the platform. Returns a 0–1000 score and RF feature importances.

**Score logic:** `score = round((1 - P(default)) * 1000)`
- score ≥ 700 → P(default) < 0.30, aligning with the 0.29 optimised threshold → **APPROVED**
- score < 700 → **REJECTED**

**Request:**
```json
{
  "userId": "uuid",
  "loanApplicationId": "uuid",
  "salary": 5000.00,
  "requestedAmount": 20000.00,
  "termMonths": 24,
  "userType": "GENERAL",
  "organizationId": null
}
```

**Response:**
```json
{
  "score": 742,
  "shap_values": {
    "income": 0.312,
    "age": 0.198,
    "state_default_rate": 0.147
  }
}
```

### `POST /api/predict` — Full pipeline endpoint

Accepts all 11 original training features. Supports single and batch requests. Returns `"Default"` / `"No Default"` predictions with probabilities.

---

## Project structure

```
nivuba-ml/
├── backend/
│   ├── app.py          # FastAPI app — all endpoints
│   └── schemas.py      # Pydantic schemas (PipelineInput, NivubaInput, NivubaResponse, ...)
├── src/
│   ├── custom_transformers.py  # scikit-learn custom transformers
│   ├── global_constants.py     # feature label lists, COLUMNS_TO_KEEP
│   ├── pipeline.py             # pipeline definition
│   └── utils.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── Dockerfile
├── start.sh            # GeoIP download (optional) → uvicorn
├── requirements.txt
└── docker-compose.dev.yml
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `MAXMIND_LICENSE_KEY` | No | If set, downloads GeoLite2-Country.mmdb at startup for country-level logging. The service starts fine without it. |

---

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn backend.app:app --port 7860
```

Interactive docs: `http://localhost:7860/docs`

---

## Running with Docker

**Standalone (dev):**
```bash
docker compose -f docker-compose.dev.yml up --build
```

**Full stack (from monorepo root):**
```bash
docker compose up --build
```

The service is ready when `GET /health` returns `{"status": "ok"}`. The model download from Hugging Face Hub takes ~60 seconds on first run; subsequent starts use the Docker layer cache.

---

## Testing

```bash
pip install -r requirements-test.txt
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```
