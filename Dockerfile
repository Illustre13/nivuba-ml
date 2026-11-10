FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./backend ./backend

# Bake the trained pipeline into the image so the container starts offline.
# `models/` is git-ignored, so it is fetched from Hugging Face Hub at build time
# rather than copied from the build context. Override MODEL_* to pin another artifact.
ARG MODEL_REPO_ID=JensBender/loan-default-prediction-pipeline
ARG MODEL_FILENAME=loan_default_rf_pipeline.joblib
ARG MODEL_REVISION=v1.0
RUN mkdir -p models geoip_db && python -c "\
import shutil; from huggingface_hub import hf_hub_download; \
p = hf_hub_download(repo_id='${MODEL_REPO_ID}', filename='${MODEL_FILENAME}', revision='${MODEL_REVISION}'); \
shutil.copy(p, 'models/${MODEL_FILENAME}')"

ENV PIPELINE_LOCAL_PATH=/app/models/${MODEL_FILENAME}

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
