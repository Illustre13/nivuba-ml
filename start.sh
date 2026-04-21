#!/bin/sh
set -e

mkdir -p /app/geoip_db

echo "Starting Nivuba ML service with local model fallback if Hugging Face is unavailable."
exec uvicorn backend.app:app --host 0.0.0.0 --port 7860
