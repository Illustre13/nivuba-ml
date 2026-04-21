FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./backend ./backend
COPY ./models ./models

EXPOSE 7860

CMD ["sh", "-c", "mkdir -p /app/geoip_db && exec uvicorn backend.app:app --host 0.0.0.0 --port 7860"]
