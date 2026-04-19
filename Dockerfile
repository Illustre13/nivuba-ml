FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl tar dos2unix && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./backend ./backend
COPY ./start.sh .

RUN dos2unix ./start.sh && chmod +x ./start.sh

EXPOSE 7860

CMD ["./start.sh"]
