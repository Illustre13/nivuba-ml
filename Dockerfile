# Use Python 3.10 slim image based on Linux Debian Bookworm distribution (suitable for ML workloads)
FROM python:3.10-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install curl (to download GeoLite2), tar (to unzip it) & dos2unix (to fix Windows line endings)
RUN apt-get update && apt-get install -y --no-install-recommends curl tar dos2unix && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY ./src ./src
COPY ./backend ./backend
COPY ./frontend ./frontend
COPY ./start.sh .

# Fix start script line endings and make it executable
RUN dos2unix ./start.sh && chmod +x ./start.sh

# Expose Gradio port
EXPOSE 7860  

# Command to run the application
CMD ["./start.sh"]
