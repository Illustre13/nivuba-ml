#!/bin/bash
set -e  # exit immediately if any command fails

# Create geoip_db directory (if it doesn't exist already)
mkdir -p /app/geoip_db

# Download GeoLite2-Country.mmdb only when a license key is provided and the file doesn't exist yet
# If the key is absent the app still starts — client country will not be logged (handled gracefully in app.py)
if [ ! -f /app/geoip_db/GeoLite2-Country.mmdb ]; then
  MAXMIND_LICENSE_KEY=$(echo "$MAXMIND_LICENSE_KEY" | tr -d '"' | tr -d "'" | tr -d '[:space:]')

  if [ -n "$MAXMIND_LICENSE_KEY" ]; then
    echo "Downloading GeoLite2-Country.mmdb..."
    curl -L -o /app/geoip_db/GeoLite2-Country.tar.gz \
      "https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-Country&license_key=${MAXMIND_LICENSE_KEY}&suffix=tar.gz"
    tar -xzf /app/geoip_db/GeoLite2-Country.tar.gz -C /app/geoip_db --strip-components=1
    rm /app/geoip_db/GeoLite2-Country.tar.gz
    echo "Successfully downloaded GeoLite2-Country.mmdb."
  else
    echo "MAXMIND_LICENSE_KEY not set — skipping GeoIP download. Client country will not be logged."
  fi
fi

# Start the FastAPI app
uvicorn backend.app:app --host 0.0.0.0 --port 7860
