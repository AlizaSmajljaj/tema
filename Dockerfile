FROM python:3.11-slim

# Install GHC via apt (smaller than ghcup for Docker)
RUN apt-get update && apt-get install -y \
    ghc \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project
COPY . .

# Expose port
EXPOSE 8765

# Start the web server
CMD ["python", "-m", "server.main", "--mode", "web"]