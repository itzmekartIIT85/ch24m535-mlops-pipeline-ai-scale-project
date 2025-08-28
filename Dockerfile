# ---------------------------
# Base image
# ---------------------------
FROM python:3.10-slim

# ---------------------------
# Environment variables
# ---------------------------
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV PROCESSED_DIR=/app/data/processed

# ---------------------------
# Set working directory
# ---------------------------
WORKDIR $APP_HOME

# ---------------------------
# Install system dependencies
# ---------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless \
        wget \
        curl \
        git \
        build-essential \
        netcat-openbsd \
        ca-certificates \
        unzip \
        apt-transport-https \
        gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------
# Copy application files
# ---------------------------
COPY requirements.txt .
COPY src/ src/
COPY data/ data/

# ---------------------------
# Install Python dependencies
# ---------------------------
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---------------------------
# Expose port for FastAPI
# ---------------------------
EXPOSE 8000

# ---------------------------
# Command to run app
# ---------------------------
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

