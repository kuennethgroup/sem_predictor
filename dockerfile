FROM python:3.11-slim

# Working directory inside the container
WORKDIR /app

# Update system dependencies and upgrade critical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    && apt-get upgrade -y libexpat1 libssl3 libffi8 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY ./ /app/

# Optional: run Streamlit as a non-root user (recommended for security)
# RUN useradd -m appuser && chown -R appuser /app
# USER appuser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck for Streamlit app
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
