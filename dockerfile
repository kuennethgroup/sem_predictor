FROM python:3.11-slim

# Arbeitsverzeichnis
WORKDIR /home/lukas/Masterthesis/app

# Systemabhängigkeiten aktualisieren und installieren + manuelles Upgrade kritischer Bibliotheken
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    && apt-get upgrade -y libexpat1 libssl3 libffi8  \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies kopieren und installieren
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App-Code kopieren
COPY ./ ./

# Exponiere den Standardport für Streamlit
EXPOSE 8501

# Gesundheitscheck der Streamlit-App
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Führe Streamlit als nicht-root User aus (optional)
# RUN useradd -m appuser && chown -R appuser /home/lukas/Masterthesis/app
# USER appuser

# Starte Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
