FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy folders
COPY app/ app/
COPY model/ model/
COPY frontend/ frontend/

# Env vars for Pathlib to resolve
ENV MODEL_PATH=/app/model/model.onnx
ENV METADATA_PATH=/app/model/metadata.json

EXPOSE 8000

# Set Python path to find app module
ENV PYTHONPATH=/app

CMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]
