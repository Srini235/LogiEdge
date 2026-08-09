# Base image strictly per rubric
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies BEFORE copying the model or code
RUN pip install --no-cache-dir -r requirements.txt

# Copy the normalization stats
COPY training_stats.npy .

# Copy the TFLite model (This layer breaks cache when a new model is pushed)
COPY anomaly_model.tflite .

# Copy the inference microservice code
COPY inference_service.py .

# Accept environment variable for model path (Defaults to anomaly_model.tflite)
ENV MODEL_PATH="anomaly_model.tflite"

# Create the external data directory expected by the SQLite logger
RUN mkdir -p /data_pipeline

# Command to run the inference service
CMD ["python", "inference_service.py"]
