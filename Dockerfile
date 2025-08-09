# Multi-stage Dockerfile for Speaking Feedback Tool
# Supports both CPU and GPU deployments

# Base image with Python and CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/zoom_recordings /app/models/custom /app/logs

# Set permissions
RUN chmod +x /app/app.py

# Expose ports
EXPOSE 5001 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Default command
CMD ["python3", "app.py"]

# Development stage (optional)
FROM base as development

# Install development dependencies
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Development command
CMD ["python3", "app.py", "--debug"]

# Production stage
FROM base as production

# Install production dependencies
RUN pip3 install --no-cache-dir gunicorn

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "app:app"]

# GPU stage with Triton support
FROM base as gpu

# Install GPU-specific dependencies
RUN pip3 install --no-cache-dir \
    tritonclient[grpc] \
    tritonclient[http] \
    nemo-toolkit[asr] \
    torch-tensorrt

# GPU command
CMD ["python3", "app.py", "--gpu"] 