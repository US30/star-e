# StAR-E Dockerfile
# Multi-stage build for optimized image size

FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --wheel-dir /wheels .

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Copy source code
COPY src/ src/
COPY dashboard/ dashboard/

# Create data directory
RUN mkdir -p data/raw data/processed mlruns

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STAR_E_DATA_DIR=/app/data
ENV MLFLOW_TRACKING_URI=/app/mlruns

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["uvicorn", "star_e.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
