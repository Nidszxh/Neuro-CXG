# Neuro-CXG Dockerfile
# Build:  docker build -t neuro-cxg .
# Run:    docker run --gpus all \
#           -v /path/to/data:/workspace/data \
#           -v /path/to/results:/workspace/results \
#           neuro-cxg

FROM nvidia/cuda:12.1.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

# Create a non-root user for runtime security
RUN useradd -m -u 1000 neuro

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git wget curl \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /workspace/
RUN chown -R neuro:neuro /workspace

USER neuro

# Verify environment loads cleanly
HEALTHCHECK --interval=60s --timeout=30s --retries=1 \
    CMD python3 -c "from src.core.config import validate_environment; validate_environment()" || exit 1

CMD ["python3", "src/run_pipeline.py", "--auto"]