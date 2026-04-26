# Neuro-CXG Dockerfile
# Build: docker build -t neuro-cxg .
# Run: docker run --gpus all -v /path/to/data:/data neuro-cxg

FROM nvidia/cuda:12.1.0-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git wget curl libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /workspace/

CMD ["python", "src/run_pipeline.py", "--auto"]