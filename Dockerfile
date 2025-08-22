FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directories
RUN mkdir -p /runpod-volume/.cache/huggingface
RUN mkdir -p /tmp/.cache/huggingface

# Set permissions
RUN chmod +x runpod_handler.py

# Expose port (for local testing)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=30s --retries=3 --start-period=120s \
    CMD python -c "import runpod_handler; print('Health check passed')" || exit 1

# Start the RunPod handler
CMD ["python", "runpod_handler.py"]