# Use RunPod's recommended CUDA base image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY runpod_handler.py .

# Create cache directory for Hugging Face models
RUN mkdir -p /root/.cache/huggingface

# Pre-download the model (optional, for faster startup)
# RUN python3 -c "from transformers import AutoModel, AutoProcessor; AutoModel.from_pretrained('google/siglip2-so400m-patch14-384'); AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the RunPod handler
CMD ["python3", "runpod_handler.py"]