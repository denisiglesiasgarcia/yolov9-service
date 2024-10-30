# Base image
FROM python:3.11

# Install system dependencies including GPU support packages
RUN apt-get update && apt-get install --yes \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GPU support
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV TORCH_DEVICE=auto