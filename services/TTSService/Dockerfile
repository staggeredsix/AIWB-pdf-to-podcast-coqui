FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=0

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ffmpeg libsndfile1 sox \
        git curl unzip build-essential \
        libsox-dev libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install TorToiSe requirements
RUN pip install \
    ujson \
    fastapi \
    uvicorn \
    redis \
    httpx \
    soundfile \
    numpy \
    scipy \
    librosa \
    transformers \
    tokenizers \
    inflect \
    progressbar2 \
    einops \
    rotary_embedding_torch \
    k_diffusion \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    opentelemetry-instrumentation \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-redis \
    opentelemetry-instrumentation-httpx \
    opentelemetry-instrumentation-requests \
    opentelemetry-instrumentation-urllib3 \
    opentelemetry-util-http

# Install TorToiSe TTS
RUN pip install TorToiSe

# Add shared package
COPY shared /shared
RUN pip install /shared

# Copy service code
WORKDIR /app
COPY services/TTSService /app

# Expose port
EXPOSE 8889

# Start the TTS service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8889"]
