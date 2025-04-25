FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    httpx \
    redis \
    beautifulsoup4 \
    requests \
    aiohttp \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-requests \
    opentelemetry-instrumentation-redis \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-instrumentation-httpx \
    opentelemetry-instrumentation-urllib3 \
    ujson

# Copy shared package
COPY shared /shared
RUN pip install /shared

# Copy backend modules
COPY app/backend /app/backend

# Copy the service code
COPY research_service.py ./

EXPOSE 8005

CMD ["uvicorn", "research_service:app", "--host", "0.0.0.0", "--port", "8005"]
