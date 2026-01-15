# Multi-stage build for RAG application
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser (used by cab.py and crawl4ai)
RUN python -m playwright install-deps chromium || true
RUN python -m playwright install chromium

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p files logs vector_store

# Make startup script executable
COPY startup.sh .
RUN chmod +x startup.sh

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run the startup script with bash (script uses bash-specific syntax)
CMD ["bash", "startup.sh"]