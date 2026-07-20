FROM python:3.12-slim

WORKDIR /comp7940-lab

# Install system dependencies for OCR, image processing, and TLS cert generation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Generate self-signed SSL certificate for HTTPS admin panel
# CN can be overridden at build time: docker build --build-arg CERT_CN=192.168.1.100
ARG CERT_CN=localhost
RUN mkdir -p /etc/ssl/private /etc/ssl/certs && \
    openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
    -keyout /etc/ssl/private/key.pem \
    -out /etc/ssl/certs/cert.pem \
    -subj "/CN=${CERT_CN}/O=Internal/C=XX" && \
    chmod 600 /etc/ssl/private/key.pem

# Create non-root user for running the container
RUN useradd -m -u 1000 bot

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application packages
COPY app/ app/
COPY workers/ workers/

# Create writable directories (mounted as volumes at runtime)
RUN mkdir -p logs temp && chown -R bot:bot logs temp

# Non-root user is set via docker-compose user: directive;
# the useradd above ensures the user exists.

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1
