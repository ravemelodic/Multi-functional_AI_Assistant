FROM python:3.12-slim

WORKDIR /comp7940-lab

# Install system dependencies for OCR and image processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python files explicitly
COPY ChatGPT_HKBU.py .
COPY image_to_video.py .
COPY tasks.py .
COPY worker.py .
COPY chatbot_agent.py .

# Copy configuration file
COPY config.ini .

# Create necessary directories
RUN mkdir -p logs temp

# Set environment variable for unbuffered output
ENV PYTHONUNBUFFERED=1
