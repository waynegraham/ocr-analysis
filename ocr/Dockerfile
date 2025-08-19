FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Set known location for NLTK data
ENV NLTK_DATA=/usr/local/nltk_data

# Set working directory
WORKDIR /app

# Copy everything in
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# ✅ Pre-download NLTK data
RUN python -m nltk.downloader -d $NLTK_DATA punkt punkt_tab

# Default command
CMD ["python", "runner.py"]