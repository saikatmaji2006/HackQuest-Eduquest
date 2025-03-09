<<<<<<< HEAD
=======
# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

>>>>>>> f41f548 (frontend)
FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
<<<<<<< HEAD
    FLASK_ENV=production \
    PORT=5000
=======
    FLASK_ENV=development \
    FLASK_DEBUG=1 \
    LEETCODE_SESSION="" \
    CSRF_TOKEN=""
>>>>>>> f41f548 (frontend)

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
<<<<<<< HEAD
        gcc \
        python3-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs && \
    chmod -R 777 logs
=======
    gcc \
    python3-dev \
    curl \
    git && \
    rm -rf /var/lib/apt/lists/*

# Set up non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy and install requirements
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Install LeetCode API package
RUN pip install --user --no-cache-dir leetcode-api

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p logs
RUN mkdir -p data/leetcode
>>>>>>> f41f548 (frontend)

# Expose port
EXPOSE 5000

<<<<<<< HEAD
# Set the entrypoint to use gunicorn
ENTRYPOINT ["gunicorn"]

# Configure the CMD to set gunicorn parameters
CMD ["--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--log-level", "info", "app:app"]
=======
# Start the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--log-level", "info", "app:app"]
>>>>>>> f41f548 (frontend)
