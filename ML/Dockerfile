FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY ML/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] gunicorn python-multipart

# Copy project
COPY ML /app/ML

EXPOSE 8000

CMD ["uvicorn", "ML.server:app", "--host", "0.0.0.0", "--port", "8000"]


