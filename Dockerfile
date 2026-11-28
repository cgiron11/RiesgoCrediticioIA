FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY agent/ ./agent/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
