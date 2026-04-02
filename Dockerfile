# Multi-stage ML image: CI `docker build` smoke test and optional batch runs (not Airflow scheduler).
# syntax=docker/dockerfile:1

FROM python:3.11-bookworm AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY config ./config
COPY params.yaml dvc.yaml ./
COPY dags ./dags
RUN chown -R app:app /app
USER app
CMD ["python", "-c", "import sklearn; print('ok')"]
