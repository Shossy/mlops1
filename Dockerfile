# ============================================================
# Multi-stage Dockerfile for House Rent ML pipeline
# Stage 1 (builder): install all Python deps in a full image
# Stage 2 (runtime): slim image with only runtime packages
# ============================================================

# --------------- Stage 1: Builder ---------------
FROM python:3.11 AS builder

WORKDIR /build

COPY requirements.txt .

# --user installs to /root/.local so we can cherry-pick it later
RUN pip install --user --no-cache-dir -r requirements.txt


# --------------- Stage 2: Runtime ---------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Bring only the installed packages from the builder
COPY --from=builder /root/.local /root/.local
ENV PATH="/root/.local/bin:${PATH}"

# Copy project source and config — no raw data, no .git, no venv
COPY src/          src/
COPY config/       config/
COPY params.yaml   dvc.yaml  ./

# Default entrypoint runs the training pipeline; override with docker run args
CMD ["python", "src/train.py", "data/prepared", "data/models"]
