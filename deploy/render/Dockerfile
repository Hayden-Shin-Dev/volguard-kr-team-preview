FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONPATH=/app/src VOLGUARD_PROJECT_DIR=/app PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
COPY pyproject.toml ./
COPY src/ ./src/
COPY deploy/render/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps .
COPY --from=frontend /build/dist ./frontend/dist
COPY scripts/public-preview.js ./scripts/public-preview.js
COPY deploy/seed/ ./deploy/seed/
CMD ["python", "-m", "volguard.cloud_preview"]
