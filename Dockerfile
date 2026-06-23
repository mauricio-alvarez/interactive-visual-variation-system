FROM node:22-alpine AS frontend

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VISGEN_DEVICE=cpu
ENV VISGEN_DISABLE_SAFETY_CHECKER=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY app ./app
COPY config ./config
COPY docs ./docs
COPY examples ./examples
COPY --from=frontend /frontend/dist ./frontend/dist
COPY scripts ./scripts
COPY README.md .

RUN mkdir -p outputs/sessions models data/processed

EXPOSE 8000

CMD ["python", "scripts/serve.py"]
