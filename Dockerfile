# Stage 1: Build the Vue 3 Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Production Python Backend serving the static frontend
FROM python:3.11-slim
RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc libpq-dev python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /app

# Copy python dependency manifest first for layer caching
COPY backend/pyproject.toml backend/uv.lock ./backend/
RUN cd backend && uv sync --frozen

# Copy backend app source and copy built frontend dist
COPY backend/ ./backend/
COPY locales/ ./locales/
COPY --from=frontend-builder /frontend/dist ./frontend/dist

WORKDIR /app/backend
EXPOSE 5001

CMD ["uv", "run", "uvicorn", "run:app", "--host", "0.0.0.0", "--port", "5001"]
