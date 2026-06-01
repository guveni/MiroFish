FROM python:3.11-slim

# Install system dependencies for psycopg2 and file parsing
RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc libpq-dev python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

# Set working directory to /app
WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY backend/pyproject.toml backend/uv.lock ./

# Sync backend dependencies
RUN uv sync --frozen

# Copy locales to /locales (since backend code resolves to /locales)
COPY locales /locales

# Copy backend source
COPY backend /app

EXPOSE 5001

CMD ["uv", "run", "gunicorn", "-b", "0.0.0.0:5001", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "run:app"]
