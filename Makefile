# MiroFish Makefile - Multi-user Productionized Environment Management

.PHONY: help setup setup-backend setup-all dev dev-no-graph backend frontend build test test-focused clean up down logs ps shell-backend shell-frontend verify-gemini verify-ollama verify-keiro

# Default target showing help
help:
	@echo "========================================================================"
	@echo "MiroFish Multi-User Productionized Command Panel"
	@echo "========================================================================"
	@echo "Local Setup & Installation:"
	@echo "  make setup          - Install Node.js dependencies for root & frontend"
	@echo "  make setup-backend  - Sync backend Python dependencies using 'uv'"
	@echo "  make setup-all      - Setup both frontend and backend dependencies"
	@echo ""
	@echo "Docker Compose Commands (Multi-User Local Development):"
	@echo "  make up             - Build and start all services in background (Postgres, Neo4j, Inngest, Backend, Frontend)"
	@echo "  make down           - Stop and remove all containers and networks"
	@echo "  make logs           - View real-time logs from all running containers"
	@echo "  make ps             - List status of all docker-compose managed containers"
	@echo "  make shell-backend  - Access backend container's interactive shell"
	@echo "  make shell-frontend - Access frontend container's interactive shell"
	@echo ""
	@echo "Local Native Development (No Docker):"
	@echo "  make dev            - Run the full dev stack using node script"
	@echo "  make dev-no-graph   - Run frontend & backend natively without Neo4j"
	@echo "  make backend        - Run Flask backend natively"
	@echo "  make frontend       - Run Vite Vue 3 frontend natively"
	@echo "  make build          - Build production static frontend bundle"
	@echo ""
	@echo "Testing & Verification:"
	@echo "  make test           - Run backend unit/integration tests with pytest"
	@echo "  make verify-gemini  - Verify Gemini Vertex AI settings & model config"
	@echo "  make verify-ollama  - Verify local Ollama server connectivity"
	@echo "  make verify-keiro   - Verify Keiro search engine API integration"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove logs, temp files, and artifacts"
	@echo "========================================================================"

# --- Local Setup & Installation ---

setup:
	npm install && cd frontend && npm install

setup-backend:
	cd backend && uv sync

setup-all: setup setup-backend

# --- Docker Compose multi-user stack ---

up:
	docker compose up --build -d
	@echo ""
	@echo "========================================================================"
	@echo "  MiroFish Services Started Successfully!"
	@echo "========================================================================"
	@echo "  Access the components of your stack using the local links below:"
	@echo ""
	@echo "  - Vue 3 Frontend       : http://localhost:3000"
	@echo "  - Flask Backend API    : http://localhost:5001"
	@echo "  - Inngest Dev Server   : http://localhost:8288"
	@echo "  - Neo4j Graph Browser  : http://localhost:7474"
	@echo "  - Postgres Database    : localhost:5432 (mirofish / mirofish)"
	@echo ""
	@echo "  To view logs, run      : make logs"
	@echo "  To stop services, run  : make down"
	@echo "========================================================================"
	@echo ""

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

shell-backend:
	docker compose exec backend sh

shell-frontend:
	docker compose exec frontend sh

# --- Native Development ---

dev:
	npm run dev

dev-no-graph:
	npm run dev:no-graph

backend:
	npm run backend

frontend:
	npm run frontend

build:
	npm run build

# --- Testing & Verification ---

test:
	cd backend && uv run pytest

verify-gemini:
	npm run verify:gemini

verify-ollama:
	npm run verify:ollama

verify-keiro:
	npm run verify:keiro

# --- Cleanup ---

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf backend/app/uploads/temp/*
