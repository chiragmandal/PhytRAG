# PhytRAG Makefile
# All commands assume you are at the project root.
# Ollama must be running natively on macOS before running `make up`.

.DEFAULT_GOAL := help
.PHONY: help setup up down logs ingest eval test lint clean ui

# ── Colours for terminal output ──────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m

help: ## Show this help message
	@echo ""
	@echo "$(BOLD)PhytRAG - Plant Biology RAG Service$(RESET)"
	@echo ""
	@echo "$(CYAN)First-time setup$(RESET)"
	@echo "  make setup    Pull Ollama model (one-time, ~2 GB)"
	@echo ""
	@echo "$(CYAN)Daily workflow$(RESET)"
	@echo "  make up       Start all services (api, qdrant, prometheus, grafana, mlflow, ui)"
	@echo "  make ingest   Download papers from PMC OA and index into Qdrant"
	@echo "  make query Q=\"What is GA20ox?\"   Run a test query"
	@echo "  make ui       Open the Streamlit UI in the browser"
	@echo "  make down     Stop all services"
	@echo "  make logs     Tail container logs"
	@echo ""
	@echo "$(CYAN)Development$(RESET)"
	@echo "  make test     Run unit tests with coverage"
	@echo "  make lint     Run ruff linter"
	@echo "  make eval     Run retrieval evaluation (logs to MLflow)"
	@echo "  make clean    Remove Docker volumes (WARNING: deletes all indexed data)"
	@echo ""

# ── First-time setup ──────────────────────────────────────────────────────────

setup: ## Pull the Ollama LLM model (llama3.2:3b, ~2 GB, one-time)
	@echo "$(BOLD)Pulling Ollama model: llama3.2:3b$(RESET)"
	@echo "This downloads ~2 GB. It is a one-time operation."
	ollama pull llama3.2:3b
	@echo "$(GREEN)Model ready.$(RESET)"

# ── Service lifecycle ─────────────────────────────────────────────────────────

up: ## Start all services in the background
	@echo "$(BOLD)Starting PhytRAG services...$(RESET)"
	@echo "Checking Ollama is running..."
	@curl -sf http://localhost:11434/api/tags > /dev/null || \
		(echo "ERROR: Ollama is not running. Start it with: ollama serve" && exit 1)
	docker compose up -d --build
	@echo ""
	@echo "$(GREEN)Services started:$(RESET)"
	@echo "  UI (Streamlit): http://localhost:8501"
	@echo "  API docs:       http://localhost:8000/docs"
	@echo "  Qdrant UI:      http://localhost:6333/dashboard"
	@echo "  Grafana:        http://localhost:3001  (admin/admin)"
	@echo "  MLflow:         http://localhost:5002"
	@echo "  Prometheus:     http://localhost:9091"
	@echo ""
	@echo "Next: run 'make ingest' to index the plant biology corpus."

down: ## Stop all services
	docker compose down

logs: ## Tail logs from all containers
	docker compose logs -f --tail=50

# ── Data pipeline ─────────────────────────────────────────────────────────────

ingest: ## Download PMC papers and index into Qdrant (run once after `make up`)
	@echo "$(BOLD)Step 1: Downloading plant biology papers from PMC OA...$(RESET)"
	python -m ingestion.download_corpus
	@echo ""
	@echo "$(BOLD)Step 2: Chunking, embedding, and indexing into Qdrant...$(RESET)"
	python -m ingestion.chunk_and_embed
	@echo ""
	@echo "$(GREEN)Ingestion complete. Run 'make query Q=\"Your question\"' to test.$(RESET)"

# ── Query test ────────────────────────────────────────────────────────────────

Q ?= What is the role of GA20ox in Arabidopsis stem elongation?
query: ## Run a test query. Override with: make query Q="Your question"
	@echo "$(BOLD)Query: $(Q)$(RESET)"
	@curl -s -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d "{\"q\": \"$(Q)\"}" | python3 -m json.tool

ui: ## Open the Streamlit UI (starts the container if not already running)
	@echo "$(BOLD)Streamlit UI: http://localhost:8501$(RESET)"
	@docker-compose up -d ui
	@open http://localhost:8501 2>/dev/null || xdg-open http://localhost:8501 2>/dev/null || true

# ── Evaluation ────────────────────────────────────────────────────────────────

eval: ## Run retrieval evaluation and log results to MLflow
	@echo "$(BOLD)Running retrieval evaluation (results logged to MLflow)...$(RESET)"
	python -m eval.retrieval_eval

# ── Development ───────────────────────────────────────────────────────────────

test: ## Run unit tests with coverage report
	pytest -v --cov=app --cov=ingestion --cov-report=term-missing

lint: ## Run ruff linter and formatter check
	ruff check .
	ruff format --check .

lint-fix: ## Auto-fix ruff lint issues
	ruff check --fix .
	ruff format .

# ── Maintenance ───────────────────────────────────────────────────────────────

clean: ## Remove Docker volumes (deletes ALL indexed Qdrant data and MLflow runs)
	@echo "$(BOLD)WARNING: This will delete all Qdrant data and MLflow runs.$(RESET)"
	@read -p "Are you sure? [y/N] " REPLY; \
		[ "$$REPLY" = "y" ] && docker compose down -v && echo "Volumes removed." || echo "Aborted."
