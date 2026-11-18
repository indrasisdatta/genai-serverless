# Policy Agent - Makefile

.PHONY: help install test eval-smoke eval-full lint format docker-up docker-down

help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run unit tests"
	@echo "  eval-smoke    - Run smoke evals with gate"
	@echo "  eval-full     - Run full evals with gate"
	@echo "  lint          - Run linters"
	@echo "  format        - Format code with ruff/black"
	@echo "  docker-up     - Start PostgreSQL with pgvector"
	@echo "  docker-down   - Stop PostgreSQL"
	@echo "  generate-data - Generate golden and smoke datasets"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

eval-smoke:
	python -m evals.run_evals --dataset evals/smoke.jsonl --gate

eval-full:
	python -m evals.run_evals --dataset evals/golden.jsonl --gate

lint:
	ruff check .
	mypy app/

format:
	ruff check --fix .
	black .

docker-up:
	docker compose up -d db

docker-down:
	docker compose down

generate-data:
	python evals/generate_golden.py
	python evals/generate_smoke.py evals/golden.jsonl evals/smoke.jsonl

list-models:
	python scripts/list_models.py

ask:
	python -m scripts.ask $(filter-out $@,$(MAKECMDGOALS))

# Allow passing args to ask
%:
	@: