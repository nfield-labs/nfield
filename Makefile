.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test all docs build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install all extras and dev tooling
	uv sync --all-extras --dev

lint: ## Check lint and formatting (no changes)
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format: ## Auto-fix lint and apply formatting
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

typecheck: ## Run mypy in strict mode
	uv run mypy src/ --strict

test: ## Run the unit test suite
	uv run pytest tests/unit/

all: lint typecheck test ## Lint, type-check, and test

docs: ## Serve the documentation site locally
	uv run --group docs mkdocs serve

build: ## Build the wheel and sdist
	uv build

clean: ## Remove build artifacts and tool caches
	rm -rf dist build ./*.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov site
