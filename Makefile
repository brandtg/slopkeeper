# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

# slopkeeper development tasks. Run `make` or `make help` for the target list.

PYTHON ?= python3
ARGS   ?=

# Zero-dep is a RUNTIME rule, not a dev-tools rule: dev tools ride in via
# the `dev` dependency-group in pyproject.toml (`uv sync --group dev`) and
# never touch the project's runtime dependencies.

.DEFAULT_GOAL := help

.PHONY: help
help: ## List available targets.
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: run
run: ## Run the CLI from source; pass ARGS, e.g. make run ARGS="status tmp/docs".
	uv run slopkeeper $(ARGS)

.PHONY: test
test: ## Run the unit suite under pytest.
	uv run pytest -q

.PHONY: cov
cov: ## Run the suite under coverage and report per-module misses.
	uv run pytest -q --cov=slopkeeper --cov-report=term-missing

.PHONY: lint
lint: ## Run ruff (dev-only, never a runtime dependency).
	uv run ruff check --select E,F src tests

.PHONY: fmt
fmt: ## Format all source and tests in place.
	uv run ruff format src tests

.PHONY: fmt-check
fmt-check: ## Fail if any file needs formatting.
	@uv run ruff format --check src tests

.PHONY: check
check: lint test license-check ## Run the canonical gate: lint + tests + license (what CI runs).

.PHONY: license
license: ## Annotate all files with REUSE license headers (MIT).
	uv run reuse annotate \
		--license MIT \
		--copyright "Greg Brandt <brandt.greg@gmail.com>" \
		--skip-unrecognized \
		--recursive .

.PHONY: license-check
license-check: ## Check that all files have REUSE license headers.
	uv run reuse lint

.PHONY: build
build: ## Build the sdist + wheel into dist/.
	uv build

.PHONY: docs
docs: ## Build the docs site strictly (mkdocs-material, dev-only).
	uv run mkdocs build --strict -f docs/site/mkdocs.yml

.PHONY: docs-serve
docs-serve: ## Serve the docs site locally at http://localhost:8000.
	uv run mkdocs serve -f docs/site/mkdocs.yml

.PHONY: install
install: ## Install the package into the active/default environment (plain pip).
	$(PYTHON) -m pip install .

.PHONY: install-pipx
install-pipx: ## Install the package as a pipx tool (the `slopkeeper` binary on $PATH).
	pipx install .

.PHONY: dist-check
dist-check: ## Build sdist+wheel, then verify BOTH install and run via plain pip + pipx on $(PYTHON).
	uv build
	set -e; \
	venv=$$(mktemp -d)/venv; \
	$(PYTHON) -m venv $$venv && \
	$$venv/bin/pip install --quiet dist/*.whl && \
	$$venv/bin/slopkeeper --version && \
	$$venv/bin/slopkeeper strategies show hybrid >/dev/null && echo "wheel: strategies package-data OK"; \
	venv2=$$(mktemp -d)/venv; \
	$(PYTHON) -m venv $$venv2 && \
	$$venv2/bin/pip install --quiet dist/*.tar.gz && \
	$$venv2/bin/slopkeeper --version && echo "sdist: OK"; \
	export PIPX_HOME=$$(mktemp -d) PIPX_BIN_DIR=$$(mktemp -d); \
	pipx install --quiet --force dist/*.whl >/dev/null && \
	$$PIPX_BIN_DIR/slopkeeper --version && echo "pipx: OK"; \
	rm -rf $$(dirname $$venv) $$(dirname $$venv2) "$$PIPX_HOME" "$$PIPX_BIN_DIR"

.PHONY: clean
clean: ## Remove build artifacts and caches.
	rm -rf dist .coverage htmlcov .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +

.PHONY: bump
bump: ## Bump version: make bump part=patch|minor|major
	uv version --bump $(part)
