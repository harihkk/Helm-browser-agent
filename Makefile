.PHONY: help install dev test run clean lint reset-db

PYTHON ?= python3
VENV   ?= venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

help:
	@echo "Targets:"
	@echo "  make install    Create venv and install dependencies"
	@echo "  make dev        Install + Playwright browsers"
	@echo "  make run        Start the agent server on http://localhost:8000"
	@echo "  make test       Run smoke tests (no network)"
	@echo "  make lint       Quick syntax check across the codebase"
	@echo "  make reset-db   Wipe agent_memory.db (recreated on next run)"
	@echo "  make clean      Remove caches and screenshots/"

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dev: install
	$(PY) -m playwright install chromium

run:
	$(PY) run.py

test:
	$(PY) tests/unit/test_smoke.py

lint:
	@$(PY) -c "import ast, pathlib, sys; \
files = [p for p in pathlib.Path('.').rglob('*.py') if 'venv' not in p.parts]; \
[ast.parse(p.read_text(), str(p)) for p in files]; \
print(f'OK: {len(files)} files parse')"

reset-db:
	rm -f agent_memory.db

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf screenshots/*
