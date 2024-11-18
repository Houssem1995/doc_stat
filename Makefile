.PHONY: help setup test clean lint format run docker-build docker-run

# Python interpreter to use
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
STREAMLIT := $(VENV)/bin/streamlit
PYLINT := $(VENV)/bin/pylint
BLACK := $(VENV)/bin/black

# Project directories
SRC_DIR := src
TEST_DIR := tests
BUILD_DIR := build
DIST_DIR := dist

help:
	@echo "Available commands:"
	@echo "make setup      - Create virtual environment and install dependencies"
	@echo "make test      - Run tests"
	@echo "make lint      - Run linter"
	@echo "make format    - Format code using Black"
	@echo "make clean     - Remove build artifacts and cache files"
	@echo "make run       - Run the Streamlit app"
	@echo "make build     - Build the project"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-run   - Run Docker container"

setup: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

test: setup
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

lint: setup
	$(PYLINT) $(SRC_DIR) $(TEST_DIR)

format: setup
	$(BLACK) $(SRC_DIR) $(TEST_DIR)

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(DIST_DIR)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

run: setup
	$(STREAMLIT) run app.py

build: clean setup test
	$(PYTHON) setup.py sdist bdist_wheel

docker-build:
	docker build -t docstat .

docker-run:
	docker run -p 8501:8501 docstat

# Development dependencies
dev-setup: setup
	$(PIP) install -r requirements-dev.txt

# Run type checking
type-check: setup
	$(VENV)/bin/mypy $(SRC_DIR)

# Generate documentation
docs: setup
	$(VENV)/bin/sphinx-build -b html docs/source docs/build

# Security check
security-check: setup
	$(VENV)/bin/safety check

# Create requirements.txt from setup.py
requirements:
	$(PIP) freeze > requirements.txt

# Run all quality checks
quality-check: lint type-check security-check test

# Install pre-commit hooks
pre-commit: setup
	$(VENV)/bin/pre-commit install

# Update dependencies
update-deps: setup
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) freeze > requirements.txt

# Create a new release
release: quality-check
	$(PYTHON) scripts/create_release.py

# Database migrations (if using a database)
db-migrate: setup
	$(PYTHON) scripts/db_migrate.py

# Backup data (if applicable)
backup:
	$(PYTHON) scripts/backup.py

# Restore data (if applicable)
restore:
	$(PYTHON) scripts/restore.py

# Run development server with hot reload
dev: setup
	$(STREAMLIT) run app.py --server.runOnSave=true

# Profile the application
profile: setup
	$(PYTHON) -m cProfile -o profile.stats app.py
	$(PYTHON) scripts/analyze_profile.py

# Generate test coverage report
coverage: test
	$(VENV)/bin/coverage html
	@echo "Coverage report generated in htmlcov/index.html" 