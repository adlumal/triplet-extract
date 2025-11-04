#!/bin/bash
set -e

echo "Running Black formatting check..."
black --check --diff triplet_extract/ tests/ examples/

echo ""
echo "Running Ruff linting..."
ruff check triplet_extract/ tests/ examples/

echo ""
echo "Running tests..."
pytest tests/ -v

echo ""
echo "All checks passed!"
