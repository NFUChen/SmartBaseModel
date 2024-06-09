# Makefile for formatting Python code using ruff and isort

.PHONY: format

# Default target
format: ruff isort

# Run isort for sorting imports
isort:
	python3 -m isort .
# Run ruff for formatting
ruff:
	ruff format .