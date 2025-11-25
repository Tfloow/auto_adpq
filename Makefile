ruff:
	ruff format src/
	ruff check src/ --fix

# Build the package
build:
	python -m build