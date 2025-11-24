ruff:
    ruff format --check src/
    ruff check src/ --fix

# Build the package
build:
    python -m build