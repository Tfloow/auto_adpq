ruff:
	ruff format src/
	ruff check src/ --fix

# Build the package
build:
	python -m build

coverage:
	coverage run -m pytest
	coverage report -m
	coverage html