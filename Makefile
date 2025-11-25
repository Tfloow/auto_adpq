all: ruff coverage docs build

ruff:
	@ruff format src/
	@ruff check src/ --fix

# Build the package
build:
	@python -m build

coverage:
	@coverage run -m pytest
	@coverage report -m
	@coverage html

.PHONY: docs
docs:
	@python -m sphinx -b html docs docs/_build/html