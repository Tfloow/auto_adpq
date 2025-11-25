all: ruff coverage docs build

ruff:
	@ruff format src/
	@ruff check src/ --fix

# Build the package
build:
	@python -m build

coverage:
	@coverage run -m pytest -m "not slow"
	@coverage report -m
	@coverage html

.PHONY: docs
docs:
	@python -m sphinx -b html docs docs/_build/html
	@sphinx-multiversion docs docs/_build/html-mv

clean:
	@rm -rf dist/
	@rm -rf build/
	@rm -rf src/auto_adpq.egg-info/
	@rm -rf docs/_build/
	@rm -rf .ruff_cache/
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -rf .mypy_cache/
	@rm -rf .coverage
	@rm -rf coverage.xml
	@rm -rf src/auto_adpq/__pycache__/
	@rm -rf src/auto_adpq/tests/__pycache__/