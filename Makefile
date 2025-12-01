all: ruff coverage docs build

RUFF_TO_CHECK= src/ tests/ examples/

ruff:
	@ruff format $(RUFF_TO_CHECK)
	@ruff check $(RUFF_TO_CHECK) --fix

# Build the package
build:
	@python -m build

test:
	@pytest -q -m "not slow"

coverage:
	@coverage run -m pytest -q -m "not slow"
	@coverage report -m
	@coverage html

profiling:
	@python -m cProfile -o profile.out examples/quantization_for_profiling.py
	@snakeviz profile.out

.PHONY: docs
docs:
	@python -m sphinx -b html docs docs/_build/html

.PHONY: docs-mv
docs-mv: docs
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