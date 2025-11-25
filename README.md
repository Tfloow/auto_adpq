# auto_adpq

Adaptive Post-Training Quantization tooling (replicating AdpQ)

This repository implements tools and reference code to reproduce the ideas from
[AdpQ: A Zero-shot Calibration Free Adaptive Post Training Quantization Method for LLMs](https://arxiv.org/abs/2405.13358).

This README explains how to install, run tests, build documentation (including
multi-version docs), and contribute.

- [auto\_adpq](#auto_adpq)
  - [Installation](#installation)
  - [Quick usage](#quick-usage)
  - [Running tests \& linters](#running-tests--linters)
  - [Building the documentation](#building-the-documentation)
  - [Contributing](#contributing)
  - [Development notes](#development-notes)
  - [License](#license)

## Installation

Install from PyPI (recommended):

```powershell
python -m pip install auto_adpq
```

Install the latest development version directly from GitHub:

```powershell
python -m pip install "git+https://github.com/Tfloow/auto_adpq.git"
```

To develop locally (editable install):

```powershell
git clone https://github.com/Tfloow/auto_adpq.git
cd auto_adpq
python -m pip install -e .
```

Makefile helper:

```powershell
# Run formatting, linting, coverage and docs targets as defined in Makefile
make
```

## Quick usage

Import the package and use the public API. Example (replace with real API):

```python
from auto_adpq import Auto_AdpQ
```

Add a short usage snippet here specific to the package functions you expect
users to try first.

## Running tests & linters

Coverage test: **66%**

- Run tests with pytest:

```powershell
pytest -q
```

- Run full coverage report (Makefile target):

```powershell
make coverage
```

- Format & lint with `ruff` (Makefile target):

```powershell
make ruff
```

## Building the documentation

This project uses Sphinx for documentation. There are two common workflows:

- Build a single-version site (useful for local writing and previews):

```powershell
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

- Build a multi-version site using `sphinx-multiversion` (we configure this in
	`docs/conf.py`). This produces one static site containing each built branch
	and tag (useful for publishing versioned docs with a dropdown selector):

```powershell
python -m pip install -r docs/requirements.txt
sphinx-multiversion docs docs/_build/html-mv
```

Notes about versions
- The project includes a small template `docs/_templates/versions.html` which
	renders a versions dropdown when the site is built with `sphinx-multiversion`.
- Adjust `smv_tag_whitelist` and `smv_branch_whitelist` in `docs/conf.py` to
	control which tags/branches are included in the build.

## Contributing

Contributions are welcome. A suggested workflow:

1. Fork the repository and create a feature branch.
2. Add tests for new functionality.
3. Run `ruff` to format and lint.
4. Open a pull request describing the change.

Please include unit tests and keep the public API stable when possible.

## Development notes

- Docs templates: `docs/_templates/versions.html` — version switcher used by
	`sphinx-multiversion`.
- Makefile targets: `make ruff`, `make coverage`, `make docs` (runs single and
	multiversion builds).

## License

This work is under [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

