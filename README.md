[![Docs](https://img.shields.io/badge/Docs-latest-blue?logo=readme&logoColor=f5f5f5)](https://auto-adpq.readthedocs.io/en/latest/) [![Paper](https://img.shields.io/badge/Paper-2405.13358-B31B1B?logo=arxiv&logoColor=f5f5f5)](https://arxiv.org/abs/2405.13358) [![PyPI](https://img.shields.io/badge/PyPi-0.3.5-3775A9?logo=pypi&logoColor=f5f5f5)](https://pypi.org/project/auto-adpq/)  [![HF](https://img.shields.io/badge/HuggingFace-collection-FFD21E?logo=huggingface&logoColor=f5f5f5)](https://huggingface.co/collections/Tfloow/adpq) [![CI](https://github.com/Tfloow/auto_adpq/actions/workflows/ci.yml/badge.svg)](https://github.com/Tfloow/auto_adpq/actions/workflows/ci.yml) [![Build and Release](https://github.com/Tfloow/auto_adpq/actions/workflows/release.yml/badge.svg)](https://github.com/Tfloow/auto_adpq/actions/workflows/release.yml) 

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
    - [Debug mode](#debug-mode)
  - [Documentation](#documentation)
    - [Building the documentation](#building-the-documentation)
  - [Tasklist](#tasklist)
  - [Quantized models](#quantized-models)
    - [Performances](#performances)
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

The most simple way to quantize a model is to follow a similar script as in [examples/simple_quantization.py](examples/simple_quantization.py).

## Running tests & linters

Coverage test: **91%**

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

### Debug mode

To obtain logs of the package, it is possible to enable the logging module. To activate it please create the new environment variable `AUTO_ADPQ_DEBUG` by running:

```powershell
# Linux
export AUTO_ADPQ_DEBUG=1

# Windows
$Env:AUTO_ADPQ_DEBUG = 1
```

## Documentation

The documentation can be found [here](https://auto-adpq.readthedocs.io/en/latest/).

### Building the documentation

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

## Tasklist

- [x] Solve the datapacking issue #1
- [ ] Support efficient inference (maybe wrap in SpQR?)
- [ ] Optimize pydantic module `AdpQQuantizedWeights`
  - Currently, there is a major overhead when creating a new object to validate the field. Since it is used internally only, we could ditch the Pydantic module but would need to ensure proper dump and load function
- [ ] Support model and integrate with `.safetensors`

## Quantized models

Pre-quantized models are available in this [collection](https://huggingface.co/collections/Tfloow/adpq). They are *simulated* models meaning they are stored as `bf16` values instead of the quantized versions. If I stored them in the custom format, I would either need an algorithm to reconstruct the weights in full at runtime or develop a custom CUDA kernel, which is quite tough.

Nonetheless, those models represent the quality and rounding errors that a typical quantized model can meet.

### Performances

![Current performance](examples/figs/ppl_vs_effective_bits.png)

___

<table>
    <thead>
        <tr>
            <th width="40%">Model Variant</th>
            <th width="30%">Quantization Method</th>
            <th width="30%">PPL (Perplexity)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3"><strong>meta-llama/Llama-3.1-8B</strong></td>
            <td>Baseline</td>
            <td>4.8693</td>
        </tr>
        <tr>
            <td>BNB</td>
            <td>5.0733</td>
        </tr>
        <tr>
            <td><strong>AdpQ</strong></td>
            <td><strong>5.3671</strong></td>
        </tr>
        <tr>
            <td rowspan="5"><strong>meta-llama/Llama-3.1-8B-Instruct</strong></td>
            <td>Baseline</td>
            <td>4.9080</td>
        </tr>
        <tr>
            <td>BNB</td>
            <td>4.9993</td>
        </tr>
        <tr>
            <td><strong>AdpQ</strong></td>
            <td><strong>5.0069</strong></td>
        </tr>
        <tr>
            <td>AWQ</td>
            <td>5.0440</td>
        </tr>
         <tr>
            <td>GPTQ</td>
            <td>nan</td>
        </tr>
        <tr>
            <td rowspan="4"><strong>meta-llama/Llama-3.2-1B</strong></td>
            <td>Baseline</td>
            <td>6.5546</td>
        </tr>
        <tr>
            <td><strong>AdpQ 9%</strong></td>
            <td><strong>6.9491</strong></td>
        </tr>
        <tr>
            <td>BNB</td>
            <td>6.9971</td>
        </tr>
        <tr>
            <td><strong>AdpQ 2%</strong></td>
            <td><strong>7.0380</strong></td>
        </tr>
        <tr>
            <td rowspan="3"><strong>meta-llama/Llama-3.2-3B-Instruct</strong></td>
            <td>Baseline</td>
            <td>5.7864</td>
        </tr>
        <tr>
            <td>AWQ</td>
            <td>5.8339</td>
        </tr>
        <tr>
            <td><strong>AdpQ</strong></td>
            <td><strong>5.9040</strong></td>
        </tr>
    </tbody>
</table>

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

