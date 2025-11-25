Build instructions (PowerShell)
-------------------------------

1. Create and activate a virtual environment

```powershell
python -m venv .venv
# then
.\.venv\Scripts\Activate.ps1
```

2. Install documentation dependencies

```powershell
pip install -r docs/requirements.txt
```

3. Build the HTML docs

```powershell
python -m sphinx -b html docs docs/_build/html
```

4. Open the generated docs in your browser

```powershell
start docs\_build\html\index.html
```

Notes:
- The `conf.py` adds `src/` to `sys.path` so Sphinx autodoc can import the package.
- If you prefer, install dependencies into your environment or use `pipx` for `sphinx-build`.
