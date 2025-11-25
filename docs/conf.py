import os
import sys

# Ensure `src` is on sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

project = "auto_adpq"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

# If you prefer RTD theme, switch `html_theme` back to 'sphinx_rtd_theme'.
# Load custom CSS from `_static` if present
html_css_files = [
    "custom.css",
]

# Furo theme options: use a small logo and tweak sidebar width
html_logo = "_static/logo.svg"
html_theme_options = {
    "sidebar_hide_name": False,
}

# Autodoc defaults
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
