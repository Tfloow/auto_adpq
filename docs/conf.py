import os
import sys

# Ensure `src` is on sys.path so autodoc can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

project = "auto_adpq"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_multiversion",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

# Include the custom templates directory (already present) and add a versions
# template into the sidebar so users can pick a release/version.
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "versions.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
    ]
}
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

# sphinx-multiversion settings: builds multiple branches/tags into one static site
# Adjust the regexes below to match your branch and tag naming conventions.
smv_tag_whitelist = r"^v\d+\.\d+(?:\.\d+)?$"        # tags like v1.2 or v1.2.3
smv_branch_whitelist = r"^(main|master|stable)$"       # branches to include
smv_remote_whitelist = r"^(origin)$"                   # remote to use
smv_released_pattern = r"^v?\d+\.\d+(?:\.\d+)?$"   # which versions are considered released
smv_latest_version = "main"                             # which build is treated as 'latest'

# Optionally tweak output directory name behaviour; defaults are usually fine.
# smv_outputdir_format = "{ref.name}"

