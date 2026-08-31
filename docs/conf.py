"""Sphinx configuration for the AMLGraphX public documentation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "AMLGraphX"
author = "AMLGraphX contributors"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_typehints = "description"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "furo"
html_title = "AMLGraphX"

html_static_path = ["_static"]

html_theme_options = {
    "navigation_with_keys": True,
}

pygments_style = "sphinx"
pygments_dark_style = "monokai"

# Documentation examples are illustrative. Notebook execution, when notebooks
# are added later, belongs in a separately controlled CI workflow.
nb_execution_mode = "off"
