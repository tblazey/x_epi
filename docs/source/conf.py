# Configuration file for the Sphinx documentation builder.

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../x_epi/bin"))

# -- Project information

project = "x_epi"
copyright = "2023, Tyler Blazey"
author = "Blazey"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxarg.ext",
    "sphinx.ext.intersphinx",
    "numpydoc",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

numpydoc_class_members_toctree = False

autodoc_mock_imports = ["numpy", "matplotlib", "scipy", "pypulseq", "nibabel"]

# -- Options for HTML output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_prev_next": False,
    "navbar_end": ["theme-switcher", "search-field.html", "navbar-icon-links.html"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/numpy/numpydoc",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "text": "Home",
        "image_dark": "static/x_epi_logo.png",
        "alt_text": "XEPI Logo",
    },
}

html_logo = "static/x_epi_logo.png"

html_context = {
    "default_mode": "light",
}

# -- Options for EPUB output
epub_show_urls = "footnote"

root_doc = "index"
autosummary_generate = True
