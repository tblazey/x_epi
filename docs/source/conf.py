# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../x_epi/bin'))

# -- Project information

project = 'x_epi'
copyright = '2023, Tyler Blazey'
author = 'Blazey'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinxarg.ext',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

autodoc_mock_imports = ['numpy', 'matplotlib', 'scipy', 'pypulseq']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'prev_next_buttons_location': None
}

# -- Options for EPUB output
epub_show_urls = 'footnote'

root_doc = 'index'