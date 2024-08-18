import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

source_suffix = ['.rst', '.md']
master_doc = 'index'

project = 'Nimbus-Inference'
copyright = '2024, J.L. Rumberger and N.F. Greenwald'
author = 'J.L. Rumberger and N.F. Greenwald'
release = '0.0.1'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

exclude_patterns = ['_build', '**.ipynb_checkpoints', '/path/to/notebook/directory',
                    'Thumbs.db', '.DS_Store', '**/Thumbs.db', '**/.DS_Store']

nbsphinx_allow_errors = True

napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True
autosummary_generate = True

exclude_patterns = []
html_theme = 'sphinx_rtd_theme'
