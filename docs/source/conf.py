# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
#from mrf import __version__
# contents of docs/conf.py
from pkg_resources import get_distribution
release = get_distribution('mrf').version
# for example take major/minor
version = '.'.join(release.split('.')[:2])

project = 'mrf'
copyright = '2019, Jiaxuan Li, Pieter van Dokkum'
author = 'Jiaxuan Li, Pieter van Dokkum'

# The full version, including alpha/beta/rc tags
#release = __version__
#version = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'recommonmark',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel'
    ] #'sphinx.ext.autodoc', 
 
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = 'sphinx'

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'   #'sphinx_rtd_theme'  alabaster 

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
#html_favicon = "_static/favicon.png"
html_logo = "_static/df-logo.png"

html_sidebars = {
#'index':    ['sidebarintro.html', 'searchbox.html'],
    '**':       ['sidebarintro.html', 'localtoc.html', 'relations.html',
                 'searchbox.html']
}