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
import os
import sys

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "torch-influence"
copyright = "2022, Alston Lo, Juhan Bae"
author = "Alston Lo, Juhan Bae"

# The full version, including alpha/beta/rc tags
import torch_influence

release = torch_influence.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

napoleon_use_param = True
autoclass_content = "both"

# FIXME: hacky formatting
import torch
from torch.utils import data


def typehints_format_fn(annotation, config):
    name_register = {
        torch.nn.Module: ":class:`torch.nn.Module`",
        torch.Tensor: ":class:`torch.Tensor`",
        data.DataLoader: ":class:`torch.utils.data.DataLoader`",
        torch.device: ":class:`torch.device`"
    }

    return name_register.get(annotation, None)


autodoc_member_order = "bysource"
typehints_formatter = typehints_format_fn

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
