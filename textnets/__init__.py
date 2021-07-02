# -*- coding: utf-8 -*-

"""Top-level package for Textnets.

Citation for this package: :cite:`Boy2020`. Functionality based on :cite:`Bail2016`."""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # type: ignore

from .corpus import Corpus  # noqa: F401
from .network import Textnet  # noqa: F401


__author__ = """John D. Boy"""
__email__ = "jboy@bius.moe"
__version__ = version(__name__)
