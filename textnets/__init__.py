# -*- coding: utf-8 -*-

"""
Top-level package for Textnets.

Citation for this package: :cite:t:`Boy2020`.

Functionality based on :cite:t:`Bail2016`.
"""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # type: ignore

from . import examples  # noqa: F401
from .config import params  # noqa: F401
from .corpus import Corpus  # noqa: F401
from .network import Textnet  # noqa: F401

__all__ = ["Corpus", "Textnet", "params", "examples"]

__author__ = "John D. Boy"
__email__ = "jboy@bius.moe"
__version__ = version(__name__)


def _repr_html_():
    import spacy

    packages = ["python-igraph", "leidenalg", "spacy"]
    package_versions = map(version, packages)
    language_models = spacy.util.get_installed_models()
    model_versions = map(spacy.util.get_package_version, language_models)
    pairs = dict(
        **dict(zip(packages, package_versions)),
        **dict(zip(language_models, model_versions)),
    )
    dl = "\n".join(
        [f"<dt><tt>{pkg}</tt></dt><dd>{ver}</dd>" for pkg, ver in pairs.items()]
    )
    return f"""
    <style scoped>
      .full-width {{ width: 100%; }}
      summary {{
        cursor: help;
        list-style: none;
      }}
      details[open] summary {{
        margin-bottom: 1em;
      }}
    </style>
    <details>
      <summary>
        <table class="full-width">
          <tr style="font-weight: 600;">
            <td style="text-align: left;">
              <kbd>textnets</kbd>
            </td>
            <td style="color: darkgray;">
              Version: {__version__}
            </td>
          </tr>
        </table>
      </summary>
      <dl>{dl}</dl>
    </details>"""
