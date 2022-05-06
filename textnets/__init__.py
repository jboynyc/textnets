# -*- coding: utf-8 -*-

"""
Top-level package for Textnets.

Citation for this package: :cite:t:`Boy2020`.

Functionality based on :cite:t:`Bail2016`.
"""

from importlib.metadata import version

from . import examples  # noqa: F401
from .config import init_seed, params  # noqa: F401
from .corpus import Corpus  # noqa: F401
from .network import Textnet  # noqa: F401

__all__ = [
    "Corpus",
    "load_corpus",
    "Textnet",
    "load_textnet",
    "params",
    "init_seed",
    "examples",
]

__author__ = "John D. Boy"
__email__ = "jboy@bius.moe"
__version__ = version(__name__)

#: Load a corpus from file.
load_corpus = Corpus.load

#: Load a textnet from file.
load_textnet = Textnet.load


def _repr_html_() -> str:
    import spacy

    packages = ["igraph", "leidenalg", "spacy"]
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
