# -*- coding: utf-8 -*-

"""Implements configuration parameter features.

Global Parameters
-----------------

>>> import textnets as tn
>>> tn.params.update({"lang": "de", "autodownload": True})
>>> tn.params["seed"]

``autodownload`` (default: False)
  If True, **textnets** should attempt to download any required language
  models.

``ffca_cutoff`` (default: 0.3)
  Membership degree threshold (*alpha*) for concept lattice (see
  :cite:`Tho2006`).

``lang`` (default: en_core_web_sm)
  Default language model to use.

``resolution_parameter`` (default: 0.1)
  Resolution parameter (*gamma*) for community detection (see
  :cite:`Reichardt2006,Traag2019`).

``seed`` (default: random integer)
  Specify a seed for the random number generator to get reproducible results
  for graph layouts and community detection.

``tuning_parameter`` (default: 0.5)
  Tuning parameter (*alpha*) for inverse edge weights (see :cite:`Opsahl2010`).
"""

from __future__ import annotations

import os
from collections import UserDict
from random import randint
from warnings import warn


class _Configuration(UserDict):
    """Container for global parameters."""

    _params = {
        "autodownload",
        "ffca_cutoff",
        "resolution_parameter",
        "seed",
        "tuning_parameter",
    }

    def __setitem__(self, key, item):
        if key not in self._params:
            warn(f"Parameter '{key}' not known. Skipping.")
        else:
            self.data[key] = item

    def _repr_html_(self) -> str:
        rows = [f"<tr><td>{par}</td><td>{val}</td></tr>" for par, val in self.items()]
        return f"""
          <table class="full-width">
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            {os.linesep.join(rows)}
            <tr style="font-weight: 600;">
              <td style="text-align: left;">
                <kbd>params</kbd>
              </td>
            </tr>
          </table>"""


params = _Configuration(seed=randint(0, 10_000))
