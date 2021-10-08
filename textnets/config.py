# -*- coding: utf-8 -*-

"""Implements configuration parameter features."""

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
