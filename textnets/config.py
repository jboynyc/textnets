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
  :cite:t:`Tho2006`).

``lang`` (default: en_core_web_sm)
  Default language model to use.

``resolution_parameter`` (default: 0.1)
  Resolution parameter (*gamma*) for community detection (see
  :cite:t:`Reichardt2006,Traag2019`).

``seed`` (default: random integer)
  Specify a seed for the random number generator to get reproducible results
  for graph layouts and community detection.

``tuning_parameter`` (default: 0.5)
  Tuning parameter (*alpha*) for inverse edge weights (see
  :cite:t:`Opsahl2010`).
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
from collections import UserDict
from pathlib import Path
from typing import Any, Union
from warnings import warn

from wasabi import msg, table


class TextnetsConfiguration(UserDict):
    """Container for global parameters."""

    _valid = {
        "autodownload",
        "ffca_cutoff",
        "lang",
        "resolution_parameter",
        "seed",
        "tuning_parameter",
    }

    def __setitem__(self, key: str, item: Any) -> None:
        """Set configuration value."""
        if key not in self._valid:
            warn(f"Parameter '{key}' not known. Skipping.")
        else:
            self.data[key] = item

    def save(self, target: Union[os.PathLike[str], str]) -> None:
        """
        Save parameters to file.

        Parameters
        ----------
        target : path
            Location of file to save parameters to.
        """
        conn = sqlite3.connect(Path(target))
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS params(data json)")
            conn.execute("INSERT INTO params VALUES (?)", [json.dumps(self.data)])

    def load(self, source: Union[os.PathLike[str], str]) -> None:
        """
        Load parameters from file.

        Parameters
        ----------
        source : path
            Location of file to load parameters from.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        """
        if not Path(source).exists():
            raise FileNotFoundError(f"File '{source}' does not exist.")
        conn = sqlite3.connect(Path(source))
        with conn as c:
            ser = c.execute(
                "SELECT rowid, * FROM params ORDER BY rowid DESC LIMIT 1"
            ).fetchone()[1]
        params = json.loads(ser)
        self.update(params)
        msg.info(f"Updated global parameters with values loaded from '{source}'.")

    def __repr__(self) -> str:
        return table(self.data, header=["Parameter", "Value"], divider=True)

    def _repr_html_(self) -> str:
        rows = [f"<tr><td>{par}</td><td>{val}</td></tr>" for par, val in self.items()]
        return f"""
          <table class="full-width">
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            {os.linesep.join(rows)}
            <tr style="font-weight: 600;">
              <td colspan="2" style="text-align: left;">
                <kbd>params</kbd>
              </td>
            </tr>
          </table>"""


default_params = {
    "autodownload": False,
    "ffca_cutoff": 0.3,
    "lang": "en_core_web_sm",
    "resolution_parameter": 0.1,
    "tuning_parameter": 0.5,
}

#: Container for global parameters.
params = TextnetsConfiguration(seed=random.randint(0, 10_000), **default_params)


def init_seed() -> None:
    """Initialize the random seed."""
    random.seed(params["seed"])
