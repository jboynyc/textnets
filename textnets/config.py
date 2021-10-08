# -*- coding: utf-8 -*-

"""Implements configuration parameter features."""

from __future__ import annotations

import os
from collections import UserDict
from random import randint


class _Configuration(UserDict):
    def _repr_html_(self) -> str:
        rows = [f"<tr><td>{par}</td><td>{val}</td></tr>" for par, val in self.items()]
        return f"""
          <table class="full-width">
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            {os.linesep.join(rows)}
            <tr style="font-weight: 600;">
              <td style="text-align: left;">
                <kbd>config</kbd>
              </td>
            </tr>
          </table>"""


config = _Configuration(seed=randint(0, 10_000))
