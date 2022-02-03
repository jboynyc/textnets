# -*- coding: utf-8 -*-

"""Implements experimental features for formal concept analysis."""

from typing import List, Tuple

import pandas as pd
from toolz import memoize

import textnets as tn


class FormalContext:
    """
    Class providing experimental FCA features.

    Textnets inherits methods from this class for treating its incidence matrix
    as a formal context.

    Parameters
    ----------
    im : pandas.DataFrame
        Incidence matrix of bipartite graph.
    """

    def __init__(self, im: pd.DataFrame) -> None:
        self.im = im

    @property
    def context(self) -> Tuple[List[str], List[str], List[List[bool]]]:
        """Return formal context of terms and documents."""
        return self._formal_context(alpha=tn.params["ffca_cutoff"])

    @memoize
    def _formal_context(
        self, alpha: float
    ) -> Tuple[List[str], List[str], List[List[bool]]]:
        # The incidence matrix is a "fuzzy formal context." We can binarize it
        # by using a cutoff. This is known as an alpha-cut. This feature is
        # experimental.
        crisp = self.im.applymap(lambda x: x >= alpha)
        reduced = crisp[crisp.any(axis=1)].loc[:, crisp.any(axis=0)]
        objects = reduced.index.tolist()
        properties = reduced.columns.tolist()
        bools = reduced.to_numpy().tolist()
        return objects, properties, bools
