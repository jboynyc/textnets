# -*- coding: utf-8 -*-

"""Implements experimental features for formal concept analysis."""

from functools import cached_property
from typing import Tuple, List

import pandas as pd


#: Membership degree threshold (alpha) for concept lattice
#: (see :cite:`Tho2006`).
FFCA_CUTOFF = 0.3


class FormalContext:
    """Textnets inherits methods from this class for treating its incidence
    matrix as a formal context."""

    def __init__(self, im: pd.DataFrame):
        self.im = im

    @cached_property
    def context(self):
        """Return formal context of terms and documents."""
        return self._formal_context(self.im, alpha=FFCA_CUTOFF)

    @staticmethod
    def _formal_context(im, alpha) -> Tuple[List[str], List[str], List[List[bool]]]:
        # The incidence matrix is a "fuzzy formal context." We can binarize it
        # by using a cutoff. This is known as an alpha-cut.
        crisp = im.applymap(lambda x: True if x >= alpha else False)
        reduced = crisp[crisp.any(axis=1)].loc[:, crisp.any(axis=0)]
        objects = reduced.index.tolist()
        properties = reduced.columns.tolist()
        bools = reduced.to_numpy().tolist()
        return objects, properties, bools
