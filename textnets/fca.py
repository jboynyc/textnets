# -*- coding: utf-8 -*-

"""Implements experimental features for formal concept analysis."""

from abc import ABC, abstractmethod
from typing import List, Tuple
from warnings import warn

from toolz import memoize

import textnets as tn


class FormalContext(ABC):
    """
    Abstract base class providing experimental FCA features.

    Textnets inherits methods from this class for treating its incidence matrix
    as a formal context.
    """

    @property
    @abstractmethod
    def im(self):
        pass

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
        try:
            from concepts import Context
        except ImportError:
            warn("Install textnets[fca] to use FCA features.")
            raise
        crisp = self.im.applymap(lambda x: x >= alpha)
        reduced = crisp[crisp.any(axis=1)].loc[:, crisp.any(axis=0)]
        objects = reduced.index.tolist()
        properties = reduced.columns.tolist()
        bools = reduced.to_numpy().tolist()
        return Context(objects, properties, bools)
