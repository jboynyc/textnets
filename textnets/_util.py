"""Utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from pandas import DataFrame, Index, Series, SparseDtype
from scipy.sparse import csr_array


class LiteFrame:
    """DataFrame wrapper for nicer subclassing."""

    def __init__(self, *args, **kwargs) -> None:
        self._df = DataFrame(*args, **kwargs)

    @property
    def columns(self) -> Index:
        return self._df.columns

    @property
    def empty(self) -> bool:
        return self._df.empty

    @property
    def index(self) -> Index:
        return self._df.index

    @property
    def T(self) -> DataFrame:
        return self._df.T

    def applymap(self, *args, **kwargs) -> DataFrame:
        return self._df.applymap(*args, **kwargs)

    def groupby(self, *args, **kwargs):
        return self._df.groupby(*args, **kwargs)

    def merge(self, *args, **kwargs) -> DataFrame:
        return self._df.merge(*args, **kwargs)

    def reset_index(self, *args, **kwargs) -> DataFrame:
        return self._df.reset_index(*args, **kwargs)

    def dot(self, *args, **kwargs) -> DataFrame:
        return self._df.dot(*args, **kwargs)

    def stack(self, *args, **kwargs):
        return self._df.stack(*args, **kwargs)

    def sum(self, *args, **kwargs) -> Series[Any]:
        return self._df.sum(*args, **kwargs)

    def to_numpy(self, *args, **kwargs) -> np.ndarray:
        return self._df.to_numpy(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self._df.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self._df.__setitem__(*args, **kwargs)

    def __matmul__(self, *args, **kwargs):
        return self._df.__matmul__(*args, **kwargs)

    def __gt__(self, *args):
        return self._df.__gt__(*args)

    def __lt__(self, *args):
        return self._df.__lt__(*args)

    def __eq__(self, *args):
        return self._df.__eq__(*args)

    def __array__(self, dtype=None) -> np.ndarray:
        return self._df.__array__(dtype=dtype)

    def to_array(self) -> np.ndarray:
        """Return numpy array with float32 numeric data."""
        a = self.to_numpy()
        return a.astype("float32")

    def to_sparse_array(self) -> csr_array:
        """Return CSR sparse array with float32 numeric data."""
        return csr_array(self.to_array())

    @property
    def density(self) -> float:
        return self._df.astype(SparseDtype("float32", 0)).sparse.density  # type: ignore
