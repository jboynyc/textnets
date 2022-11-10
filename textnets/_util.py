# -*- coding: utf-8 -*-

"""Utilities."""

from __future__ import annotations

from typing import Any

from numpy import ndarray
from pandas import DataFrame, Index, Series


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

    @property
    def values(self) -> ndarray:
        return self._df.values

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

    def to_numpy(self, *args, **kwargs) -> ndarray:
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

    def __array__(self, dtype=None) -> ndarray:
        return self._df.__array__(dtype=dtype)
