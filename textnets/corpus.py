# -*- coding: utf-8 -*-

"""Implements the features relating to language."""

from __future__ import annotations

import os
from typing import Callable, Optional, Union, List
from functools import cached_property

import spacy
from spacy.tokens.doc import Doc
import pandas as pd
from glob import glob
from toolz import compose, partial, identity


class Corpus:
    """
    Corpus of labeled documents.

    Parameters
    ----------
    data : Series
        Series containing the documents. The index must contain document
        labels.
    lang : str, optional
        The langugage model to use (default: ``en_core_web_sm``).
    """

    def __init__(
        self,
        data: pd.Series,
        doc_col: Optional[str] = None,
        lang: str = "en_core_web_sm",
    ):
        documents = data.copy()
        documents.index = documents.index.set_names(["label"])
        self.documents = documents
        self.lang = lang

    @cached_property
    def nlp(self) -> pd.Series:
        """Corpus documents with NLP applied."""
        nlp = spacy.load(self.lang, disable=["ner", "textcat"])
        return self.documents.map(_normalize_whitespace).map(nlp)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, key):
        return self.documents[key]

    @classmethod
    def from_df(
        cls,
        data: pd.DataFrame,
        doc_col: Optional[str] = None,
        lang: str = "en_core_web_sm",
    ):
        """
        Create corpus from data frame.

        Parameters
        ----------
        data : DataFrame
            Series containing the documents. The index must contain document
            labels.
        doc_col : str, optional
            If ``data`` is a data frame, this indicates which column contains the
            document texts. If none is specified, the first column with strings is
            used.
        lang : str, optional
            The langugage model to use (default: ``en_core_web_sm``).

        Returns
        -------
        Corpus
        """
        object_cols = data.select_dtypes(include="object").columns
        if not doc_col and object_cols.empty:
            raise NoDocumentColumnException("No suitable document column.")
        elif not doc_col:
            doc_col = object_cols[0]
        return cls(data.copy()[doc_col], lang=lang)

    @classmethod
    def from_files(
        cls,
        files: Union[str, List[str]],
        doc_labels: Optional[List[str]] = None,
        lang: str = "en_core_web_sm",
    ) -> Corpus:
        """Construct corpus from files.

        Parameters
        ----------
        files : str or list of str
            Path to files (with globbing pattern) or list of file paths.
        doc_labels : list of str, optional
            Labels for documents (default: file name without suffix).
        lang : str, optional
            The langugage model to use (default: ``en_core_web_sm``).

        Returns
        -------
        Corpus
        """
        if isinstance(files, str):
            files = glob(os.path.expanduser(files))
        assert all(os.path.exists(f) for f in files), "Some files in list do not exist."
        if not doc_labels:
            doc_labels = [os.path.basename(f).split(".")[0] for f in files]
        df = pd.DataFrame({"path": files}, index=doc_labels)
        df["raw"] = df["path"].map(_read_file)
        return cls(df, doc_col="raw", lang=lang)

    @classmethod
    def from_csv(
        cls,
        path: str,
        label_col: Optional[str] = None,
        doc_col: Optional[str] = None,
        lang: str = "en_core_web_sm",
        **kwargs
    ) -> Corpus:
        """Read corpus from comma-separated value file.

        Parameters
        ----------
        path : str
            Path to CSV file.
        label_col : str, optional
            Column that contains document labels (default: None, in which case
            the first column is used).
        doc_col : str, optional
            Column that contains document text (default: None, in which case
            the first text column is used).
        lang : str, optional
            The langugage model to use (default: ``en_core_web_sm``).
        kwargs
            Arguments to pass to `pandas.read_csv`.

        Returns
        -------
        Corpus
        """
        kwargs.setdefault("index_col", label_col)
        data = pd.read_csv(path, **kwargs)
        if not label_col or isinstance(data.index, pd.RangeIndex):
            data = data.set_index(data.columns[0])
        return cls.from_df(data, doc_col=doc_col, lang=lang)

    @classmethod
    def from_sql(
        cls,
        qry: str,
        conn: Union[str, object],
        label_col: Optional[str] = None,
        doc_col: Optional[str] = None,
        lang: str = "en_core_web_sm",
        **kwargs
    ) -> Corpus:
        """Read corpus from SQL database.

        Parameters
        ----------
        qry : str
            SQL query
        conn : str or object
            Database URI or connection object.
        label_col : str, optional
            Column that contains document labels (default: None, in which case
            the first column is used).
        doc_col : str, optional
            Column that contains document text (default: None, in which case
            the first text column is used).
        lang : str, optional
            The langugage model to use (default: ``en_core_web_sm``).
        kwargs
            Arguments to pass to `pandas.read_sql`.

        Returns
        -------
        Corpus
        """
        kwargs.setdefault("index_col", label_col)
        data = pd.read_sql(qry, conn, **kwargs)
        if not label_col or isinstance(data.index, pd.RangeIndex):
            data = data.set_index(data.columns[0])
        return cls.from_df(data, doc_col=doc_col, lang=lang)

    def tokenized(
        self,
        remove: List[str] = [],
        stem: bool = True,
        remove_stop_words: bool = True,
        remove_urls: bool = True,
        remove_numbers: bool = True,
        remove_punctuation: bool = True,
        lower: bool = True,
    ) -> pd.DataFrame:
        """Return tokenized version of corpus in tidy format.

        Parameters
        ----------
        remove : list of str, optional
            Additional tokens to remove.
        stem : bool, optional
            Return token stems (default: True).
        remove_stop_words : bool, optional
            Remove stop words (default: True).
        remove_urls : bool, optional
            Remove URL and email address tokens (default: True).
        remove_numbers : bool, optional
            Remove number tokens (default: True).
        remove_punctuation : bool, optional
            Remove punctuation marks, brackets, and quotation marks
            (default: True).

        Returns
        -------
        pd.DataFrame
            A data frame with document labels (index), tokens (term), and
            per-document counts (n).
        """
        func = compose(
            partial(_remove_additional, token_list=remove) if remove else identity,
            _lower if lower else identity,
            _stem if stem else _as_text,
            _remove_stop_words if remove_stop_words else identity,
            _remove_urls if remove_urls else identity,
            _remove_numbers if remove_numbers else identity,
            _remove_punctuation if remove_punctuation else identity,
        )
        return self._return_tidy_text(func)

    def noun_phrases(self, remove: List[str] = []) -> pd.DataFrame:
        """Return noun phrases from corpus in tidy format.

        Parameters
        ----------
        remove : list of str, optional
            Additional tokens to remove.

        Returns
        -------
        pd.DataFrame
            A data frame with document labels (index), noun phrases
            (term), and per-document counts (n).
        """
        func = compose(
            partial(_remove_additional, token_list=remove) if remove else identity,
            _noun_chunks,
        )
        return self._return_tidy_text(func)

    def _return_tidy_text(self, func: Callable[[Doc], List[str]]) -> pd.DataFrame:
        return (
            pd.melt(
                self.nlp.map(func).apply(pd.Series).reset_index(),
                id_vars="label",
                value_name="term",
            )
            .rename(columns={"variable": "n"})
            .groupby(["label", "term"])
            .count()
            .reset_index()
            .set_index("label")
        )


def _read_file(file_name: str) -> str:
    """Read contents of file ignoring any unicode errors."""
    return open(file_name, "rb").read().decode("utf-8", "replace").strip()


def _normalize_whitespace(s: str) -> str:
    """Replace all whitespace with single spaces."""
    return " ".join(s.split())


def _noun_chunks(doc: Doc) -> List[str]:
    """Return only the noun chunks in lower case."""
    return [
        chunk.lower_
        for chunk in doc.noun_chunks
        if not all(token.is_stop for token in chunk)
    ]


def _remove_stop_words(doc: Doc) -> Doc:
    """Return document without stop words."""
    return [word for word in doc if not word.is_stop]


def _remove_urls(doc: Doc) -> Doc:
    """Return document without URLs or email addresses."""
    return [word for word in doc if not word.like_url and not word.like_email]


def _remove_numbers(doc: Doc) -> Doc:
    """Return document without numbers."""
    return [word for word in doc if not word.like_num]


def _remove_punctuation(doc: Doc) -> Doc:
    """Return document without punctuation, brackets and quotation marks."""
    return [
        word
        for word in doc
        if not word.is_punct and not word.is_bracket and not word.is_quote
    ]


def _stem(doc: Doc) -> List[str]:
    """Return list of word stem strings."""
    return [word.lemma_ for word in doc]


def _as_text(doc: Doc) -> List[str]:
    """Turn document into list of strings."""
    return [word.text for word in doc]


def _lower(doc: List[str]) -> List[str]:
    """Return list of strings in lower case."""
    return [s.lower() for s in doc]


def _remove_additional(doc: List[str], token_list: List[str]) -> List[str]:
    """Return list of strings without specified tokens."""
    return [s for s in doc if s not in token_list]


class NoDocumentColumnException(Exception):
    """Raised if no suitable document column is specified or found."""
