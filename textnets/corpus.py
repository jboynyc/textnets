# -*- coding: utf-8 -*-

"""Implements the features relating to language."""

from __future__ import annotations

import os
import sqlite3
from glob import glob
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union
from warnings import warn

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Token
from spacy.tokens.doc import Doc
from toolz import compose, identity, memoize, partial

import textnets as tn

#: Mapping of language codes to spacy language model names.
LANGS = {
    "ca": "ca_core_news_sm",  # Catalan
    "da": "da_core_news_sm",  # Danish
    "de": "de_core_news_sm",  # German
    "el": "el_core_news_sm",  # Greek
    "en": "en_core_web_sm",  # English
    "es": "es_core_news_sm",  # Spanish
    "fi": "fi_core_news_sm",  # Finnish
    "fr": "fr_core_news_sm",  # French
    "hr": "hr_core_news_sm",  # Croatian
    "it": "it_core_news_sm",  # Italian
    "ja": "ja_core_news_sm",  # Japanese
    "ko": "ko_core_news_sm",  # Korean
    "lt": "lt_core_news_sm",  # Lithuanian
    "mk": "mk_core_news_sm",  # Macedonian
    "nb": "nb_core_news_sm",  # Norwegian
    "nl": "nl_core_news_sm",  # Dutch
    "pl": "pl_core_news_sm",  # Polish
    "pt": "pt_core_news_sm",  # Portuguese
    "ro": "ro_core_news_sm",  # Romanian
    "ru": "ru_core_news_sm",  # Russian
    "sv": "sv_core_news_sm",  # Swedish
    "uk": "uk_core_news_sm",  # Ukrainian
    "zh": "zh_core_web_sm",  # Chinese
}

#: Custom type for objects resembling documents (token sequences).
DocLike = Union[Doc, Sequence[Token]]

_INSTALLED_MODELS = spacy.util.get_installed_models()


class Corpus:
    """
    Corpus of labeled documents.

    Parameters
    ----------
    data : Series
        Series containing the documents. The index must contain document
        labels.
    lang : str, optional
        The langugage model to use (default set by "lang" parameter).

    Raises
    ------
    ValueError
        If the supplied data is empty.

    Attributes
    ----------
    documents : Series
        The corpus documents.
    lang : str
        The language model used (ISO code or spaCy model name).
    """

    def __init__(
        self,
        data: pd.Series,
        lang: Optional[str] = None,
    ) -> None:
        if data.empty:
            raise ValueError("Corpus data is empty.")
        documents = data.copy()
        if missings := documents.isna().sum():
            warn(f"Dropping {missings} empty document(s).")
            documents = documents[~documents.isna()]
        documents.index = documents.index.set_names(["label"])
        self.documents = documents
        if lang is None:
            lang = tn.params["lang"]
        self.lang = LANGS.get(lang, lang)
        if self.lang not in _INSTALLED_MODELS:
            warn(f"Language model '{self.lang}' is not yet installed.")

    @property
    def nlp(self) -> pd.Series:
        """Corpus documents with NLP applied."""
        return self._nlp(self.lang)

    @memoize
    def _nlp(self, lang: str) -> pd.Series:
        try:
            nlp = spacy.load(lang, exclude=["ner", "textcat"])
        except OSError as err:
            if tn.params["autodownload"]:
                try:
                    spacy.cli.download(lang)  # type: ignore
                    _INSTALLED_MODELS.append(lang)
                    return self._nlp(lang)
                except (KeyError, OSError):
                    pass
            elif lang in LANGS.values():
                raise err
            nlp = spacy.blank(lang)
            warn(f"Using basic '{lang}' language model.")
        return self.documents.map(_normalize_whitespace).map(nlp)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, key: str) -> str:
        return self.documents[key]

    @classmethod
    def from_df(
        cls,
        data: pd.DataFrame,
        doc_col: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Corpus:
        """
        Create corpus from data frame.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing documents. The index must contain document
            labels.
        doc_col : str, optional
            Indicates which column of ``data`` contains the document texts. If
            none is specified, the first column with strings is used.
        lang : str, optional
            The langugage model to use (default set by "lang" parameter).

        Raises
        ------
        NoDocumentColumnException
            If no document column can be detected.

        Returns
        -------
        `Corpus`
        """
        object_cols = data.select_dtypes(include="object").columns
        if doc_col is None and object_cols.empty:
            raise NoDocumentColumnException("No suitable document column.")
        if doc_col is None:
            doc_col = str(object_cols[0])
        return cls(data.copy()[doc_col], lang=lang)

    @classmethod
    def from_dict(
        cls,
        data: dict[Any, str],
        lang: Optional[str] = None,
    ) -> Corpus:
        """
        Create corpus from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing the documents as values and document labels
            as keys.
        lang : str, optional
            The langugage model to use (default set by "lang" parameter).

        Returns
        -------
        `Corpus`
        """
        return cls(pd.Series(data), lang=lang)

    @classmethod
    def from_files(
        cls,
        files: Union[str, list[str], list[Path]],
        doc_labels: Optional[list[str]] = None,
        lang: Optional[str] = None,
    ) -> Corpus:
        """Construct corpus from files.

        Parameters
        ----------
        files : str or list of str or list of Path
            Path to files (with globbing pattern) or list of file paths.
        doc_labels : list of str, optional
            Labels for documents (default: file name without suffix).
        lang : str, optional
            The langugage model to use (default set by "lang" parameter).

        Raises
        ------
        IsADirectoryError
            If the provided path is a directory. (Use globbing.)
        FileNotFoundError
            If the provided path does not exist.

        Returns
        -------
        `Corpus`
        """
        if isinstance(files, str):
            files = glob(os.path.expanduser(files))
        files = [Path(f) for f in files]
        for file in files:
            if file.expanduser().is_file():
                pass
            elif file.expanduser().exists():
                raise IsADirectoryError(file.name)
            else:
                raise FileNotFoundError(file.name)
        if not doc_labels:
            doc_labels = [file.stem for file in files]
        data = pd.DataFrame({"path": files}, index=doc_labels)
        data["raw"] = data["path"].map(_read_file)
        return cls.from_df(data, doc_col="raw", lang=lang)

    @classmethod
    def from_csv(
        cls,
        path: str,
        label_col: Optional[str] = None,
        doc_col: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
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
            The langugage model to use (default set by "lang" parameter).
        kwargs
            Arguments to pass to `pandas.read_csv`.

        Returns
        -------
        `Corpus`
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
        lang: Optional[str] = None,
        **kwargs,
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
            The langugage model to use (default set by "lang" parameter).
        kwargs
            Arguments to pass to `pandas.read_sql`.

        Returns
        -------
        `Corpus`
        """
        kwargs.setdefault("index_col", label_col)
        data = pd.read_sql(qry, conn, **kwargs)
        if not label_col or isinstance(data.index, pd.RangeIndex):
            data = data.set_index(data.columns[0])
        return cls.from_df(data, doc_col=doc_col, lang=lang)

    def save(self, target: Union[os.PathLike[Any], str]) -> None:
        """
        Save a corpus to file.

        Parameters
        ----------
        target : str or path
            File to save the corpus to. If the file exists, it will be
            overwritten.
        """
        conn = sqlite3.connect(Path(target))
        meta = {"lang": self.lang}
        with conn as c:
            self.documents.to_sql("corpus_documents", c, if_exists="replace")
            pd.Series(meta, name="values").to_sql(
                "corpus_meta", c, if_exists="replace", index_label="keys"
            )

    @classmethod
    def load(cls, source: Union[os.PathLike[Any], str]) -> Corpus:
        """
        Load a corpus from file.

        Parameters
        ----------
        source : str or path
            File to read the corpus from. This should be a file created by
            `Corpus.save`.

        Raises
        ------
        FileNotFoundError
            If the specified path does not exist.

        Returns
        -------
        `Corpus`
        """
        if not Path(source).exists():
            raise FileNotFoundError(f"File '{source}' does not exist.")
        conn = sqlite3.connect(Path(source))
        with conn:
            documents = pd.read_sql(
                "SELECT * FROM corpus_documents", conn, index_col="label"
            )
            meta = pd.read_sql("SELECT * FROM corpus_meta", conn, index_col="keys")[
                "values"
            ]
        return cls.from_df(documents, lang=meta["lang"])

    def tokenized(
        self,
        remove: Optional[list[str]] = None,
        stem: bool = True,
        remove_stop_words: bool = True,
        remove_urls: bool = True,
        remove_numbers: bool = True,
        remove_punctuation: bool = True,
        lower: bool = True,
        sublinear: bool = True,
    ) -> TidyText:
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
        lower : bool, optional
            Make lower-case (default: True).
        sublinear : bool, optional
            Apply sublinear scaling when calculating *tf-idf* term weights
            (default: True).

        Returns
        -------
        `pandas.DataFrame`
            A data frame with document labels (index), tokens (term), and
            per-document counts (n).
        """
        func = compose(
            partial(_remove_additional, token_list=remove)
            if remove is not None
            else identity,
            _lower if lower else identity,
            _stem if stem else _as_text,
            _remove_stop_words if remove_stop_words else identity,
            _remove_urls if remove_urls else identity,
            _remove_numbers if remove_numbers else identity,
            _remove_punctuation if remove_punctuation else identity,
        )
        tt = self._make_tidy_text(func)
        return _tf_idf(tt, sublinear)

    def noun_phrases(
        self,
        normalize: bool = False,
        remove: Optional[list[str]] = None,
        sublinear: bool = True,
    ) -> TidyText:
        """Return noun phrases from corpus in tidy format.

        Parameters
        ----------
        normalize : bool, optional
            Return lemmas of noun phrases (default: False).
        remove : list of str, optional
            Additional tokens to remove.
        sublinear : bool, optional
            Apply sublinear scaling when calculating *tf-idf* term weights
            (default: True).

        Returns
        -------
        `pandas.DataFrame`
            A data frame with document labels (index), noun phrases
            (term), and per-document counts (n).
        """
        func = compose(
            partial(_remove_additional, token_list=remove)
            if remove is not None
            else identity,
            partial(_noun_chunks, normalize=normalize),
        )
        tt = self._make_tidy_text(func)
        return _tf_idf(tt, sublinear)

    def ngrams(
        self,
        size: int,
        remove: Optional[list[str]] = None,
        stem: bool = False,
        remove_stop_words: bool = False,
        remove_urls: bool = False,
        remove_numbers: bool = False,
        remove_punctuation: bool = False,
        lower: bool = False,
        sublinear: bool = True,
    ) -> TidyText:
        """Return n-grams of length n from corpus in tidy format.

        Parameters
        ----------
        size : int
            Size of n-grams to return.
        remove : list of str, optional
            Additional tokens to remove.
        stem : bool, optional
            Return token stems (default: False).
        remove_stop_words : bool, optional
            Remove stop words (default: False).
        remove_urls : bool, optional
            Remove URL and email address tokens (default: False).
        remove_numbers : bool, optional
            Remove number tokens (default: False).
        remove_punctuation : bool, optional
            Remove punctuation marks, brackets, and quotation marks
            (default: False).
        lower : bool, optional
            Make lower-case (default: False).
        sublinear : bool, optional
            Apply sublinear scaling when calculating *tf-idf* term weights
            (default: True).

        Returns
        -------
        `pandas.DataFrame`
            A data frame with document labels (index), n-grams (term), and
            per-document counts (n).
        """
        func = compose(
            partial(_ngrams, n=size),
            partial(_remove_additional, token_list=remove)
            if remove is not None
            else identity,
            _lower if lower else identity,
            _stem if stem else _as_text,
            _remove_stop_words if remove_stop_words else identity,
            _remove_urls if remove_urls else identity,
            _remove_numbers if remove_numbers else identity,
            _remove_punctuation if remove_punctuation else identity,
        )
        tt = self._make_tidy_text(func)
        return _tf_idf(tt, sublinear)

    def _make_tidy_text(self, func: Callable[[Doc], list[str]]) -> TidyText:
        tt = (
            pd.melt(
                self.nlp.map(func).apply(pd.Series).reset_index(),
                id_vars=["label"],
                value_name="term",
            )
            .rename(columns={"variable": "n"})
            .groupby(["label", "term"])
            .count()
            .reset_index()
            .set_index("label")
        )
        return TidyText(tt)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with {len(self.documents)} documents "
            + f"using language model '{self.lang}'>"
        )

    def _repr_html_(self) -> str:
        tbl = pd.DataFrame(self.documents).to_html(
            header=False,
            notebook=False,
            border=0,
            classes=("full-width", "left-align"),
            max_rows=10,
        )
        return f"""
            <style scoped>
              .full-width {{ width: 100%; }}
              .left-align td, .left-align th {{ text-align: left; }}
              summary {{
                cursor: help;
                list-style: none;
              }}
              details[open] summary {{
                margin-bottom: 1em;
              }}
            </style>
            <details>
              <summary>
                <table class="full-width">
                  <tr style="font-weight: 600;">
                    <td style="text-align: left;">
                      <kbd>{self.__class__.__name__}</kbd>
                    </td>
                    <td style="color: dodgerblue;">
                      <svg width="1ex" height="1ex">
                        <rect width="1ex" height="1ex" fill="dodgerblue">
                      </svg>
                      Docs: {self.documents.shape[0]}
                    </td>
                    <td style="color: darkgray;">
                      Lang: {self.lang}
                    </td>
                  </tr>
                </table>
              </summary>
              {tbl}
            </details>"""


def _read_file(file_name: Path) -> str:
    """Read contents of file ignoring any unicode errors."""
    return file_name.read_bytes().decode("utf-8", "replace").strip()


def _normalize_whitespace(string: str) -> str:
    """Replace all whitespace with single spaces."""
    return " ".join(string.split())


def _noun_chunks(doc: Doc, normalize: bool) -> list[str]:
    """Return only the noun chunks in lower case."""
    return [
        (chunk.lemma_ if normalize else " ".join([t.lower_ for t in chunk]))
        for chunk in doc.noun_chunks
        if not all(token.is_stop for token in chunk)
    ]


def _remove_stop_words(doc: DocLike) -> list[Token]:
    """Return document without stop words."""
    return [word for word in doc if not word.is_stop]


def _remove_urls(doc: DocLike) -> list[Token]:
    """Return document without URLs or email addresses."""
    return [word for word in doc if not word.like_url and not word.like_email]


def _remove_numbers(doc: DocLike) -> list[Token]:
    """Return document without numbers."""
    return [word for word in doc if not word.like_num]


def _remove_punctuation(doc: DocLike) -> list[Token]:
    """Return document without punctuation, brackets and quotation marks."""
    return [
        word
        for word in doc
        if not word.is_punct and not word.is_bracket and not word.is_quote
    ]


def _stem(doc: DocLike) -> list[str]:
    """Return list of word stem strings."""
    return [word.lemma_ for word in doc]


def _as_text(doc: DocLike) -> list[str]:
    """Turn document into list of strings."""
    return [word.text for word in doc]


def _lower(doc: list[str]) -> list[str]:
    """Return list of strings in lower case."""
    return [s.lower() for s in doc]


def _remove_additional(doc: list[str], token_list: list[str]) -> list[str]:
    """Return list of strings without specified tokens."""
    return [s for s in doc if s not in token_list]


def _ngrams(doc: list[str], n: int) -> list[str]:
    """Return list of n-gram strings."""
    return [" ".join(t) for t in zip(*[doc[offset:] for offset in range(n)])]


def _tf_idf(tidy_text: Union[pd.DataFrame, TidyText], sublinear: bool) -> TidyText:
    """Calculate term frequency/inverse document frequency."""
    if sublinear:
        tidy_text["tf"] = tidy_text["n"].map(_sublinear_scaling)
    else:
        totals = tidy_text.groupby(tidy_text.index).sum().rename(columns={"n": "total"})
        tidy_text = tidy_text.merge(totals, right_index=True, left_index=True)
        tidy_text["tf"] = tidy_text["n"] / tidy_text["total"]
    idfs = np.log10(len(set(tidy_text.index)) / tidy_text["term"].value_counts())
    tt = tidy_text.merge(pd.DataFrame(idfs), left_on="term", right_index=True).rename(
        columns={"term_y": "idf"}
    )
    tt["term_weight"] = tt["tf"] * tt["idf"]
    return TidyText(tt[["term", "n", "term_weight"]])


def _sublinear_scaling(n: Union[int, float]) -> float:
    """Logarithmic scaling function."""
    return 1 + np.log10(n) if n > 0 else 0


class NoDocumentColumnException(Exception):
    """Raised if no suitable document column is specified or found."""


class TidyText(pd.DataFrame):
    """Collection of tokens with per-document counts."""
