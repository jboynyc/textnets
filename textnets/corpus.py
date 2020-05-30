# -*- coding: utf-8 -*-

import os
from typing import Callable, Optional, Union, List

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
    data : DataFrame
        Data frame containing the documents. The index contains document
        labels.
    doc_col : str, optional
        Indicates which column of `data` contains the document texts. If
        none is specified, the first column with strings is assumed to
        contain document texts.
    lang : str, optional
        The langugage model to be used. Defaults to en_core_web_sm.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        doc_col: Optional[str] = None,
        lang: str = "en_core_web_sm",
    ):
        nlp = spacy.load(lang, disable=["ner", "textcat"])
        self._df = data.copy()
        self._df.index = self._df.index.set_names(["label"])
        if not doc_col:
            doc_col = self._df.select_dtypes(include="object").columns[0]
        self._df["nlp"] = self._df[doc_col].map(_normalize_whitespace).map(nlp)

    @classmethod
    def from_files(
        cls,
        files: Union[str, List[str]],
        doc_labels: Optional[List[str]] = None,
        lang: str = "en_core_web_sm",
    ):
        """Construct corpus from files.
        """
        if isinstance(files, str):
            files = glob(os.path.expanduser(files))
        assert all(os.path.exists(f) for f in files), "Some files in list do not exist."
        if not doc_labels:
            doc_labels = [os.path.basename(f).split(".")[0] for f in files]
        df = pd.DataFrame({"path": files}, index=doc_labels)
        df["raw"] = df["path"].map(_read_file)
        return cls(df, doc_col="raw", lang=lang)

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
                self._df["nlp"].map(func).apply(pd.Series).reset_index(),
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
    return [word for word in doc if not word.is_stop]


def _remove_urls(doc: Doc) -> Doc:
    return [word for word in doc if not word.like_url and not word.like_email]


def _remove_numbers(doc: Doc) -> Doc:
    return [word for word in doc if not word.like_num]


def _remove_punctuation(doc: Doc) -> Doc:
    return [
        word
        for word in doc
        if not word.is_punct and not word.is_bracket and not word.is_quote
    ]


def _stem(doc: Doc) -> List[str]:
    return [word.lemma_ for word in doc]


def _as_text(doc: Doc) -> List[str]:
    return [word.text for word in doc]


def _lower(doc: List[str]) -> List[str]:
    return [s.lower() for s in doc]


def _remove_additional(doc: List[str], token_list: List[str]) -> List[str]:
    return [s for s in doc if s not in token_list]
