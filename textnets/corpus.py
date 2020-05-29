# -*- coding: utf-8 -*-

import os
import spacy
import pandas as pd
from glob import glob
from toolz import compose, partial, identity


class Corpus:
    def __init__(self, data, doc_col=None, lang="en_core_web_sm"):
        nlp = spacy.load(lang, disable=["ner", "textcat"])
        self._df = data.copy()
        self._df.index = self._df.index.set_names(["label"])
        if not doc_col:
            doc_col = self._df.select_dtypes(include="object").columns[0]
        self._df["nlp"] = self._df[doc_col].map(_normalize_whitespace).map(nlp)

    @classmethod
    def from_files(cls, files, doc_labels=None, lang="en_core_web_sm"):
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
        remove=[],
        stem=True,
        remove_stop_words=True,
        remove_urls=True,
        remove_numbers=True,
        remove_punctuation=True,
        lower=True,
    ):
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

    def noun_phrases(self, remove=[]):
        func = compose(
            partial(_remove_additional, token_list=remove) if remove else identity,
            _noun_chunks,
        )
        return self._return_tidy_text(func)

    def _return_tidy_text(self, func):
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


def _read_file(file_name):
    return open(file_name, "rb").read().decode("utf-8", "replace").strip()


def _normalize_whitespace(doc):
    return " ".join(doc.split())


def _noun_chunks(doc):
    return [
        chunk.lower_
        for chunk in doc.noun_chunks
        if not all(token.is_stop for token in chunk)
    ]


def _remove_stop_words(doc):
    return [word for word in doc if not word.is_stop]


def _remove_urls(doc):
    return [word for word in doc if not word.like_url and not word.like_email]


def _remove_numbers(doc):
    return [word for word in doc if not word.like_num]


def _remove_punctuation(doc):
    return [
        word
        for word in doc
        if not word.is_punct and not word.is_bracket and not word.is_quote
    ]


def _stem(doc):
    return [word.lemma_ for word in doc]


def _as_text(doc):
    return [word.text for word in doc]


def _lower(doc):
    return [s.lower() for s in doc]


def _remove_additional(doc, token_list):
    return [s for s in doc if s not in token_list]
