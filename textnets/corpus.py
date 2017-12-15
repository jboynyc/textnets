# -*- coding: utf-8 -*-

import os
import spacy
import pandas as pd
from glob import glob
from toolz import compose, identity


class TextCorpus:
    def __init__(self, files, lang='en', group_labels=None):
        if isinstance(files, str):
            files = glob(os.path.expanduser(files))
        assert all(os.path.exists(f) for f in files), \
            'Some files in list do not exist.'
        nlp = spacy.load(lang)
        if not group_labels:
            group_labels = [os.path.basename(f).split('.')[0] for f in files]
        self._df = pd.DataFrame({'path': files},
                                index=group_labels)
        self._df['raw'] = self._df['path'].map(_read_file)
        self._df['nlp'] = self._df['raw'].map(nlp)

    def tokenized(self, remove_stop_words=True, remove_urls=True,
                  remove_numbers=True, remove_punctuation=True, stem=True):
        func = compose(
            _stem if stem else _as_text,
            _remove_whitespace,
            _remove_stop_words if remove_stop_words else identity,
            _remove_urls if remove_urls else identity,
            _remove_numbers if remove_numbers else identity,
            _remove_punctuation if remove_punctuation else identity)
        return self._return_tidy_text(func)

    def noun_phrases(self):
        func = _noun_chunks
        return self._return_tidy_text(func)

    def _return_tidy_text(self, func):
        return pd.melt(self._df['nlp'].map(func).apply(pd.Series)
                       .reset_index(), id_vars='index', value_name='word')\
            .rename(columns={'variable': 'n'})\
            .groupby(['index', 'word'])\
            .count()\
            .reset_index()\
            .set_index('index')


def _read_file(file_name):
    return open(file_name, 'rb').read()\
        .decode('utf-8', 'replace')\
        .strip()


def _noun_chunks(doc):
    return [chunk.lower_ for chunk in doc.noun_chunks]


def _remove_stop_words(doc):
    return [word for word in doc if not word.is_stop]


def _remove_urls(doc):
    return [word for word in doc if not word.like_url
            and not word.like_email]


def _remove_numbers(doc):
    return [word for word in doc if not word.like_num]


def _remove_punctuation(doc):
    return [word for word in doc if not word.is_punct]


def _remove_whitespace(doc):
    return [word for word in doc if not word.is_space]


def _stem(doc):
    return [word.lemma_ for word in doc]


def _as_text(doc):
    return [word.lower_ for word in doc]
