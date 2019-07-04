#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `textnets` package."""

import os
from glob import glob
import pytest

from click.testing import CliRunner

from textnets import TextCorpus, Textnets
from textnets import cli

import igraph as ig


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 2
    assert 'Usage:' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert 'Show this message and exit.' in help_result.output

def test_sotu():
    """Test main classes using SOTU test corpus."""

    corpus_files = glob(
            os.path.expanduser('~/nltk_data/corpora/state_union/*.txt'))[:5]

    c = TextCorpus(corpus_files)
    assert c._df.shape[0] == len(corpus_files)
    assert c._df.shape[1] == 3

    noun_phrases = c.noun_phrases()
    assert set(noun_phrases.columns) == {'word', 'n'}
    tn_np = Textnets(noun_phrases)
    assert tn_np.graph.vcount() > 0
    assert tn_np.graph.ecount() > 0
    assert set(tn_np._df.columns) == {'word', 'n', 'tf_idf'}
    g_np_groups = tn_np.project(node_type='doc')
    assert g_np_groups.vcount() > 0
    assert g_np_groups.ecount() > 0
    g_np_words = tn_np.project(node_type='term')
    assert g_np_words.vcount() > 0
    assert g_np_words.ecount() > 0

    tokenized = c.tokenized()
    assert set(tokenized.columns) == {'word', 'n'}
    tn_t = Textnets(tokenized)
    assert tn_t.graph.vcount() > 0
    assert tn_t.graph.ecount() > 0
    assert set(tn_t._df.columns) == {'word', 'n', 'tf_idf'}
    g_t_groups = tn_t.project(node_type='doc')
    assert g_t_groups.vcount() > 0
    assert g_t_groups.ecount() > 0
    g_t_words = tn_t.project(node_type='term')
    assert g_t_words.vcount() > 0
    assert g_t_words.ecount() > 0
