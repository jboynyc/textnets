#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `textnets` package."""

from click.testing import CliRunner

from textnets import Corpus, Textnet
from textnets import cli, examples


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 2
    assert 'Usage:' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert 'Show this message and exit.' in help_result.output

def test_corpus():
    """Test Corpus class using small data frame."""

    c = Corpus(examples.moon_landing)
    assert c._df.shape[0] == 7
    assert c._df.shape[1] == 2

    noun_phrases = c.noun_phrases()
    assert set(noun_phrases.columns) == {'term', 'n'}

    noun_phrases_remove = c.noun_phrases(remove=['moon'])
    assert set(noun_phrases_remove.columns) == {'term', 'n'}

    tokenized = c.tokenized()
    assert set(tokenized.columns) == {'term', 'n'}

    nostem = c.tokenized(stem=False)
    assert set(nostem.columns) == {'term', 'n'}

    nopunct = c.tokenized(remove_punctuation=False)
    assert set(nopunct.columns) == {'term', 'n'}

    upper = c.tokenized(lower=False)
    assert set(upper.columns) == {'term', 'n'}

def test_textnet():
    """Test Textnet class using small data frame."""

    c = Corpus(examples.moon_landing)
    noun_phrases = c.noun_phrases()

    tn_np = Textnet(noun_phrases)
    assert tn_np.graph.vcount() > 0
    assert tn_np.graph.ecount() > 0
    assert set(tn_np._df.columns) == {'term', 'n', 'tf_idf'}
    g_np_groups = tn_np.project(node_type='doc')
    assert g_np_groups.vcount() > 0
    assert g_np_groups.ecount() > 0
    g_np_words = tn_np.project(node_type='term')
    assert g_np_words.vcount() > 0
    assert g_np_words.ecount() > 0

def test_plotting(tmpdir):
    """Test Textnet plotting."""

    c = Corpus(examples.moon_landing)
    noun_phrases = c.noun_phrases()
    tn_np = Textnet(noun_phrases)
    out = tmpdir.join('plot.png')
    plot = tn_np.plot(target=str(out))
    assert len(plot._objects) > 0
    assert len(tmpdir.listdir()) == 1
