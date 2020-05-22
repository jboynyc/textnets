#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `textnets` package."""

import pytest

from click.testing import CliRunner

from textnets import Corpus, Textnet
from textnets import cli

import igraph as ig
import pandas as pd


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
    """Test main classes using small data frame."""

    moon_landing = pd.DataFrame(
        {'paper': ['The Guardian',
                   'New York Times',
                   'Boston Globe',
                   'Houston Chronicle',
                   'Washington Post',
                   'Chicago Tribune',
                   'Los Angeles Times'],
         'headline': ['3:29 am Man Steps Onto the Moon',
                      'Men Walk on Moon -- Astronauts Land on Plain, Collect Rocks, Plant Flag',
                      'Man Walks on Moon',
                      'Armstrong and Aldrich Take One Small Step for Man on the Moon',
                      'The Eagle Has Landed Two Men Walk on the Moon',
                      'Giant Leap for Mankind',
                      'Walk on Moon That\'s on Small Step for Man, One Giant Leap for Mankind']
        }).set_index('paper')

    c = Corpus(moon_landing)
    assert c._df.shape[0] == 7
    assert c._df.shape[1] == 2

    noun_phrases = c.noun_phrases()
    assert set(noun_phrases.columns) == {'term', 'n'}

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

    tokenized = c.tokenized()
    assert set(tokenized.columns) == {'term', 'n'}

    tn_t = Textnet(tokenized)
    assert tn_t.graph.vcount() > 0
    assert tn_t.graph.ecount() > 0
    assert set(tn_t._df.columns) == {'term', 'n', 'tf_idf'}
    g_t_groups = tn_t.project(node_type='doc')
    assert g_t_groups.vcount() > 0
    assert g_t_groups.ecount() > 0
    g_t_words = tn_t.project(node_type='term')
    assert g_t_words.vcount() > 0
    assert g_t_words.ecount() > 0
