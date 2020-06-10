#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `textnets` package."""

import sqlite3

import pandas as pd

from click.testing import CliRunner

from textnets import Corpus, Textnet
from textnets import cli, examples


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 2
    assert "Usage:" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "Show this message and exit." in help_result.output


def test_corpus():
    """Test Corpus class using small data frame."""

    c = Corpus(examples.moon_landing)
    assert len(c.documents) == 7

    noun_phrases = c.noun_phrases()
    assert set(noun_phrases.columns) == {"term", "n"}

    noun_phrases_remove = c.noun_phrases(remove=["moon"])
    assert set(noun_phrases_remove.columns) == {"term", "n"}

    tokenized = c.tokenized()
    assert set(tokenized.columns) == {"term", "n"}

    nostem = c.tokenized(stem=False)
    assert set(nostem.columns) == {"term", "n"}

    nopunct = c.tokenized(remove_punctuation=False)
    assert set(nopunct.columns) == {"term", "n"}

    upper = c.tokenized(lower=False)
    assert set(upper.columns) == {"term", "n"}


def test_corpus_df():
    df = pd.DataFrame({"headlines": examples.moon_landing, "meta": list("ABCDEFG")})
    c = Corpus.from_df(df, doc_col="headlines")
    assert len(c.documents) == 7


def test_corpus_csv(tmpdir):
    out = tmpdir.join("corpus.csv")
    examples.moon_landing.to_csv(out)
    c = Corpus.from_csv(out)
    assert len(c.documents) == 7


def test_corpus_sql():
    with sqlite3.connect(":memory:") as conn:
        examples.moon_landing.to_sql("headlines", conn)
        c = Corpus.from_sql("SELECT * FROM headlines", conn)
    assert len(c.documents) == 7


def test_textnet():
    """Test Textnet class using small data frame."""

    c = Corpus(examples.moon_landing)
    noun_phrases = c.noun_phrases()

    tn_np = Textnet(noun_phrases)
    assert tn_np.graph.vcount() > 0
    assert tn_np.graph.ecount() > 0
    g_np_groups = tn_np.project(node_type="doc")
    assert g_np_groups.vcount() > 0
    assert g_np_groups.ecount() > 0
    g_np_words = tn_np.project(node_type="term")
    assert g_np_words.vcount() > 0
    assert g_np_words.ecount() > 0


def test_context():
    """Test formal context creation from textnet."""

    c = Corpus(examples.moon_landing)
    tn = Textnet(c.tokenized(), sublinear=False)
    ctx = tn.context
    assert len(ctx) == 3


def test_plot(tmpdir):
    """Test Textnet plotting."""

    c = Corpus(examples.moon_landing)
    noun_phrases = c.noun_phrases()
    tn_np = Textnet(noun_phrases)
    out = tmpdir.join("plot-1.png")
    plot = tn_np.plot(target=str(out))
    assert len(plot._objects) > 0
    assert len(tmpdir.listdir()) == 1


def test_plot_projected(tmpdir):
    """Test ProjectedTextnet plotting."""

    c = Corpus(examples.moon_landing)
    tn = Textnet(c.tokenized())
    papers = tn.project(node_type="doc")
    out = tmpdir.join("plot-2.png")
    plot = papers.plot(show_clusters=True, label_nodes=True, target=str(out))
    assert len(plot._objects) > 0
    assert len(tmpdir.listdir()) == 1
