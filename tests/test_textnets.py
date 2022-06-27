#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `textnets` package."""

import sqlite3

import pandas as pd
import textnets as tn


def test_corpus(corpus):
    """Test Corpus class using small data frame."""

    assert len(corpus.documents) == 7

    noun_phrases = corpus.noun_phrases()
    assert noun_phrases.sum().n == 24
    assert set(noun_phrases.columns) == {"term", "n", "term_weight"}

    noun_phrases_remove = corpus.noun_phrases(remove=["moon"])
    assert noun_phrases_remove.sum().n == 20
    assert set(noun_phrases_remove.columns) == {"term", "n", "term_weight"}

    noun_phrases_remove = corpus.noun_phrases(normalize=True)
    assert set(noun_phrases_remove.columns) == {"term", "n", "term_weight"}

    tokenized = corpus.tokenized()
    assert tokenized.sum().n == 43
    assert set(tokenized.columns) == {"term", "n", "term_weight"}

    nostem = corpus.tokenized(stem=False)
    assert set(nostem.columns) == {"term", "n", "term_weight"}

    nopunct = corpus.tokenized(remove_punctuation=False)
    assert set(nopunct.columns) == {"term", "n", "term_weight"}

    upper = corpus.tokenized(lower=False)
    assert set(upper.columns) == {"term", "n", "term_weight"}

    ngrams = corpus.ngrams(3)
    assert ngrams.sum().n == 67
    assert set(ngrams.columns) == {"term", "n", "term_weight"}


def test_corpus_missing(testdata, recwarn):
    """Test Corpus class on series with missing data."""
    s = pd.concat([testdata, pd.Series([None], index=["Missing"])])
    corpus = tn.Corpus(s)
    assert len(recwarn) == 1
    w = recwarn.pop(UserWarning)
    assert str(w.message) == "Dropping 1 empty document(s)."
    assert len(corpus.documents) == 7


def test_corpus_czech(recwarn):
    """Test Corpus class using Czech language documents."""
    s = pd.Series(
        [
            "Holka modrooká nesedávej tam",
            "Holka modrooká nesedávej u potoka",
            "podemele tvoje oči",
            "vezme li tě bude škoda",
            "V potoce je hastrmánek",
            "V potoce je velká voda",
            "V potoce se voda točí",
            "zatahá tě za copánek",
        ]
    )
    # This raises a warning about an uninstalled language model
    corpus = tn.Corpus(s, lang="cs")
    assert len(corpus.documents) == 8
    # This raises another warning about lacking a language model
    tokenized = corpus.tokenized()
    # Relax this test for now because of a deprecation warning in Python 3.10
    assert len(recwarn) >= 2
    assert tokenized.sum().n > 8
    w1 = recwarn.pop(UserWarning)
    assert str(w1.message) == "Language model 'cs' is not yet installed."
    w2 = recwarn.pop(UserWarning)
    assert str(w2.message) == "Using basic 'cs' language model."


def test_corpus_df(testdata):
    """Test creating a corpus from a data frame."""
    df = pd.DataFrame({"headlines": testdata, "meta": list("ABCDEFG")})
    c = tn.Corpus.from_df(df, doc_col="headlines")
    assert len(c.documents) == 7


def test_corpus_dict(testdata):
    """Test creating a corpus from a dictionary."""
    data = testdata.to_dict()
    c = tn.Corpus.from_dict(data)
    assert len(c.documents) == 7


def test_corpus_csv(tmpdir, testdata):
    """Test creating a corpus from a CSV file."""
    out = tmpdir.join("corpus.csv")
    testdata.to_csv(out)
    c = tn.Corpus.from_csv(out)
    assert len(c.documents) == 7


def test_corpus_sql(testdata):
    """Test creating a corpus from a SQL query."""
    with sqlite3.connect(":memory:") as conn:
        testdata.to_sql("headlines", conn)
        c = tn.Corpus.from_sql("SELECT * FROM headlines", conn)
    assert len(c.documents) == 7


def test_corpus_save_and_load(corpus, tmpdir):
    """Test roundtrip of saving and loading a corpus from file."""
    out = tmpdir.join("out.corpus")
    corpus.save(out)
    loaded = tn.load_corpus(out)
    assert all(corpus.documents == loaded.documents)
    assert corpus.lang == loaded.lang


def test_textnet_save_and_load(corpus, tmpdir):
    """Test roundtrip of saving and loading a textnet from file."""
    out = tmpdir.join("out.textnet")
    net = tn.Textnet(
        corpus.tokenized(),
        connected=True,
        doc_attrs={"test": {"New York Times": 1, "Los Angeles Times": 3}},
    )
    net.save(out)
    loaded = tn.load_textnet(out)
    assert net.nodes["id"] == loaded.nodes["id"]
    assert net.edges["weight"] == loaded.edges["weight"]
    assert net.summary == loaded.summary


def test_config_save_and_load(tmpdir):
    """Test roundtrip of saving and loading configuration parameters."""
    out = tmpdir.join("out.params")
    defaults = tn.params.copy()
    tn.params.update({"lang": "cs", "autodownload": True})
    changed = tn.params.copy()
    tn.params.save(out)
    tn.params.update(defaults)
    tn.params.load(out)
    assert tn.params == changed


def test_textnet(corpus):
    """Test Textnet class using sample data."""
    noun_phrases = corpus.noun_phrases()

    n_np = tn.Textnet(noun_phrases)
    assert n_np.graph.vcount() > 0
    assert n_np.graph.ecount() > 0
    g_np_groups = n_np.project(node_type="doc")
    assert g_np_groups.vcount() > 0
    assert g_np_groups.ecount() > 0
    g_np_words = n_np.project(node_type="term")
    assert g_np_words.vcount() > 0
    assert g_np_words.ecount() > 0


def test_textnet_birank(corpus):
    """Test calculating BiRank."""

    noun_phrases = corpus.noun_phrases()
    n_np = tn.Textnet(noun_phrases)

    assert len(n_np.birank) == n_np.graph.vcount()
    assert len(n_np.cohits) == n_np.graph.vcount()
    assert len(n_np.hits) == n_np.graph.vcount()
    bgrm = tn.network.bipartite_rank(n_np, normalizer="BGRM", max_iter=200)
    assert len(bgrm) == n_np.graph.vcount()


def test_textnet_birank_connected(corpus):
    n_np = tn.Textnet(corpus.tokenized(), min_docs=1, connected=True)

    assert len(n_np.birank) == n_np.graph.vcount()


def test_textnet_clustering(corpus):
    """Test calculating clustering coefficients."""

    noun_phrases = corpus.noun_phrases()
    n_np = tn.Textnet(noun_phrases, connected=True)

    assert len(n_np.clustering) == n_np.graph.vcount()


def test_context(corpus):
    """Test formal context creation from textnet."""

    n = tn.Textnet(corpus.tokenized(sublinear=False))
    ctx = n.context
    assert len(ctx) == 3


def test_save(tmpdir, corpus):
    """Test Textnet graph saving."""

    noun_phrases = corpus.noun_phrases()
    n_np = tn.Textnet(noun_phrases)
    out = tmpdir.join("graph.graphml")
    n_np.save_graph(str(out))
    assert len(tmpdir.listdir()) == 1


def test_plot(tmpdir, corpus):
    """Test Textnet plotting."""

    noun_phrases = corpus.noun_phrases()
    n_np = tn.Textnet(noun_phrases)
    out = tmpdir.join("plot-0.png")
    plot = n_np.plot(target=str(out))
    assert len(plot._objects) > 0
    assert len(tmpdir.listdir()) == 1


def test_plot_layout(tmpdir, corpus):
    """Test Textnet plotting with bipartite layout and node labels."""

    noun_phrases = corpus.noun_phrases()
    n_np = tn.Textnet(noun_phrases)
    out = tmpdir.join("plot-1.png")
    plot = n_np.plot(target=str(out), bipartite_layout=True, label_nodes=True)
    assert len(plot._objects) > 0
    assert len(tmpdir.listdir()) == 1


def test_plot_projected(tmpdir, corpus):
    """Test ProjectedTextnet plotting."""

    n = tn.Textnet(corpus.tokenized())
    papers = n.project(node_type="doc")
    out = tmpdir.join("plot-2.png")
    plot = papers.plot(show_clusters=True, label_nodes=True, target=str(out))
    assert len(plot._objects) > 0
    assert len(tmpdir.listdir()) == 1


def test_plot_backbone(tmpdir, corpus):
    """Test ProjectedTextnet plotting with alpha cut."""

    n = tn.Textnet(corpus.tokenized())
    papers = n.project(node_type="doc")
    out = tmpdir.join("plot-3.png")
    plot = papers.plot(alpha=0.4, label_nodes=True, target=str(out))
    assert len(plot._objects) > 0
    assert len(tmpdir.listdir()) == 1


def test_plot_scaled(tmpdir, corpus):
    """Test ProjectedTextnet plotting with scaled nodes."""

    n = tn.Textnet(corpus.tokenized())
    papers = n.project(node_type="doc")
    out = tmpdir.join("plot-4.png")
    plot = papers.plot(scale_nodes_by="betweenness", label_nodes=True, target=str(out))
    assert len(plot._objects) > 0
    assert len(tmpdir.listdir()) == 1


def test_plot_filtered(tmpdir, corpus):
    """Test ProjectedTextnet plotting filtered labels."""

    n = tn.Textnet(corpus.tokenized())
    papers = n.project(node_type="doc")
    out = tmpdir.join("plot-5.png")
    plot = papers.plot(
        label_nodes=True,
        label_edges=True,
        node_label_filter=lambda v: v.degree() > 2,
        edge_label_filter=lambda e: e["weight"] > 0.1,
        target=str(out),
    )
    assert len(plot._objects) > 0
    assert len(tmpdir.listdir()) == 1


def test_html_repr(corpus):
    assert tn._repr_html_() != ""
    assert corpus._repr_html_() != ""
    assert tn.Textnet(corpus.tokenized())._repr_html_() != ""
