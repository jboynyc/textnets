========
Tutorial
========

This tutorial walks you through all the steps required to analyze and visualize
your data using **textnets**. The tutorial first presents a self-contained
example before addressing miscellaneous other issues related to using
**textnets**.

Example
-------

.. tip::

   Download this example as a Jupyter notebook so you can follow along:
   :jupyter-download-notebook:`tutorial`.

   You can also make this tutorial "live" so you can adjust the example code
   and re-run it.

   .. thebe-button:: Do it live!

To use **textnets** in a project, you typically start with the following import:

.. jupyter-execute::

   import textnets as tn

You can set a fixed seed to ensure that results are reproducible across runs of
your script (see :cite:t:`Sandve2013`):

.. jupyter-execute::

   tn.params["seed"] = 42

Construct the corpus from the example data:

.. jupyter-execute::

   corpus = tn.Corpus(tn.examples.moon_landing)

What is this `moon_landing` example all about? (Hint: Click on the output below
to see what's in our corpus.)

.. jupyter-execute::

   corpus

.. note::

   Hat tip to Chris Bail for this example data!

Next, we create the textnet:

.. jupyter-execute::

   t = tn.Textnet(corpus.tokenized(), min_docs=1)

We're using `tokenized` with all defaults, so **textnets** is removing stop
words, applying stemming, and removing punctuation marks, numbers, URLs and the
like. However, we're overriding the default setting for ``min_docs``, opting to
keep even words that appear in only one document (that is, a single newspaper
headline).

When dealing with large corpora, you may also want to supply the argument
``remove_weak_edges=True`` to remove edges with a weight far below the average.
This will result in a sparser graph.

Let's take a look:

.. jupyter-execute::

   t.plot(label_nodes=True,
          show_clusters=True)

The ``show_clusters`` options marks the partitions found by the Leiden
community detection algorithm (see :doc:`here <la:multiplex>`). It identifies
document--term groups that appear to form part of the same theme in the texts.

You may be wondering: why is the moon drifting off by itself in the network
plot? That's because the word moon appears exactly once in each document, so
its *tf-idf* value for each document is 0.

Let's visualize the same thing again, but this time scale the nodes according
to their BiRank (a centrality measure for bipartite networks) and the edges
according to their weights.

.. jupyter-execute::

   t.plot(label_nodes=True,
          show_clusters=True,
          scale_nodes_by="birank",
          scale_edges_by="weight")

We can also visualize the projected networks.

First, the network of newspapers:

.. jupyter-execute::

    papers = t.project(node_type=tn.DOC)
    papers.plot(label_nodes=True)

As before in the bipartite network, we can see the *Houston Chronicle*,
*Chicago Tribune* and *Los Angeles Times* cluster more closely together.

Next, the term network:

.. jupyter-execute::

   words = t.project(node_type=tn.TERM)
   words.plot(label_nodes=True,
              show_clusters=True)

Aside from visualization, we can also analyze our corpus using network metrics.
For instance, documents with high betweenness centrality (or "cultural
betweenness"; :cite:t:`Bail2016`) might link together themes, thereby stimulating
exchange across symbolic divides.

.. jupyter-execute::

   papers.top_betweenness()

As we can see, the *Los Angeles Times* is a cultural bridge linking the
headline themes of the East Coast newspapers to the others.

.. jupyter-execute::

   words.top_betweenness()

It's because the *Times* uses the word "walk" in its headline, linking the "One
Small Step" cluster to the "Man on Moon" cluster.

We can produce the term network plot again, this time scaling nodes according
to their betweenness centrality, and pruning edges from the network using
"backbone extraction" :cite:p:`Serrano2009`.

We can also use ``color_clusters`` (instead of ``show_clusters``) to color
nodes according to their partition.

And we can filter node labels, labeling only those nodes that have a
betweenness centrality score above the median. This is particularly useful in
high-order networks where labeling every single node would cause too much
visual clutter.

.. jupyter-execute::

   words.plot(label_nodes=True,
              scale_nodes_by="betweenness",
              color_clusters=True,
              alpha=0.5,
              edge_width=[10*w for w in words.edges["weight"]],
              edge_opacity=0.4,
              node_label_filter=lambda n: n.betweenness() > words.betweenness.median())

Another measure we can use is the textual spanning measure introduced by
:cite:t:`Stoltz2019`, which can help identify "discursive holes" in the
document-to-document network.

.. jupyter-execute::

   papers.plot(label_nodes=True,
               scale_nodes_by="spanning")

Larger document nodes are similar to nodes that are dissimilar from one
another, so they can be thought of as spanning a wider "distance" in the
discursive field than the smaller ones.

Wrangling Text & Mangling Data
------------------------------

How to go from this admittedly contrived example to working with your own data?
The following snippets are meant to help you get started. The first thing is to
get your data in the right shape.

A textnet is built from a collection—or *corpus*—of texts, so we use the
`Corpus` class to get our data ready. The following snippets assume that you
have imported textnets as above.

From a Dictionary
~~~~~~~~~~~~~~~~~

You may already have your texts in a Python data structure, such as a
dictionary mapping document labels (keys) to documents (values). In that case,
you can use the `from_dict <Corpus.from_dict>` method to construct your
`Corpus`.

.. code:: python

   data = {f"Documento {label+1}": doc for label, doc in enumerate(docs)}
   corpus = tn.Corpus.from_dict(data, lang="it")

You can specify which `language model <https://spacy.io/models>`__ you would
like to use using the ``lang`` argument. The default is English, but you don’t
have to be monolingual to use **textnets**. (Languages in `LANGS` are fully
supported since we can use spacy's statistical language models. Other languages
are only partially supported, so `noun_phrases` will likely not function.)

From Pandas
~~~~~~~~~~~

`Corpus` can read documents directly from pandas' `Series <pd:pandas.Series>`
or `DataFrame <pd:pandas.DataFrame>`; mangling your data into the appropriate
format should only take :doc:`one or two steps
<pd:getting_started/intro_tutorials/10_text_data>`. The important thing is to
have the texts in one column, and the document labels as the index.

.. code:: python

   corpus = tn.Corpus(series, lang="nl")
   # or alternately:
   corpus = tn.Corpus.from_df(df, doc_col="tekst", lang="nl")

If you do not specify ``doc_col``, **textnets** assumes that the first column
containing strings is the one you meant.

From a database or CSV file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use `Corpus` to load your documents from a database or
comma-separated value file using `from_sql` and `from_csv` respectively.

.. code:: python

   import sqlite3

   with sqlite3.connect("documents.db") as conn:
       articles = tn.Corpus.from_sql("SELECT title, text FROM articles", conn)

As before, you do can specify a ``doc_col`` to specify which column contains
your texts. You can also specify a ``label_col`` containing document labels. By
default, `from_sql` uses the first column as the ``label_col`` and the first
column after that containing strings as the ``doc_col``.

.. code:: python

   blog = tn.Corpus.from_csv("blog-posts.csv",
                             label_col="slug",
                             doc_col="summary"
                             sep=";")

Both `from_sql` and `from_csv` accept additional keyword arguments that are
passed to `pandas.read_sql` and `pandas.read_csv` respectively.

From Files
~~~~~~~~~~

Perhaps you have each document you want to include in your textnet stored on
disk in a separate text file. For such cases, `Corpus` comes with a utility,
`from_files`. You can pass it a path using a `globbing
<https://en.wikipedia.org/wiki/Glob_(programming)>`__ pattern:

.. code:: python

   corpus = tn.Corpus.from_files("/path/to/texts/*.txt")

You can also pass it a list of paths:

.. code:: python

   corpus = tn.Corpus.from_files(["kohl.txt", "schroeder.txt", "merkel.txt"],
                                 doc_labels=["Kohl", "Schröder", "Merkel"],
                                 lang="de")

You can optionally pass explicit labels for your documents using the argument
``doc_labels``. Without this, labels are inferred from file names by stripping
off the file suffix.

Break It Up
~~~~~~~~~~~

The textnet is built from chunks of texts. `Corpus` offers three methods for
breaking your texts into chunks: `tokenized`, `ngrams`, and `noun_phrases`. The
first breaks your texts up into individual words, the second into n-grams of
desired size, while the third looks for `noun phrases
<https://en.wikipedia.org/wiki/Noun_phrase>`__ such as “my husband,” “our prime
minister,” or “the virus.”

.. code:: python

   trigrams = corpus.ngrams(3)

.. code:: python

   np = corpus.noun_phrases(remove=["Lilongwe", "Mzuzu", "Blantyre"])

.. warning::
   For large corpora, some of these operations can be computationally intense.
   Use your friendly neighborhood HPC cluster or be prepared for your laptop to
   get hot.

An optional boolean argument, ``sublinear``, can be passed to `tokenized`,
`ngrams`, and `noun_phrases`. It decides whether to use sublinear (logarithmic)
scaling when calculating *tf-idf* term weights. The default is ``True``,
because sublinear scaling is considered good practice in the information
retrieval literature :cite:p:`Manning2008`, but there may be good reason to
turn it off.

Calling these methods results in another data frame, which we can feed to
`Textnet` to make our textnet.

Make Connections
----------------

A textnet is a `bipartite network
<https://en.wikipedia.org/wiki/Bipartite_graph>`__  of *terms* (words or
phrases) and *documents* (which often represent the people or groups who
authored them). We create the textnet from the processed corpus using the
`Textnet` class.

.. code:: python

   t = tn.Textnet(np)

`Textnet` takes a few optional arguments. The most important one is
``min_docs``. It determines how many documents a term must appear in to be
included in the textnet. A term that appears only in a single document creates
no link, so the default value is 2. However, this can lead to a very noisy
graph, and usually only terms that appear in a significant proportion of
documents really indicate latent topics, so it is common to pass a higher
value.

``connected`` is a boolean argument that decides whether only the largest
connected component of the resulting network should be kept. It defaults to
``False``.

``doc_attrs`` allows setting additional attributes for documents that become
node attributes in the resulting network. For instance, if texts represent
views of members of different parties, we can set a party attribute.

.. code:: python

   t = tn.Textnet(corpus.tokenized(), doc_attr=df[["party"]].to_dict())

Seeing Results
--------------

You are now ready to see the first results. `Textnet` comes with a utility
method, `plot <Textnet.plot>`, which allows you to quickly visualize the
bipartite network.

For bipartite network, it can be helpful to use a layout option, such as
``bipartite_layout``, ``circular_layout``, or ``sugiyama_layout``, which help
to spatially separate the two node types.

You may want terms that are used in more documents to appear bigger in the
plot. In that case, use the ``scale_nodes_by`` argument with the value
``degree``. Other useful options include ``label_term_nodes``,
``label_doc_nodes``, and ``label_edges``. These are all boolean options, so
your can enable them by passing the value ``True``.

Finally, enabling ``show_clusters`` will draw polygons around detected groups
of nodes with a community structure.

Projecting
----------

Depending on your research question, you may be interested either in how terms
or documents are connected. You can project the bipartite network into a
single-mode network of either kind.

.. code:: python

   groups = t.project(node_type=tn.DOC)
   print(groups.summary)

The resulting network only contains nodes of the chosen type (`DOC` or `TERM`).
Edge weights are calculated, and node attributes are maintained. The `m
<ProjectedTextnet.m>` property gives you access to the projected graph's
weighted adjacency matrix.

Like the bipartite network, the projected textnet also has a `plot
<ProjectedTextnet.plot>` method. This takes an optional argument, ``alpha``,
which can help "de-clutter" the resulting visualization by removing edges. The
value for this argument is a significance value, and only edges with a
significance value at or below the chosen value are kept. What remains in the
pruned network is called the "backbone" in the network science literature.
Commonly chosen values for ``alpha`` are in the range between 0.2 and 0.6 (with
lower values resulting in more aggressive pruning).

In visualizations of the projected network, you may want to scale nodes
according to centrality. Pass the argument ``scale_nodes_by`` with a value of
"betweenness," "closeness," "harmonic," "degree," "strength," or
"eigenvector_centrality." In the document network, you can also use textual
spanning, as demonstrated above.

Label nodes using the boolean argument ``label_nodes``. As above,
``show_clusters`` will mark groups of nodes with a community structure.

Analysis
--------

Use `top_cluster_nodes <TextnetBase.top_cluster_nodes>` to interpret the
community structure of your textnet. Clusters in the bipartite or word network
can be interpreted as discursive categories or latent themes in the corpus
:cite:p:`Gerlach2018`. Clusters in the document network can be interpreted as
groupings.

The tutorial above gives some examples of using centrality measures to analyze
your corpus. Aside from `top_betweenness`, the package also provides the
methods `top_closeness` (weighted closeness), `top_harmonic` (weighted harmonic
centrality), `top_degree` (unweighted degree), `top_strength` (weighted
degree), `top_ev` (eigenvector centrality), `top_pagerank` (PageRank
centrality), and `top_spanning` (textual spanning). In the bipartite network,
you can use `top_birank`, `top_hits` and `top_cohits` to see nodes ranked by
variations of a bipartite centrality measure :cite:p:`He2017`. By default, they
each output the ten top nodes for each centrality measure.

Saving
------

You can save both the network that underlies a textnet as well as
visualizations. Assuming you want to save the projected term network, called
``words``, that we created above, you can do so as follows:

.. code:: python

   words.save_graph("term_network.gml")

This will create a file in the current directory in Graph Modeling Language
(GML) format. This can then be opened by Pajek, yEd, Gephi and other programs.
Consult the docs for `Textnet.save_graph` for a list of supported formats.

If instead you want to save a plot of a network, pass the ``target`` keyword to
the `Textnet.plot` method.

.. code:: python

   words.plot(label_nodes=True, color_clusters=True, target="term_network.svg")

Supported file formats include PNG, EPS and SVG.
