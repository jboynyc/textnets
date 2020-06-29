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
   :jupyter-download:notebook:`tutorial`.

To use **textnets** in a project, you typically need the following imports:

.. jupyter-execute::

   from textnets import Corpus, Textnet

For the purposes of demonstration, we also import the bundled example data:

.. jupyter-execute::

   from textnets import examples

Construct the corpus from the example data:

.. jupyter-execute::

   corpus = Corpus(examples.moon_landing)

What is this `moon_landing` example all about?

.. jupyter-execute::

   display(examples.moon_landing)

.. note::

   Hat tip to Chris Bail for this example data!

Next, we create the textnet:

.. jupyter-execute::

   tn = Textnet(corpus.tokenized(), min_docs=1)

We're using `tokenized` with all defaults, so **textnets** is removing stop
words, applying stemming, and removing punctuation marks, numbers, URLs and the
like. However, we're overriding the default setting for ``min_docs``, opting to
keep even words that appear in only one document (that is, a single newspaper
headline).

Let's take a look:

.. jupyter-execute::

   tn.plot(label_term_nodes=True,
           label_doc_nodes=True,
           show_clusters=True)

The ``show_clusters`` options marks the partitions found by the Leiden
community detection algorithm (see :doc:`here <la:multiplex>`). It identifies
document--term groups that appear to form part of the same theme in the texts.

You may be wondering: why is the moon drifting off by itself in the network
plot? That's because the word moon appears exactly once in each document, so
its *tf-idf* value for each document is 0.

We can also visualize the projected networks.

First, the network of newspapers:

.. jupyter-execute::

    papers = tn.project(node_type='doc')
    papers.plot(label_nodes=True)

As before in the bipartite projection, we can see the *Houston Chronicle*,
*Chicago Tribune* and *Los Angeles Times* cluster more closely together.

Next, the term network:

.. jupyter-execute::

   words = tn.project(node_type='term')
   words.plot(label_nodes=True,
              show_clusters=True)

Aside from visualization, we can also analyze our corpus using network metrics.
For instance, documents with high betweenness centrality (or "cultural
betweenness"; :cite:`Bail2016`) might link together themes, thereby stimulating
exchange across symbolic divides.

.. jupyter-execute::

   papers.top_betweenness()

As we can see, the *Los Angeles Times* is a cultural bridge linking the
headline themes of the East Coast newspapers to the others.

.. jupyter-execute::

   words.top_betweenness()

It's because the *Times* uses the word "walk" in its headline, linking the "One
Small Step" cluster to the "Man on Moon" cluster.

We can produce the term graph plot again, this time scaling nodes according to
their betweenness centrality.

.. jupyter-execute::

   words.plot(label_nodes=True,
              scale_nodes_by='betweenness')

In addition to `betweenness`, we could also use `closeness` and
`eigenvector_centrality` to scale nodes. We can also filter node labels,
labeling only those nodes that have a betweenness centrality score above the
median. This can be useful in high-order graphs where labeling every single
node would cause too much clutter.

.. jupyter-execute::

   words.plot(label_nodes=True,
              node_label_filter=lambda n: n.betweenness() > words.betweenness.median(),
              scale_nodes_by='betweenness')

Wrangling Text & Mangling Data
------------------------------

How to go from this admittedly contrived example to working with your own data?
The following snippets are meant to help you get started. The first thing is to
get your data in the right shape.

A textnet is built from a collection—or *corpus*—of texts, so we use the
`Corpus` class to get our data ready. Each of the following snippets assumes
that you have imported `Corpus` and `Textnet` like in the preceding example.

From Pandas
~~~~~~~~~~~

You may already have your texts in a Python data structure. `Corpus` can read
documents directly from pandas' `Series <pd:pandas.Series>` or `DataFrame
<pd:pandas.DataFrame>`; mangling your data into the appropriate format should
only take :doc:`one or two easy steps <pd:getting_started/dsintro>`. The
important thing is to have the texts in one column, and the document labels as
the index.

.. code:: python

   corpus = Corpus(series, lang='nl')
   # or alternately:
   corpus = Corpus.from_df(df, doc_col='tekst', lang='nl')

If you do not specify ``doc_col``, **textnets** assumes that the first column
containing strings is the one you meant.

You can specify which `language model <https://spacy.io/models>`__ you would
like to use using the ``lang`` argument. The default is English, but you don’t
have to be monolingual to use **textnets**. (See `LANGS` for supported
languages.)

From a database or CSV file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use `Corpus` to load your documents from a database or
comma-separated value file using `from_sql` and `from_csv` respectively.

.. code:: python

   import sqlite3

   with sqlite3.connect('documents.db') as conn:
       articles = Corpus.from_sql('SELECT title, text FROM articles', conn)

As before, you do can specify a ``doc_col`` to specify which column contains
your texts. You can also specify a ``label_col`` containing document labels. By
default, `from_sql` uses the first column as the ``label_col`` and the first
column after that containing strings as the ``doc_col``.

.. code:: python

   blog = Corpus.from_csv('blog-posts.csv',
                          label_col='slug',
                          doc_col='summary'
                          sep=';')

Both `from_sql` and `from_csv` accept additional keyword arguments that are
passed to `pandas.read_sql` and `pandas.read_csv` respectively.

From Files
~~~~~~~~~~

Perhaps you have each document you want to include in your textnet stored on
disk in a separate text file. For such cases, `Corpus` comes with a utility,
`from_files()`. You can simply pass a path to it using a `globbing
<https://en.wikipedia.org/wiki/Glob_(programming)>`__ pattern:

.. code:: python

   corpus = Corpus.from_files('/path/to/texts/*.txt')

You can also pass it a list of paths:

.. code:: python

   corpus = Corpus.from_files(['kohl.txt', 'schroeder.txt', 'merkel.txt'],
                              doc_labels=['Kohl', 'Schröder', 'Merkel'],
                              lang='de')

This example demonstrates another feature: you can optionally pass explicit
labels for your documents using the argument ``doc_labels``. Without this,
labels are inferred from file names (by stripping off the file suffix).


Break It Up
~~~~~~~~~~~

The textnet is built from chunks of texts. `Corpus` offers two
methods for breaking your texts into chunks: `tokenized` and
`noun_phrases`. The first breaks your texts up into individual
words, while the latter looks for `noun
phrases <https://en.wikipedia.org/wiki/Noun_phrase>`__ such as “my
husband,” “our prime minister,” or “the virus.”

.. code:: python

   np = corpus.noun_phrases(remove=['Lilongwe', 'Mzuzu', 'Blantyre'])

.. warning::
   For large corpora, this can be a computationally intense task. Use your
   friendly neighborhood HPC cluster or be prepared for your laptop to get hot.

The result of this is another data frame, which we can feed to `Textnet` to
make our textnet.

Make Connections
----------------

A textnet is a `bipartite network
<https://en.wikipedia.org/wiki/Bipartite_graph>`__  of *terms* (words or
phrases) and *documents* (which often represent the people or groups who
authored them). We create the textnet from the processed corpus using the
`Textnet` class.

.. code:: python

   tn = Textnet(np)

`Textnet` takes a few optional arguments. The most important one is
``min_docs``. It determines how many documents a term must appear in to be
included in the textnet. A term that appears only in a single document creates
no link, so the default value is 2. However, this can lead to a very noisy
graph, and usually only terms that appear in a significant proportion of
documents really indicate latent topics, so it is common to pass a higher
value.

A boolean argument, ``sublinear``, decides whether to use sublinear
(logarithmic) scaling when calculating *tf-idf* for edge weights. The default
is ``True`` because sublinear scaling is considered good practice in the
information retrieval literature (:cite:`Manning2008`), but there may be good
reason to turn it off.

``doc_attr`` allows setting additional attributes for documents that become
node attributes in the resulting network graph. For instance, if texts
represent views of members of different parties, we can set a party attribute.

.. code:: python

   tn = Textnet(corpus.tokenized(), doc_attr=df[['party']].to_dict())

Seeing Results
--------------

You are now ready to see the first results. `Textnet` comes with a utility
method, `plot <Textnet.plot>`, which allows you to quickly visualize the bipartite
graph.

For bipartite graphs, it can be helpful to use a layout option, such as
``bipartite_layout``, ``circular_layout``, or ``sugiyama_layout``, which help
to spatially separate the two node types.

You may want terms that are used in more documents to appear bigger in the
graph. In that case, use the ``scale_nodes_by`` argument with the value
``degree``. Other useful options include ``label_term_nodes``,
``label_doc_nodes``, and ``label_edges``. These are all boolean options, so
simply pass the value ``True`` to enable them.

Finally, enabling ``show_clusters`` will draw polygons around detected groups
of nodes with a community structure.

Projecting
----------

Depending on your research question, you may be interested either in how terms
or documents are connected. You can project the bipartite network into a
single-mode network of either kind.

.. code:: python

   groups = tn.project(node_type='doc')
   groups.summary()

The resulting network only contains nodes of the chosen type (``doc`` or
``term``). Edge weights are calculated, and node attributes are maintained.

Like the bipartite network, the projected textnet also has a `plot
<ProjectedTextnet.plot>` method. This takes an optional argument, ``alpha``,
which can help "de-clutter" the resulting visualization by removing edges. The
value for this argument is a significance value, and only edges with a
significance value at or below the chosen value are kept. What remains in the
pruned graph is called the "backbone" in the network science literature.
Commonly chosen values for ``alpha`` are in the range between 0.2 and 0.6 (with
lower values resulting in more aggressive pruning).

In visualizations of the projected network, you may want to scale nodes
according to centrality. Pass the argument ``scale_nodes_by`` with a value of
"betweenness," "closeness," "strength" or "eigenvector_centrality."

Label nodes using the boolean argument ``label_nodes``. As above,
``show_clusters`` will mark groups of nodes with a community structure.

From the Command Line
---------------------

In addition to providing a Python package, **textnets** can also be used as a
command-line tool.

.. code:: bash

   $ textnets --lex noun_phrases --node-type groups ~/nltk_data/corpora/state_union | gzip > sotu_groups.graphmlz

Run ``textnets --help`` for usage instructions.

.. warning::
   The command-line tool is not currently maintained and may be removed in
   future releases.
