========
Tutorial
========

To use **textnets** in a project, you need the following import::

    from textnets import Corpus, Textnet

Wrangling Text & Mangling Data
------------------------------

A textnet is built from a collection – or *corpus* – of texts, so we use
the `Corpus` class to get our data ready.

From a Data Frame
~~~~~~~~~~~~~~~~~

You may already have your texts in a Python data structure. `Corpus`
can read documents directly from a `pandas <https://pandas.io>`__
`DataFrame`; mangling your data into the appropriate format should
only take `one or two easy
steps <https://pandas.pydata.org/docs/getting_started/dsintro.html#from-dict-of-series-or-dicts>`__.
The important thing is to have the texts in one column, and the document
labels as the index.

.. code:: python

   corpus = Corpus(df, doc_col='tekst', lang='nl_core_news_sm')

If you do not specify ``doc_col``, **textnets** will assume that the
first column containing strings is the one you meant.

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
                              lang='de_core_news_sm')

The above example demonstrates two other features:

1. You can optionally pass explicit labels for your documents using the
   argument ``doc_labels``. Without this, labels are inferred from file
   names (by stripping off the file suffix).
2. You can specify which `language model <https://spacy.io/models>`__
   you would like to use. The default is English, but you don’t have to
   be monolingual to use **textnets**.

Break It Up
~~~~~~~~~~~

The textnet is built from chunks of texts. `Corpus` offers two
methods for breaking your texts into chunks: `tokenized` and
`noun_phrases`. The first breaks your texts up into individual
words, while the latter looks for `noun
phrases <https://en.wikipedia.org/wiki/Noun_phrase>`__ such as “my
husband,” “our prime minister,” or “the virus.”

.. code:: python

   np = corpus.noun_phrases(remove=['Lilongwe'])

The result of this is another data frame, which we can feed to `Textnet` to
make our textnet.

Make Connections
----------------

A textnet is a `bipartite network <https://en.wikipedia.org/wiki/Bipartite_graph>`__  of *terms* (words or phrases) and *documents* (which often represent the people or groups who authored them). We create the textnet from the processed corpus using the `Textnet` class.

.. code:: python

   tn = Textnet(np)

`Textnet` takes a few optional arguments. The most important one is ``min_docs``. It determines how many documents a term must appear in to be included in the textnet. A term that appears only in a single document creates no link, so the default value is 2. However, this can lead to a very noisy graph, and usually only terms that appear in a significant proportion of documents really indicate latent topics, so it is common to pass a higher value.

A boolean argument, ``sublinear``, decides whether to use sublinear (logarithmic) scaling when calculating tf-idf for edge weights. The default is ``True`` because sublinear scaling is considered good practice in the information retrieval literature, but there may be good reason to disable it.

``doc_attr`` allows setting additional attributes for documents that become node attributes in the resulting network graph. For instance, if texts represent views of members of different parties, we can set a party attribute.

.. code:: python

   tn = Textnet(corpus.tokenized(), doc_attr=df[['party']].to_dict())

Seeing Results
--------------

You are now ready to see the first results. `Textnet` comes with a utility method, `plot`, which allows you to quickly visualize the bipartite graph.

Projecting
----------

Depending on your research question, you may be interested either in how terms or documents are connected. For that purpose, you can project the bipartite network into a single-mode network.

.. code:: python

   groups = tn.project(node_type='doc')
   groups.summary()

The resulting network will only contain nodes of one type. Edge weights are calculated, and node attributes are maintained.

*to be continued*
