=======
History
=======

0.5.3 (2021-09-24)
------------------
* Adds Catalan, Macedonian and Russian language models.
* Significantly speeds up backbone extraction by implementing the disparity
  filter integrand in Cython. (If the compiled extension cannot be loaded for
  some reason, it falls back on an interpreted function.)
* `PyPI <http://pypi.org/project/textnets>`_ *should* now receive binary wheels
  for Mac, Windows and Linux (via GitHub Actions) to ease installation on each
  platform.
* Improved type annotations.
* Update several dependencies.

0.5.2 (2021-08-24)
------------------
* Improve the handling of edge cases when initializing the `Corpus` and
  `Textnet` classes, such as empty data being provided.
* Added ability to run the tutorial in the documentation interactively using
  `thebe <https://thebelab.readthedocs.io/>`_.
* Update to spacy 3.1 and bump other dependencies.

0.5.1 (2021-07-06)
------------------
* Adds `Corpus.ngrams` method as alternative to `Corpus.noun_phrases`. This is
  useful when working in languages that do not have noun chunks, such as
  Chinese.
* Fixes a bug in `Corpus.from_files`.
* Introduces HTML representations of core classes for nicer integration in
  Jupyter notebooks.
* Updates several dependencies.

0.5.0 (2021-06-28)
------------------
* Migrate continuous integration testing from Travis to GitHub Actions.
* Continuous integration tests now run for MacOS and Windows too.
* Update to Spacy 3 and bump other dependency versions.
* Improvements to documentation.
* Handle dependencies and build project using Poetry (PEP 517 and 518).
* Remove deprecated command-line interface.

0.4.11 (2020-11-09)
-------------------
* Python 3.9 compatibility!
* Updated documentation with conda-forge installation option.
* Bump versions for numerous dependencies.

0.4.10 (2020-09-14)
-------------------
* Add ``cairocffi`` dependency and update installation docs.
* Bump ``leidenalg`` dependency to version 0.8.1.

0.4.9 (2020-07-15)
------------------
* Add ``color_clusters`` option to `Textnet` plotting methods. This colors
  nodes according to their partition using a bespoke color palette.

0.4.8 (2020-07-10)
------------------
* The `Corpus` class now handles missing data (#13).
* Support for more corpus languages. If no statistical language model is
  available, `Corpus` tries to use a basic ("blank") model.
* Improved documentation around dependencies and language support.
* Added tests.

0.4.7 (2020-07-01)
------------------
* No substantive change from previous release.

0.4.6 (2020-07-01)
------------------
* Bump spacy dependency to version 2.3 because it includes several new language
  models.

0.4.5 (2020-06-29)
------------------
* `Textnet.plot` and `ProjectedTextnet.plot` now accept arguments to selectively
  suppress node or edge labels. ``node_label_filter`` and ``edge_label_filter``
  take a function that is mapped to the iterator of nodes and edges. Only nodes
  or edges for which the function returns ``True`` are displayed in the plot.
* `Corpus` now has a useful string representation.
* Documentation updates, particularly to show the label filter functionality.

0.4.4 (2020-06-19)
------------------

* Methods to report centrality measures in `TextnetBase` now return
  `pandas.Series` objects. This has some nice benefits, like seeing node labels
  alongside centrality measures and being able to call ``.hist()`` on them to
  visualize the distribution.
* Scaling of nodes by centrality in plots should bring out differences more
  clearly now.
* Improved and expanded tutorial. Among other things, it now uses short codes
  to specify language models.

0.4.3 (2020-06-17)
------------------

* Python 3.7 compatibility is here.
* New ``circular_layout`` option for `Textnet.plot`. This is based on "`Tidier
  Drawings <https://www.reingold.co/graph-drawing.shtml>`_" and looks very nice
  for some bipartite graphs.
* String representation of `Textnet` instances now gives helpful information.
* Updated documentation to note changed Python version requirement.

0.4.2 (2020-06-16)
------------------

* `ProjectedTextnet.plot` now takes an argument, ``alpha``, that allows for
  pruning the graph in order to visualize its "backbone." This is useful when
  working with hairball graphs, which is common when creating textnets. Right
  now, it uses Serrano et al.'s disparity filter. That means that edges with an
  alpha value greater than the one specified are discarded, so lower values
  mean more extreme pruning.
* Language models can now be specified using a short ISO language code.
* Bipartite networks can now be plotted using a layered layout (by Kozo
  Sugiyama). Simply pass ``sugiyama_layout=True`` to `Textnet.plot`.
* Incremental improvements to documentation.

0.4.1 (2020-06-12)
------------------

* Documented `TextnetBase` methods to output lists of nodes ranked by various
  centrality measures: `top_betweenness` and several more.
* Added `top_cluster_nodes` to output list of top nodes per cluster found via
  community detection. This is useful when trying to interpret such clusters as
  themes/topics (in the projected word-to-word graph) or as groupings (in the
  document-to-document graph).
* Small additions to documentation.

0.4.0 (2020-06-11)
------------------

Lots of changes, some of them breaking, but overall just providing nicer
abstractions over the underlying pandas and igraph stuff.

* Introduced `TextnetBase` and `ProjectedTextnet` classes, and made `Textnet` a
  descendant of the former.
* Improved code modularity to make it easier to add features.
* `Corpus` is now based on a Series rather than a DataFrame.
* Added methods for creating an instance of `Corpus`: `from_df`, `from_csv`,
  `from_sql`.
* Expanded and improved documentation.
* Added bibliography to documentation using a Sphinx bibtex plugin.
* A first contributor!

0.3.6 (2020-06-03)
------------------

* Small change to *finally* get automatic deployments to PyPI to work.

0.3.5 (2020-06-03)
------------------

* Overall improvements to documentation.
* Added ``label_edges`` argument to `Textnet.plot`.

0.3.4 (2020-06-02)
------------------

* Integrated self-contained example that can be downloaded as Jupyter notebook
  into tutorial.
* Still trying to get automatic deployments to PyPI working.

0.3.3 (2020-06-02)
------------------

* More documentation.
* Attempt to get automatic deployments to PyPI working.

0.3.2 (2020-06-02)
------------------

* Set up continuous integration with Travis CI.
* Set up pyup.io dependency safety checks.
* Expanded documentation.
* A logo!

0.3.2 (2020-05-31)
------------------

* Further improvements to documentation.

0.3.1 (2020-05-31)
------------------

* Improvements to documentation.

0.3.0 (2020-05-31)
------------------

* First release on PyPI.
