=======
History
=======

=======
0.9.3 (unreleased)
------------------
* Updates to igraph 0.10.5.
* Updates to spaCy 3.6.0, bringing support for Slovenian.
* Add dependency on ``spacy-lookups-data`` for better support of languages
  without pre-trained models (e.g., Turkish).

0.9.2 (2023-06-28)
------------------
* Improves documentation.
* Updates to scipy 1.10.
* Fixes how the compiled extension is built.

0.9.1 (2023-06-16)
------------------
* Only fixes deployment to PyPI, otherwise unchanged from previous version.

0.9.0 (2023-06-15)
------------------
* Improves documentation.
* Parallelizes execution of NLP and backbone extraction for large corpora and
  networks.
* Uses sparse matrices for some operations to avoid running out of memory.
* Makes other improvements to efficiency.
* Introduces a `NodeType` enum to differentiate document and term nodes.
* Updates to pandas 2.0 and spaCy 3.5.3.
* Now also tested for compatibility with PyPy 3.9!

0.8.8 (2023-03-21)
------------------
* Fixes bug in disparity filter.
* Updates to spaCy 3.5.1, along various other dependencies.
* Improves testing.

0.8.7 (2023-02-10)
------------------
* Updates to spaCy 3.5 and igraph 0.10.4.
* When initializing `Corpus` with data that includes duplicated document
  labels, issue a warning, and concatenate documents with shared labels.
  (Implemented in response to `#54
  <https://github.com/jboynyc/textnets/issues/54>`__).
* Various code quality improvements.

0.8.6 (2022-11-30)
------------------
* Improves documentation.
* Progress bar for long-running operations (NLP and disparity filter). The
  progress bar is not shown during non-interactive use. To disable, set
  ``tn.params["progress_bar"]`` to ``False``.
* Python 3.11 compatibility now confirmed on Windows, too.

0.8.5 (2022-10-28)
------------------
* Python 3.11 compatibility!
* Adds an optional parameter ``remove_weak_edges`` to `Textnet` to create a
  sparser graph.
* Adds `ProjectedTextnet.m` property to access one-mode graphs' adjacency
  matrices.
* Adds Stoltz and Taylor's textual spanning measure.
* Adds `TextnetBase.cluster_strength` and `TextnetBase.cluster_local_cc` to
  calculate the weighted degree and local clustering coefficient of nodes
  within the subgraph formed by the cluster they belong to.
* Improves display of top nodes per cluster (`top_cluster_nodes`).
* Adds Korean language code.
* Fixes several bugs that occurred when setting document attributes.

0.8.4 (2022-10-05)
------------------
* Updates to spaCy 3.4.1, igraph 0.10.1, and leidenalg 0.9.
* Adds Ukrainian language code.
* Improves type checking.

0.8.3 (2022-07-15)
------------------
* Updates to spaCy 3.4, bringing support for Croatian to **textnets**.
* Updates various other dependencies.
* Adds optional dependency for experimental Formal Concept Analysis features.
  To install, run ``pip install textnets[fca]``. `Graphviz
  <https://graphviz.org/>`__ must also be installed separately for lattice
  visualization purposes.

0.8.2 (2022-06-28)
------------------
* Makes `TextnetBase` an abstract base class, since it is not meant to be
  instantiated. Implements separate graph partition methods for the classes
  `Textnet` and `ProjectedTextnet` to work around an occasional bug.
* Adds Finnish and Swedish language codes.
* Improves type hints.
* Adds dark mode to docs!

0.8.1 (2022-06-27)
------------------
* Fixes `bug #36 <https://github.com/jboynyc/textnets/issues/36>`__.
* Updates dependencies, including ``igraph`` and spaCy.

0.8.0 (2022-05-06)
------------------
* Removes Python 3.7 compatibility.
* Fixes a bug in the HTML representation of the top-level module.
* Updates dependencies, including spaCy.

0.7.1 (2022-02-08)
------------------
* Fixes #35 (invisible edges when scaling by weight).
* Updates some dependencies.

0.7.0 (2021-11-12)
------------------
* Adds abilitiy to save and load an instance of `Corpus`, `Textnet` and
  `params` to and from file using `Corpus.save`, `load_corpus`, `Textnet.save`,
  `load_textnet`, `params.save` and `params.load`. The same file can be used
  for all three kinds of objects, so all relevant data for a project can be
  saved in one file.
* Some further optimization of backbone extraction.
* Adds bipartite centrality measures (HITS, CoHITS and BiRank) and a bipartite
  clustering coefficient.
* Improved testing and type hints.
* Expanded documentation with advanced topics, including the new save/load
  feature and interacting with other libraries for network analysis and machine
  learning. Docs now use the PyData theme.
* Improvements to visualization. When plotting, nodes and edges can now be
  scaled by any attribute.
* Breaking change: Term weighing now happens in the ``corpus`` submodule, so
  the ``sublinear`` argument has to be passed to the methods for term
  extraction (``tokenized``, ``noun_phrases`` and ``ngrams``). This change will
  make it easier to add additional term extraction and weighing options.
* Adds ``tn.init_seed()`` utility to quickly initialize pseudorandom number
  generator.
* Adds Python 3.10 compatibility.
* Updates dependencies, including ``igraph`` with some relevant upstream
  changes contributed by yours truly, as well as spaCy.

0.6.0 (2021-10-14)
------------------
* Adds `params` as a container for global parameters. This makes it possible to
  fix the random seed and to change the resolution parameter for the community
  detection algorithm, among others. If the parameter ``autodownload`` is set
  to true, **textnets** will attempt to download all required spaCy language
  models automatically.
* Added HTML representation for the root module that displays versions of key
  dependencies.
* Added back string representations of `Corpus` and `TextnetBase`-derived
  classes.
* Adds a `Corpus.from_dict` method.
* `Corpus` now exposes the ``lang`` attribute, so the corpus language can be
  set after initialization of a class instance.
* The bipartite layout optionally used by `Textnet.plot` is now horizontal, so
  node types are arranged in columns rather than rows. That way node labels are
  less likely to overlap.
* Adds ``label_nodes`` argument to the `Textnet.plot` method to label both types
  of nodes. Defaults to ``False``.
* Adds ``node_opacity`` and ``edge_opacity`` arguments for `Textnet.plot`.
* Makes polygons marking clusters more visually appealing by adding opacity.
* Probably fixes `a bug <https://github.com/jboynyc/textnets/issues/30>`_ that
  would occasionally result in an exception being raised during plotting
  (``IndexError: color index too large``).
* When initializing an instance of the `Textnet` class, you can now optionally
  pass the argument ``connected=True``, in which case only the largest
  component of the underlying network is kept. When creating a one-mode
  projection using `Textnet.project`, a ``connected`` argument can also be
  passed.
* Adds `TextnetBase.save_graph` to save the underlying graph (for instance, for
  further processing in Gephi).
* Improved and extended documentation and docstrings.
* Update dependencies.

0.5.4 (2021-09-24)
------------------
* Fix the cross-platform build and deploy pipeline.
* Create binary packages for conda-forge.
* Otherwise, no substantive change from previous release.

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
