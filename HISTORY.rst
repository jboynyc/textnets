=======
History
=======

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
