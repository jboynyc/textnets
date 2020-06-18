===============================================
Textnets: automated text analysis with networks
===============================================

.. image:: https://travis-ci.org/jboynyc/textnets.svg?branch=master
   :target: https://travis-ci.org/jboynyc/textnets
   :alt: Travis-CI Status

.. image:: https://readthedocs.org/projects/textnets/badge/?version=stable
   :target: https://textnets.readthedocs.io/en/stable/?badge=stable
   :alt: Documentation Status

.. image:: https://pyup.io/repos/github/jboynyc/textnets/shield.svg
   :target: https://pyup.io/repos/github/jboynyc/textnets/
   :alt: Python Dependency Freshness

.. image:: https://zenodo.org/badge/114368834.svg
   :target: https://zenodo.org/badge/latestdoi/114368834
   :alt: Latest DOI so you can please cite this software

**textnets** represents collections of texts as networks of documents and
words. This provides novel possibilities for the visualization and analysis of
texts.

.. figure:: https://textnets.readthedocs.io/en/dev/_static/impeachment-statements.svg
   :alt: Bipartite network graph

   Network of U.S. Senators and words used in their official statements
   following the acquittal vote in the Senate impeachment trial (`source
   <https://www.jboy.space/blog/enemies-foreign-and-partisan.html>`_).

This is a Python implementation of `Chris Bail's textnets package for R`_.  It
is free software under the terms of the GNU General Public License v3.

.. _`Chris Bail's textnets package for R`: https://github.com/cbail/textnets/

The idea underlying **textnets** is presented in this paper:

  Christopher A. Bail, "`Combining natural language processing and network
  analysis to examine how advocacy organizations stimulate conversation on social
  media`__," *Proceedings of the National Academy of Sciences of the United States
  of America* 113, no. 42 (2016), 11823–11828, doi:10.1073/pnas.1607151113.

__ https://doi.org/10.1073/pnas.1607151113

Features
--------

**textnets** builds on the state-of-the-art library `spacy`_ for
natural-language processing and `igraph`_ for network analysis. It uses the
`Leiden algorithm`_ for community detection, which is able to perform community
detection on the bipartite (word–group) network.

.. _`Leiden algorithm`: https://arxiv.org/abs/1810.08473
.. _`igraph`: http://igraph.org/python/
.. _`spacy`: https://spacy.io/

**textnets** seamlessly integrates with `pandas`_ and other parts of Python's
excellent `scientific stack`_. That means that you can use **textnets** in
Jupyter notebooks to analyze and visualize your data!

.. _`pandas`: https://pandas.io/
.. _`scientific stack`: https://numfocus.org/

Read `the documentation <https://textnets.readthedocs.io>`_ to find out more
about the package's features.

Learn More
----------

==================  =============================================
**Documentation**   https://textnets.readthedocs.io/
**Repository**      https://github.com/jboynyc/textnets
**Issues & Ideas**  https://github.com/jboynyc/textnets/issues
**PyPI**            https://pypi.org/project/textnets/
**DOI**             `10.5281/zenodo.3866676 <https://doi.org/10.5281/zenodo.3866676>`_
==================  =============================================
