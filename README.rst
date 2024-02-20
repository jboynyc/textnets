=====================================
Textnets: text analysis with networks
=====================================

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/jboynyc/textnets-binder/trunk?filepath=Tutorial.ipynb
   :alt: Launch on Binder

.. image:: https://github.com/jboynyc/textnets/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/jboynyc/textnets/actions/workflows/ci.yml
   :alt: CI Status

.. image:: https://readthedocs.org/projects/textnets/badge/?version=stable
   :target: https://textnets.readthedocs.io/en/stable/?badge=stable
   :alt: Documentation Status

.. image:: https://joss.theoj.org/papers/10.21105/joss.02594/status.svg
   :target: https://doi.org/10.21105/joss.02594
   :alt: Published in Journal of Open Source Software

**textnets** represents collections of texts as networks of documents and
words. This provides novel possibilities for the visualization and analysis of
texts.

.. figure:: https://textnets.readthedocs.io/en/dev/_static/impeachment-statements.svg
   :alt: Bipartite network graph

   Network of U.S. Senators and words used in their official statements
   following the acquittal vote in the 2020 Senate impeachment trial (`source
   <https://www.jboy.space/blog/enemies-foreign-and-partisan.html>`_).

**textnets** is free software under the terms of the GNU General Public License
v3.

The ideas underlying **textnets** are presented in this paper:

  Christopher A. Bail, "`Combining natural language processing and network
  analysis to examine how advocacy organizations stimulate conversation on social
  media`__," *Proceedings of the National Academy of Sciences of the United States
  of America* 113, no. 42 (2016), 11823–11828, doi:10.1073/pnas.1607151113.

__ https://doi.org/10.1073/pnas.1607151113

Initially begun as a Python implementation of `Chris Bail's textnets package
for R`_, **textnets** now comprises several unique features for term extraction
and weighing, visualization, and analysis.

.. _`Chris Bail's textnets package for R`: https://github.com/cbail/textnets/

Features
--------

**textnets** builds on `spaCy`_, a state-of-the-art library for
natural-language processing, and `igraph`_ for network analysis. It uses the
`Leiden algorithm`_ for community detection, which is able to perform community
detection on the bipartite (word–group) network.

.. _`igraph`: http://igraph.org/python/
.. _`Leiden algorithm`: https://doi.org/10.1038/s41598-019-41695-z
.. _`spaCy`: https://spacy.io/

**textnets** is installable using the ``pip`` and ``nix`` package managers. It
requires Python 3.9 or higher.

**textnets** integrates seamlessly with Python's excellent `scientific stack`_.
That means that you can use **textnets** to analyze and visualize your data in
Jupyter notebooks!

.. _`scientific stack`: https://scientific-python.org

Read `the documentation <https://textnets.readthedocs.io>`_ to learn more about
the package's features.

Citation
--------

Using **textnets** in a scholarly publication? Please cite this paper:

.. code-block:: bibtex

   @article{Boy2020,
     author   = {John D. Boy},
     title    = {textnets},
     subtitle = {A {P}ython Package for Text Analysis with Networks},
     journal  = {Journal of Open Source Software},
     volume   = {5},
     number   = {54},
     pages    = {2594},
     year     = {2020},
     doi      = {10.21105/joss.02594},
   }

Learn More
----------

==================  =============================================
**Documentation**   https://textnets.readthedocs.io/
**Repository**      https://github.com/jboynyc/textnets
**Issues & Ideas**  https://github.com/jboynyc/textnets/issues
**PyPI**            https://pypi.org/project/textnets/
**FOSDEM ’22**      https://fosdem.org/2022/schedule/event/open_research_textnets/
**DOI**             `10.21105/joss.02594 <https://doi.org/10.21105/joss.02594>`_
**Archive**         `10.5281/zenodo.3866676 <https://doi.org/10.5281/zenodo.3866676>`_
==================  =============================================

.. image:: https://textnets.readthedocs.io/en/dev/_static/textnets-logo.svg
   :alt: textnets logo
   :target: https://textnets.readthedocs.io
   :align: center
   :width: 140
