================================================
Textnets: automated text analysis with networks.
================================================

.. image:: https://zenodo.org/badge/114368834.svg
   :target: https://zenodo.org/badge/latestdoi/114368834

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

The library builds on the state-of-the-art library `spacy`_ for
natural-language processing and `igraph`_ for network analysis. It uses the
`Leiden algorithm`_ for community detection, which is able to perform community
detection on the bipartite (word–group) network. Textnets seamlessly integrates
with `pandas`_ and other parts of `Python's scientific stack`_.

.. _`Leiden algorithm`: https://arxiv.org/abs/1810.08473
.. _`igraph`: http://igraph.org/python/
.. _`spacy`: https://spacy.io/
.. _`pandas`: https://pandas.io/
.. _`Python's scientific stack`: https://numfocus.org/

For a demonstration of some of this package's features, take a look at `this
notebook`__.

__ https://gist.github.com/jboynyc/d5a850c04c5ef8d9007a27bf22112212

In addition to providing a Python library, **textnets** can also be used as a
command-line tool to generate network graphs from text corpora.

.. code:: bash

    $ textnets --lex noun_phrases --node-type groups ~/nltk_data/corpora/state_union | gzip > sotu_groups.graphmlz

Run ``textnets --help`` for usage instructions.

Learn More
----------

==================  =============================================
**Documentation**   https://textnets.readthedocs.io/
**Repository**      https://github.com/jboynyc/textnets
**Issues & Ideas**  https://github.com/jboynyc/textnets/issues
**PyPI**            https://pypi.org/project/textnets/
**DOI**             https://zenodo.org/badge/latestdoi/114368834
==================  =============================================
