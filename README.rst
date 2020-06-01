========
Textnets
========

--------------------------------------
Automated text analysis with networks.
--------------------------------------

.. image:: https://zenodo.org/badge/114368834.svg
   :target: https://zenodo.org/badge/latestdoi/114368834

This is a Python implementation of `Chris Bail's textnets package for R`_.  It
is free software under the terms of the GNU General Public License v3.

.. _`Chris Bail's textnets package for R`: https://github.com/cbail/textnets/

The underlying idea behind textnets is presented in this paper:

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
with Pandas and other parts of Python's scientific stack.

.. _`Leiden algorithm`: https://arxiv.org/abs/1810.08473
.. _`igraph`: http://igraph.org/python/
.. _`spacy`: http://spacy.io/

For a demonstration of this package's features, take a look at `this
notebook`_.

.. _`this notebook`: https://gist.github.com/jboynyc/d5a850c04c5ef8d9007a27bf22112212

In addition to providing a Python library, textnets can also be used as a
command-line tool to generate network graphs from text corpora::

    $ textnets --lex noun_phrases --node-type groups ~/nltk_data/corpora/state_union | gzip > sotu_groups.graphmlz

Run ``textnets --help`` for usage instructions.

Installing
----------

Textnets is now on the `Python Package Index`_, so it can be installed using ``pip``. In a `virtual environment`_, run ``pip install textnets``. You may also have to install an appropriate `language model`_ by issuing a command like:: 

    $ python -m spacy download en_core_web_sm

.. _`Python Package Index`: https://pypi.org/project/textnets/
.. _`virtual environment`: https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments
.. _`language model`: https://spacy.io/usage/models#download
