========
Textnets
========

Automated text analysis with networks.

This is a Python implementation of `Chris Bail's textnets package for R`_.  It
is free software under the terms of the GNU General Public License v3.

The underlying idea behind textnets is presented in this paper:

Christopher A. Bail, "`Combining natural language processing and network
analysis to examine how advocacy organizations stimulate conversation on social
media`__," *Proceedings of the National Academy of Sciences of the United States
of America* 113, no. 42 (2016), 11823–11828, doi:10.1073/pnas.1607151113.

__ https://doi.org/10.1073/pnas.1607151113

Features
--------

::

    from textnets import TextCorpus, Textnets


    c = TextCorpus('~/nltk_data/corpora/state_union/*.txt')
    noun_phrases = c.noun_phrases()
    tn = Textnets(noun_phrases)
    g_groups = tn.graph(node_type='groups')
    g_words = tn.graph(node_type='words')

The library builds on the state-of-the-art library `spacy`_ for
natural-language processing and `igraph`_ for network analysis. In addition to
providing a Python library for the analysis of text corpora, textnets can also
be used as a command-line tool to generate network graphs from text corpora.

::

    $ textnets --lex noun_phrases --node-type groups ~/nltk_data/corpora/state_union | gzip > sotu_groups.graphmlz

Run ``textnets --help`` for usage instructions.

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _`Chris Bail's textnets package for R`: https://github.com/cbail/textnets/
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`igraph`: http://igraph.org/python/
.. _`spacy`: http://spacy.io/
