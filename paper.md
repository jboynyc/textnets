---
title: 'textnets: A Python package for text analysis using networks'
tags:
  - Python
  - sociology
  - text analysis
  - network analysis
  - visualization
authors:
  - name: John D. Boy
    orcid: 0000-0003-2118-4702
    affiliation: 1
affiliations:
 - name: Assistant Professor, Faculty of Social Sciences, Leiden University
   index: 1
date: 14 September 2020
bibliography: paper.bib
---

# Background

Social scientists increasingly rely on computational tools to make sense of
vasts amounts of unstructured data generated in the wake of to the
ever-expanding digitization of social life. Electronic text, in particular, is
a growing area of interest thanks to the social and cultural insights lurking
in social media posts, digitized corpora, and web content, among other troves
[@Evans2016; @Ignatow2015].

This package aims to fill that need. `textnets` represents collections of texts
as networks of documents and words, which provides powerful possibilities for
the visualization and analysis of texts.

![Network of U.S. Senators and words used in their official statements
following the acquittal vote in the Senate impeachment trial in February
2020\. [Source: @Boy2020]\label{fig:senate}](impeachment-statements.png)

The package can operate on the bipartite network containing both document and
word nodes. \autoref{fig:senate} shows an example of a visualization
created by `textnets`. The underlying corpus is a collection of statements by
U.S. Senators following the conclusion of the impeachment trial against the
president in February 2020. Documents appear as triangles (representing the
Senators who issued the statements), and words appear as yellow squares.

`textnets` can also project one-mode networks containing only document or word
nodes, and it comprises tools to analyze them. For instance, it can visualize a
backbone graph with nodes scaled by various centrality measures. For networks
with a clear community structure, it can also output lists of nodes grouped by
cluster as identified by a community detection algorithm. This can help
identify latent themes in the texts [@Gerlach2018].

Another implementation of the textnets technique exists in the `R` programming
language by its originator [@Bail2016]. It can be found at
<https://github.com/cbail/textnets>. Feature-wise, the two implementations are
roughly on par. This implementation in Python features a modular design, which
is meant to improve ergonomics for users and potential contributors alike. This
package aims to make text analysis techniques accessible to a broader range of
researchers and students. Particularly for use in the classroom, `textnets`
aims at seamless integration with the Jupyter ecosystem [@Kluyver2016].

`textnets` is well documented; its API reference, contribution guidelines, and
a comprehensive tutorial can be found at <https://textnets.readthedocs.io>. For
easy installation, the package is included in the Python Package Index, where
it lives at <https://pypi.org/project/textnets/>. Its code repository and issue
tracker are currently hosted on GitHub at
<https://github.com/jboynyc/textnets>. A test suite is run using Travis, a
continuous integration service, before new releases are published to avoid
regressions from one version to another. Archived versions of releases are
available at doi:10.5281/zenodo.3866676.

# Statement of Need

With `textnets` it is possible to visualize and analyze textual data in novel
ways. These are some of the package's distinguishing features:

- Existing text analysis packages, such as @Benoit2018, typically to visualize
  texts as word clouds, not as network graphs. Unlike word clouds, network
  graphs can visualize not just the frequency and co-occurrence of text
  features, but also their linking role within corpora.
- The discovery of topics is typically performed using latent Dirichlet
  allocation (LDA), while `textnets` uses community detection on the term graph
  for that purpose. Unlike topic modeling using LDA, this does not require
  specifying a fixed number of topics.
- `textnets` can also cluster documents using community detection on the
  document graph. This can serve as an alternative to techniques like $k$-means
  clustering.

# Dependencies

For most heavy lifting, `textnets` uses data structures and methods from
`igraph` [@Csardi2006], `numpy`, and `pandas` [@McKinney2013]. It leverages
`spacy` for natural language processing [@Honnibal2017]. For community
detection, it relies on the Leiden algorithm in its implementation by
@Traag2020. It also depends on `scipy` [@Virtanen2020] to implement the
backbone extraction algorithm.

# References
