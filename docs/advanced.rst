.. highlight:: python

===============
Advanced Topics
===============

Saving and loading your project
-------------------------------

In this example, we define a ``project_file`` to store the configuration
parameters, corpus, and textnet. If the file exists, they are loaded from file;
else they are created and saved to file.

.. code:: python

   from pathlib import Path
   import textnets as tn


   working_dir = Path(".")
   project_file = working_dir / "my_project.db"

   if project_file.exists():
       tn.params.load(project_file)
       corpus = tn.load_corpus(project_file)
       net = tn.load_textnet(project_file)
   else:
       my_params = {"seed": 42, "autodownload": True}
       tn.params.update(my_params)
       corpus = tn.Corpus(tn.examples.digitalisierung, lang="de")
       net = tn.Textnet(corpus.noun_phrases(normalize=True))
       tn.params.save(project_file)
       corpus.save(project_file)
       net.save(project_file)

This code would only require the corpus and textnet to be created once.
Subsequent runs of the script could skip ahead to visualization or analysis.
This saves time, but also helps ensure the reproducibility of results.

Using alternate community detection algorithms
----------------------------------------------

By default, **textnets** will use the Leiden algorithm to find communities in
bipartite and projected networks. You can, however, also use other algorithms.

(These examples assume that you have already created a bipartite `Textnet`
called ``net``.)

Implemented in igraph
~~~~~~~~~~~~~~~~~~~~~

When plotting a textnet, you can supply the arguments ``show_clusters`` or
``color_clusters``. These accept a boolean value, but you can also pass a
`VertexClustering <igraph.clustering.VertexClustering>`, which is the data
structure used by ``igraph``.

If you want to use Blondel et al.'s multilevel algorithm to color the nodes of
a projected textnet, you can do so as follows:

.. code:: python

   terms = net.project(node_type="term")

   # initialize the random seed before running community detection
   tn.init_seed()
   part = terms.graph.community_multilevel(weights="weight")

   print("Modularity: ", terms.graph.modularity(part, weights="weight"))

   terms.plot(label_nodes=True, color_clusters=part)

Alternately, we can also overwrite the textnet's ``clusters`` property::

   terms.clusters = part

To return to the default (clusters detected by the Leiden algorithm), delete
the clusters property::

   del terms.clusters

Implemented in leidenalg
~~~~~~~~~~~~~~~~~~~~~~~~

The ``leidenalg`` package is installed as a dependency of **textnets**. It can
produce a variety of different partition types, and in some cases, you may want
to use a different one than the default. In this example, ``leidenalg`` is
instructed to use the method of "asymptotic surprise" to determine the graph
partition.

.. code:: python

   import leidenalg as la

   terms.clusters = la.find_partition(terms.graph,
                                      partition_type=la.SurpriseVertexPartition, 
                                      weights="weight",
                                      n_iterations=-1,
                                      seed=tn.params["seed"])

After setting the clusters like this, you can plot the network as before. You
can also output a list of nodes per partition::

   terms.top_cluster_nodes()

Implemented in cdlib
~~~~~~~~~~~~~~~~~~~~

The Community Discovery Library (`cdlib <https://cdlib.readthedocs.io/>`__)
implements a wide range of algorithms for community detection that aren't
available in ``igraph``. Some of them are also able to perform community
detection on the bipartite network.

In order to run this example, you first have to install ``cdlib``.

.. code:: python

   from cdlib.algorithms import infomap_bipartite, paris

The first example applies the Infomap community detection algorithm to the
bipartite network::

   # initialize the random seed before running community detection
   tn.init_seed()
   bi_node_community_map = infomap_bipartite(net.graph.to_networkx()).to_node_community_map()

   # overwrite clusters detected by Leiden algorithm
   net.clusters = bi_node_community_map
   print("Modularity: ", net.modularity)

   net.plot(label_nodes=True, show_clusters=True)

This example applies the Paris hierarchical clustering algorithm to the
projected network::

   docs = net.project(node_type="doc")

   # initialize the random seed before running community detection
   tn.init_seed()
   docs_node_community_map = paris(docs.graph.to_networkx()).to_node_community_map()

   # overwrite clusters detected by Leiden algorithm
   docs.clusters = docs_node_community_map
   print("Modularity: ", docs.modularity)

   docs.plot(label_nodes=True, color_clusters=True)

Implemented in karateclub
~~~~~~~~~~~~~~~~~~~~~~~~~

`Karate Club <https://karateclub.readthedocs.io/>`__ is a library of
machine-learning methods to apply to networks. Among other things, it also
implements community detection algorithms. Here's an example for using
community detection from ``karateclub`` with **textnets**.

This example requires you to first have installed ``karateclub``.

.. code:: python

   from karateclub import SCD

   cd = SCD(seed=tn.params["seed"])
   cd.fit(net.graph.to_networkx())

   net.clusters = list(cd.get_memberships().values())
   print("Modularity: ", net.modularity)

   np.plot(color_clusters=True, label_nodes=True)

Additional measures for centrality analysis
-------------------------------------------

The :doc:`tutorial` provides examples of using BiRank, betweenness, closeness
and (weighted and unweighted) degree to analyze a textnet. The `NetworkX
<https://networkx.org>`__ library implements a large variety of other
centrality measures that may also prove helpful that aren't available in
``igraph``, the library that ``textnets`` builds on, including additional
centrality measures for bipartite networks.

This example requires ``networkx`` to be installed.

.. code:: python

   import networkx as nx

   bi_btwn = nx.algorithms.bipartite.betweenness_centrality(net.graph.to_networkx())
   net.nodes["btwn"] = list(bi_btwn.values())
   docs.plot(scale_nodes_by="btwn")

   katz_centrality = nx.katz_centrality(docs.graph.to_networkx(), weight="weight")
   docs.nodes["katz"] = list(katz_centrality.values())
   docs.plot(scale_nodes_by="katz")

Alternative methods of term extraction and weighing
---------------------------------------------------

By default, **textnets** leverages spaCy language models to break up your
corpus when you call `noun_phrases`, `ngrams` or `tokenized`, and it uses
*tf-idf* term weights. There are many alternative ways of extracting terms and
weighing them, and by defining a custom function, you can use them with
**textnets**.

This example uses `YAKE! <http://yake.inesctec.pt/>`__, the popular library for
keyword extraction, to extract keywords from a corpus and weighs them according
to their significance.

This example requires ``yake`` to be installed.

.. code:: python

   import textnets as tn
   from yake import KeywordExtractor

   def yake(
      corpus: tn.Corpus,
      lang: str="en",
      ngram_size: int=3,
      top: int=50,
      window: int=2
   ) -> tn.corpus.TidyText:
      """Use YAKE keyword extraction to break up corpus."""
      kw = KeywordExtractor(
               lan=lang,
               n=ngram_size,
               top=top,
               windowsSize=window
           )
      tt = []
      for label, doc in corpus.documents.items():
          for term, sig in kw.extract_keywords(doc):
              tt.append({"label": label, "term": term, "term_weight": 1-sig, "n": 1})
      return tn.corpus.TidyText(tt).set_index("label")

The result of calling ``yake`` on an instance of ``Corpus`` can be passed to
``Textnet``.
