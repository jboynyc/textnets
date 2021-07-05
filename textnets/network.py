# -*- coding: utf-8 -*-

"""Implements the features relating to networks."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, List, Union, Iterator, Callable

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property  # type: ignore

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
from scipy.integrate import quad

from .viz import TextnetPalette
from .fca import FormalContext


#: Tuning parameter (alpha) for inverse edge weights
#: (see :cite:`Opsahl2010`).
TUNING_PARAMETER = 0.5

#: Resolution parameter (gamma) for community detection
#: (see :cite:`Reichardt2006,Traag2019`).
RESOLUTION_PARAMETER = 0.1


class TextnetBase:
    """Base class for `Textnet` and `ProjectedTextnet`."""

    def __init__(self, graph):
        self.graph = graph

    def summary(self) -> str:
        """Return summary of underlying graph."""
        return self.graph.summary()

    @property
    def vs(self):
        """Iterator of nodes (vertices)."""
        return self.graph.vs

    @property
    def es(self):
        """Iterator of edges."""
        return self.graph.es

    def vcount(self):
        """Returns the number of vertices (nodes)."""
        return self.graph.vcount()

    def ecount(self):
        """Returns the number of edges."""
        return self.graph.ecount()

    @cached_property
    def degree(self) -> pd.Series:
        """Unweighted node degree."""
        return pd.Series(self.graph.degree(), index=self.vs["id"])

    @cached_property
    def strength(self) -> pd.Series:
        """Weighted node degree."""
        return pd.Series(self.graph.strength(weights="weight"), index=self.vs["id"])

    @cached_property
    def betweenness(self) -> pd.Series:
        """Weighted betweenness centrality."""
        return pd.Series(self.graph.betweenness(weights="cost"), index=self.vs["id"])

    @cached_property
    def closeness(self) -> pd.Series:
        """Weighted closeness centrality."""
        return pd.Series(self.graph.closeness(weights="cost"), index=self.vs["id"])

    @cached_property
    def eigenvector_centrality(self) -> pd.Series:
        """Weighted eigenvector centrality."""
        return pd.Series(
            self.graph.eigenvector_centrality(weights="weight"), index=self.vs["id"]
        )

    @cached_property
    def node_types(self) -> List[bool]:
        """Returns boolean list to distinguish node types."""
        return [True if t == "term" else False for t in self.vs["type"]]

    @cached_property
    def clusters(self) -> ig.clustering.VertexClustering:
        """Return partition of graph detected by Leiden algorithm."""
        return self._partition_graph(self.graph, resolution=RESOLUTION_PARAMETER)

    @cached_property
    def modularity(self):
        """Returns graph modularity based on partition detected by Leiden algorithm."""
        return self.graph.modularity(self.clusters, weights="weight")

    def top_degree(self, n: int = 10) -> pd.Series:
        """Show nodes sorted by unweighted degree.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10).

        Returns
        -------
        pd.Series
            Ranked nodes.
        """
        return self.degree.sort_values(ascending=False).head(n)

    def top_strength(self, n: int = 10) -> pd.Series:
        """Show nodes sorted by weighted degree.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10).

        Returns
        -------
        pd.Series
            Ranked nodes.
        """
        return self.strength.sort_values(ascending=False).head(n)

    def top_betweenness(self, n: int = 10) -> pd.Series:
        """Show nodes sorted by betweenness.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10)

        Returns
        -------
        pd.Series
            Ranked nodes.
        """
        return self.betweenness.sort_values(ascending=False).head(n)

    def top_closeness(self, n: int = 10) -> pd.Series:
        """Show nodes sorted by closeness.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10)

        Returns
        -------
        pd.Series
            Ranked nodes.
        """
        return self.closeness.sort_values(ascending=False).head(n)

    def top_ev(self, n: int = 10) -> pd.Series:
        """Show nodes sorted by eigenvector centrality.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10)

        Returns
        -------
        pd.Series
            Ranked nodes.
        """
        return self.eigenvector_centrality.sort_values(ascending=False).head(n)

    def top_cluster_nodes(
        self, n: int = 10, part: Optional[ig.clustering.VertexClustering] = None
    ) -> pd.Series:
        """Show top nodes ranked by weighted degree per cluster.

        Parameters
        ----------
        n : int, optional
            How many nodes to show per cluster (default: 10)
        part : VertexClustering, optional
            Partition to use (default: Leiden partition).

        Returns
        -------
        pd.Series
            Ranked nodes.
        """
        if part is None:
            part = self.clusters
        return (
            pd.DataFrame(
                {
                    "nodes": self.vs["id"],
                    "strength": self.strength,
                    "cluster": part.membership,
                }
            )
            .sort_values("strength", ascending=False)
            .groupby("cluster")
            .agg({"nodes": lambda x: ", ".join(x[:n])})["nodes"]
        )

    def _plot(
        self,
        show_clusters: Union[bool, ig.clustering.VertexClustering] = False,
        color_clusters: Union[bool, ig.clustering.VertexClustering] = False,
        label_edges: bool = False,
        scale_nodes_by: Optional[str] = None,
        node_label_filter: Optional[Callable[[ig.Vertex], bool]] = None,
        edge_label_filter: Optional[Callable[[ig.Edge], bool]] = None,
        **kwargs,
    ) -> ig.drawing.Plot:
        if "layout" not in kwargs.keys():
            layout = self.graph.layout_fruchterman_reingold(
                weights="weight", grid=False
            )
            kwargs.setdefault("layout", layout)
        if scale_nodes_by:
            dist = getattr(self, scale_nodes_by)
            if abs(dist.skew()) < 2:
                dist **= 2
            norm = (dist - dist.mean()) / dist.std()
            mult = 20 / abs(norm).max()
            kwargs.setdefault("vertex_size", [25 + mult * z for z in norm])
        if show_clusters:
            if isinstance(show_clusters, ig.clustering.VertexClustering):
                kwargs.setdefault("mark_groups", show_clusters)
            else:
                kwargs.setdefault("mark_groups", self.clusters)
        if color_clusters:
            if isinstance(color_clusters, ig.clustering.VertexClustering):
                kwargs.setdefault(
                    "vertex_color",
                    [
                        TextnetPalette(color_clusters.n)[c]
                        for c in color_clusters.membership
                    ],
                )
            else:
                kwargs.setdefault(
                    "vertex_color",
                    [
                        TextnetPalette(self.clusters.n)[c]
                        for c in self.clusters.membership
                    ],
                )
        else:
            kwargs.setdefault(
                "vertex_color",
                ["orangered" if v else "dodgerblue" for v in self.node_types],
            )
        kwargs.setdefault(
            "vertex_shape", ["circle" if v else "square" for v in self.node_types]
        )
        kwargs.setdefault(
            "vertex_frame_color", ["black" if v else "white" for v in self.node_types]
        )
        kwargs.setdefault(
            "edge_label",
            [f"{e['weight']:.2f}" if label_edges else None for e in self.es],
        )
        kwargs.setdefault("autocurve", True)
        kwargs.setdefault("wrap_labels", True)
        kwargs.setdefault("margin", 50)
        kwargs.setdefault("edge_color", "lightgray")
        kwargs.setdefault("vertex_frame_width", 0.2)
        kwargs.setdefault("vertex_label_size", 10)
        kwargs.setdefault("edge_label_size", 8)
        if node_label_filter and "vertex_label" in kwargs:
            node_labels = kwargs.pop("vertex_label")
            filtered_node_labels = map(node_label_filter, self.vs)
            kwargs["vertex_label"] = [
                lbl if keep else None
                for lbl, keep in zip(node_labels, filtered_node_labels)
            ]
        if edge_label_filter and "edge_label" in kwargs:
            edge_labels = kwargs.pop("edge_label")
            filtered_edge_labels = map(edge_label_filter, self.es)
            kwargs["edge_label"] = [
                lbl if keep else None
                for lbl, keep in zip(edge_labels, filtered_edge_labels)
            ]
        return ig.plot(self.graph, **kwargs)

    @staticmethod
    def _partition_graph(graph, resolution):
        if graph.is_bipartite():
            part, part0, part1 = la.CPMVertexPartition.Bipartite(
                graph, resolution_parameter_01=resolution, weights="weight"
            )
            opt = la.Optimiser()
            opt.optimise_partition_multiplex(
                [part, part0, part1], layer_weights=[1, -1, -1], n_iterations=-1
            )
        else:
            part = la.find_partition(
                graph,
                la.CPMVertexPartition,
                resolution_parameter=resolution,
                weights="weight",
            )
        return part

    def _repr_html_(self):
        c = Counter(self.vs["type"])
        return f"""
            <style scoped>
              .full-width {{ width: 100%; }}
              summary {{
                cursor: help;
                list-style: none;
              }}
              details[open] summary {{
                margin-bottom: 1em;
              }}
            </style>
            <details>
              <summary>
                <table class="full-width">
                  <tr style="font-weight: 600;">
                    <td style="text-align: left;">
                      <kbd>{self.__class__.__name__}</kbd>
                    </td>
                    <td style="color: dodgerblue;">
                      <svg width="1ex" height="1ex">
                        <rect width="1ex" height="1ex" fill="dodgerblue">
                      </svg>
                      Docs: {c["doc"]}
                    </td>
                    <td style="color: orangered;">
                      <svg width="1ex" height="1ex">
                        <circle cx="50%" cy="50%" r="50%" fill="orangered">
                      </svg>
                      Terms: {c["term"]}
                    </td>
                    <td style="color: darkgray;">
                      <svg width="2ex" height="1ex">
                        <line x1="0" y1="50%" x2="100%" y2="50%"
                          stroke="darkgray"
                          stroke-width="2">
                      </svg>
                      Edges: {self.ecount()}
                    </td>
                  </tr>
                </table>
              </summary>
              <pre>{self.__doc__}</pre>
            </details>"""


class Textnet(TextnetBase, FormalContext):
    """
    Textnet for the relational analysis of meanings.

    A textnet is a bipartite network of documents and terms. Links exist
    only between the two types of nodes. Documents have a tie with terms
    they contain; the tie is weighted by *tf-idf*.

    The bipartite network can be projected into two different kinds of
    single-mode networks: document-to-document, and term-to-term.

    Experimental: The underlying incidence matrix can also be turned into a
    formal context, which can be used to construct a concept lattice.

    Parameters
    ----------
    tidy_text : DataFrame
        DataFrame of tokens with per-document counts, as created by
        `Corpus` methods `tokenized` and `noun_phrases`.
    sublinear : bool, optional
        Apply sublinear scaling to *tf-idf* values (default: True).
    doc_attrs : dict of dict, optional
        Additional attributes of document nodes.
    min_docs : int, optional
        Minimum number of documents a term must appear in to be included
        in the network (default: 2).

    Attributes
    ----------
    graph : ig.Graph
        Direct access to the igraph object.
    """

    def __init__(
        self,
        tidy_text: pd.DataFrame,
        sublinear: bool = True,
        doc_attrs: Optional[Dict[str, Dict[str, str]]] = None,
        min_docs: int = 2,
    ):
        df = _tf_idf(tidy_text, sublinear, min_docs)
        im = df.pivot(values="tf_idf", columns="term").fillna(0)
        self.im = im
        g = ig.Graph.Incidence(im.to_numpy().tolist(), directed=False)
        g.vs["id"] = np.append(im.index, im.columns).tolist()
        g.es["weight"] = im.to_numpy().flatten()[np.flatnonzero(im)]
        g.es["cost"] = [1 / pow(w, TUNING_PARAMETER) for w in g.es["weight"]]
        g.vs["type"] = ["term" if t else "doc" for t in g.vs["type"]]
        if doc_attrs:
            for name, attr in doc_attrs.items():
                g.vs[name] = [attr.get(doc) for doc in g.vs["id"]]
        self.graph = g

    def project(self, node_type: Literal["doc", "term"]) -> ig.Graph:
        """Project to one-mode network.

        Parameters
        ----------
        node_type : str
            Either ``doc`` or ``term``, depending on desired node type.

        Returns
        -------
        ig.Graph
            The projected graph with edge weights.
        """
        assert node_type in ("doc", "term"), "No valid node_type specified."
        graph_to_return = 0
        if node_type == "term":
            graph_to_return = 1
            weights = self.im.T.dot(self.im)
        else:
            weights = self.im.dot(self.im.T)
        g = self.graph.bipartite_projection(
            types=self.node_types, which=graph_to_return
        )
        for i in g.es.indices:
            edge = g.es[i]
            source, target = edge.source_vertex["id"], edge.target_vertex["id"]
            if source == target:
                edge["weight"] = 0
            else:
                edge["weight"] = weights.loc[source, target]
        g.es["cost"] = [1 / pow(w, TUNING_PARAMETER) for w in g.es["weight"]]
        return ProjectedTextnet(g)

    def plot(
        self,
        bipartite_layout: bool = False,
        sugiyama_layout: bool = False,
        circular_layout: bool = False,
        label_term_nodes: bool = False,
        label_doc_nodes: bool = False,
        **kwargs,
    ) -> ig.drawing.Plot:
        """Plot the bipartite graph.

        Parameters
        ----------
        color_clusters : bool or VertexClustering, optional
            Color nodes according to clusters detected by Leiden algorithm
            (default: False). Alternately a clustering object generated by
            another community detection algorithm can be passed.
        show_clusters : bool or VertexClustering, optional
            Mark clusters detected by Leiden algorithm (default: False).
            Alternately a clustering object generated by another community
            detection algorithm can be passed.
        bipartite_layout : bool, optional
            Use a bipartite graph layout (default: False; a
            weighted Fruchterman-Reingold layout is used unless
            another layout is specified).
        sugiyama_layout : bool, optional
            Use layered Sugiyama layout (default: False; a weighted
            Fruchterman-Reingold layout is used unless another layout is
            specified).
        circular_layout : bool, optional
            Use circular Reingold-Tilford layout (default: False; a weighted
            Fruchterman-Reingold layout is used unless another layout is
            specified).
        label_term_nodes : bool, optional
            Label term nodes (default: False).
        label_doc_nodes : bool, optional
            Label document nodes (default: False).
        label_edges : bool, optional
            Show edge weights in plot.
        node_label_filter : function, optional
            Function returning boolean value mapped to iterator of nodes to
            decide whether or not to suppress labels.
        edge_label_filter : function, optional
            Function returning boolean value mapped to iterator of edges to
            decide whether or not to suppress labels.
        scale_nodes_by : str, optional
            Name of centrality measure to scale nodes by. Possible values:
            ``betweenness``, ``closeness``, ``degree``, ``strength``,
            ``eigenvector_centrality`` (default: None).
        kwargs
            Additional arguments to pass to :doc:`ig.plot <ig:tutorial>`.

        Returns
        -------
        ig.drawing.Plot
            The plot can be directly displayed in a Jupyter notebook or saved
            as an image file.
        """
        if bipartite_layout:
            layout = self.graph.layout_bipartite(types=self.node_types)
            kwargs.setdefault("layout", layout)
        elif sugiyama_layout:
            layout = self.graph.layout_sugiyama(
                weights="weight", hgap=50, maxiter=100000
            )
            kwargs.setdefault("layout", layout)
        elif circular_layout:
            layout = self.graph.layout_reingold_tilford_circular()
            kwargs.setdefault("layout", layout)
        kwargs.setdefault(
            "vertex_label",
            [
                v["id"]
                if (v["type"] == "doc" and label_doc_nodes)
                or (v["type"] == "term" and label_term_nodes)
                else None
                for v in self.vs
            ],
        )
        return self._plot(**kwargs)


class ProjectedTextnet(TextnetBase):
    """One-mode projection of a textnet.

    Created by calling `Textnet.project()` with the desired ``node_type``.

    Attributes
    ----------
    graph : ig.Graph
        Direct access to the igraph object."""

    def alpha_cut(self, alpha: float) -> ProjectedTextnet:
        """Return graph "backbone."

        Parameters
        ----------
        alpha : float
            Threshold for edge elimination. Must be between 0 and 1. Edges with
            an alpha value above the specified threshold are removed.

        Returns
        -------
        ProjectedTextnet
            New textnet sans pruned edges.

        References
        ----------
        :cite:`Serrano2009`
        """
        if "alpha" not in self.graph.vertex_attributes():
            self.graph.es["alpha"] = list(_disparity_filter(self.graph))
        pruned = self.graph.copy()
        pruned.delete_edges(pruned.es.select(alpha_ge=alpha))
        return ProjectedTextnet(_giant_component(pruned))

    def plot(
        self, label_nodes: bool = False, alpha: Optional[float] = None, **kwargs
    ) -> ig.drawing.Plot:
        """Plot the projected graph.

        Parameters
        ----------
        color_clusters : bool or VertexClustering, optional
            Color nodes according to clusters detected by Leiden algorithm
            (default: False). Alternately a clustering object generated by
            another community detection algorithm can be passed.
        show_clusters : bool or VertexClustering, optional
            Mark clusters detected by Leiden algorithm (default: False).
            Alternately a clustering object generated by another community
            detection algorithm can be passed.
        alpha : float, optional
            Threshold for edge elimination. Must be between 0 and 1. Edges with
            an alpha value above the specified threshold are removed. This is
            useful when plotting "hairball" graphs.
        label_nodes : bool, optional
            Label nodes (default: False).
        label_edges : bool, optional
            Show edge weights in plot.
        node_label_filter : function, optional
            Function returning boolean value mapped to iterator of nodes to
            decide whether or not to suppress labels.
        edge_label_filter : function, optional
            Function returning boolean value mapped to iterator of edges to
            decide whether or not to suppress labels.
        scale_nodes_by : str, optional
            Name of centrality measure to scale nodes by. Possible values:
            ``betweenness``, ``closeness``, ``degree``, ``strength``,
            ``eigenvector_centrality`` (default: None).
        kwargs
            Additional arguments to pass to :doc:`ig.plot <ig:tutorial>`.

        Returns
        -------
        ig.drawing.Plot
            The plot can be directly displayed in a Jupyter notebook or saved
            as an image file.

        """
        if alpha is not None:
            to_plot = self.alpha_cut(alpha)
        else:
            to_plot = self
        kwargs.setdefault(
            "vertex_label", [v["id"] for v in to_plot.vs] if label_nodes else None
        )
        return to_plot._plot(**kwargs)


def _tf_idf(tidy_text: pd.DataFrame, sublinear: bool, min_docs: int) -> pd.DataFrame:
    """Calculate term frequency/inverse document frequency."""
    if sublinear:
        tidy_text["tf"] = tidy_text["n"].map(_sublinear_scaling)
    else:
        totals = tidy_text.groupby(tidy_text.index).sum().rename(columns={"n": "total"})
        tidy_text = tidy_text.merge(totals, right_index=True, left_index=True)
        tidy_text["tf"] = tidy_text["n"] / tidy_text["total"]
    idfs = np.log10(len(set(tidy_text.index)) / tidy_text["term"].value_counts())
    tt = tidy_text.merge(pd.DataFrame(idfs), left_on="term", right_index=True).rename(
        columns={"term_y": "idf"}
    )
    tt["tf_idf"] = tt["tf"] * tt["idf"]
    wc = tt.groupby("term").count()["tf"]
    tt = (
        tt.reset_index()
        .merge(wc >= min_docs, on="term", how="left")
        .rename(columns={"tf_y": "keep"})
        .set_index("label")
    )
    return tt[tt["keep"]][["term", "n", "tf_idf"]]


def _sublinear_scaling(n: Union[int, float]) -> float:
    """Logarithmic scaling function."""
    return 1 + np.log10(n) if n > 0 else 0


def _disparity_filter(g: ig.Graph) -> Iterator[float]:
    """Compute significance scores of edge weights."""
    for edge in g.es:
        source, target = edge.vertex_tuple
        degree_s = source.degree()
        sum_weights_s = source.strength(weights="weight")
        norm_weight_s = edge["weight"] / sum_weights_s
        integral_s = quad(lambda x: (1 - x) ** (degree_s - 2), 0, norm_weight_s)
        degree_t = target.degree()
        sum_weights_t = target.strength(weights="weight")
        norm_weight_t = edge["weight"] / sum_weights_t
        try:
            integral_t = quad(lambda x: (1 - x) ** (degree_t - 2), 0, norm_weight_t)
        except ZeroDivisionError:
            yield 0
        yield min(
            1 - (degree_s - 1) * integral_s[0], 1 - (degree_t - 1) * integral_t[0]
        )


def _giant_component(g: ig.Graph) -> ig.Graph:
    """Return the subgraph corresponding to the giant component."""
    size = max(g.components().sizes())
    pos = g.components().sizes().index(size)
    return g.subgraph(g.components()[pos])
