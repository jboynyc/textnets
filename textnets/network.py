# -*- coding: utf-8 -*-

"""Implements the features relating to networks."""

from __future__ import annotations

import os
import json
import random
import sqlite3
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Union
from typing.io import IO
from warnings import warn

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property  # type: ignore

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import textnets as tn
from scipy import LowLevelCallable
from scipy.integrate import quad
from toolz import memoize

from .corpus import TidyText
from .fca import FormalContext
from .viz import decorate_plot

try:
    from . import _ext  # type: ignore

    integrand = LowLevelCallable.from_cython(_ext, "df_integrand")
except ImportError:

    def integrand(x: float, degree: int) -> float:
        """Fallback version of integrand function for the disparity filter."""
        return (1 - x) ** (degree - 2)

    warn("Could not import compiled extension, backbone extraction will be slow.")


class TextnetBase:
    """
    Base class for `Textnet` and `ProjectedTextnet`.

    Attributes
    ----------
    graph : `igraph.Graph`
        Direct access to the igraph object.
    """

    def __init__(self, graph: ig.Graph) -> None:
        self.graph = graph

    @property
    def summary(self) -> str:
        """Summary of underlying graph."""
        return self.graph.summary()

    @property
    def nodes(self) -> ig.VertexSeq:
        """Iterator of nodes."""
        return self.graph.vs

    @property
    def edges(self) -> ig.EdgeSeq:
        """Iterator of edges."""
        return self.graph.es

    def vcount(self) -> int:
        """Returns the number of vertices (nodes)."""
        return self.graph.vcount()

    def ecount(self) -> int:
        """Returns the number of edges."""
        return self.graph.ecount()

    def save_graph(
        self, target: Union[str, bytes, IO], format: Optional[str] = None
    ) -> None:
        """
        Save the underlying graph.

        Parameters
        ----------
        target : str or file
            File or path that the graph should be written to.
        format : {"dot", "edgelist", "gml", "graphml", "pajek", ...}, optional
            Optionally specify the desired format (otherwise it is derived from
            the file suffix).
        """
        if not format:
            if hasattr(target, "name"):
                fname = target.name  # type: ignore
            else:
                fname = str(target)
            format = fname.split(os.path.extsep)[-1]
        self.graph.write(target, format)

    @cached_property
    def degree(self) -> pd.Series:
        """Unweighted node degree."""
        return pd.Series(self.graph.degree(), index=self.nodes["id"])

    @cached_property
    def strength(self) -> pd.Series:
        """Weighted node degree."""
        return pd.Series(self.graph.strength(weights="weight"), index=self.nodes["id"])

    @cached_property
    def betweenness(self) -> pd.Series:
        """Weighted betweenness centrality."""
        return pd.Series(self.graph.betweenness(weights="cost"), index=self.nodes["id"])

    @cached_property
    def closeness(self) -> pd.Series:
        """Weighted closeness centrality."""
        return pd.Series(self.graph.closeness(weights="cost"), index=self.nodes["id"])

    @cached_property
    def eigenvector_centrality(self) -> pd.Series:
        """Weighted eigenvector centrality."""
        return pd.Series(
            self.graph.eigenvector_centrality(weights="weight"), index=self.nodes["id"]
        )

    @cached_property
    def node_types(self) -> List[bool]:
        """Returns boolean list to distinguish node types."""
        return [t == "term" for t in self.nodes["type"]]

    @property
    def clusters(self) -> ig.VertexClustering:
        """Return partition of graph detected by the Leiden algorithm."""
        return self._partition_graph(
            resolution=tn.params["resolution_parameter"],
            seed=tn.params["seed"],
        )

    @property
    def modularity(self) -> float:
        """Returns graph modularity based on the partition detected by the
        Leiden algorithm."""
        return self.graph.modularity(self.clusters, weights="weight")

    def top_degree(self, n: int = 10) -> pd.Series:
        """
        Show nodes sorted by unweighted degree.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10).

        Returns
        -------
        `pandas.Series`
            Ranked nodes.
        """
        return self.degree.sort_values(ascending=False).head(n)

    def top_strength(self, n: int = 10) -> pd.Series:
        """
        Show nodes sorted by weighted degree.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10).

        Returns
        -------
        `pandas.Series`
            Ranked nodes.
        """
        return self.strength.sort_values(ascending=False).head(n)

    def top_betweenness(self, n: int = 10) -> pd.Series:
        """
        Show nodes sorted by betweenness.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10)

        Returns
        -------
        `pandas.Series`
            Ranked nodes.
        """
        return self.betweenness.sort_values(ascending=False).head(n)

    def top_closeness(self, n: int = 10) -> pd.Series:
        """
        Show nodes sorted by closeness.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10)

        Returns
        -------
        `pandas.Series`
            Ranked nodes.
        """
        return self.closeness.sort_values(ascending=False).head(n)

    def top_ev(self, n: int = 10) -> pd.Series:
        """
        Show nodes sorted by eigenvector centrality.

        Parameters
        ----------
        n : int, optional
            How many nodes to show (default: 10)

        Returns
        -------
        `pandas.Series`
            Ranked nodes.
        """
        return self.eigenvector_centrality.sort_values(ascending=False).head(n)

    def top_cluster_nodes(
        self, n: int = 10, part: Optional[ig.VertexClustering] = None
    ) -> pd.Series:
        """
        Show top nodes ranked by weighted degree per cluster.

        Parameters
        ----------
        n : int, optional
            How many nodes to show per cluster (default: 10)
        part : igraph.VertexClustering, optional
            Partition to use (default: Leiden partition).

        Returns
        -------
        `pandas.Series`
            Ranked nodes.
        """
        if part is None:
            part = self.clusters
        return (
            pd.DataFrame(
                {
                    "nodes": self.nodes["id"],
                    "strength": self.strength,
                    "cluster": part.membership,
                }
            )
            .sort_values("strength", ascending=False)
            .groupby("cluster")
            .agg({"nodes": lambda x: ", ".join(x[:n])})["nodes"]
        )

    @decorate_plot
    def _plot(
        self,
        *,
        show_clusters: Union[bool, ig.VertexClustering] = False,
        color_clusters: Union[bool, ig.VertexClustering] = False,
        node_opacity: Optional[float] = None,
        edge_opacity: Optional[float] = None,
        label_nodes: bool = False,
        label_edges: bool = False,
        scale_nodes_by: Optional[str] = None,
        node_label_filter: Optional[Callable[[ig.Vertex], bool]] = None,
        edge_label_filter: Optional[Callable[[ig.Edge], bool]] = None,
        **kwargs,
    ) -> ig.Plot:
        random.seed(tn.params["seed"])
        return ig.plot(self.graph, **kwargs)

    @memoize
    def _partition_graph(self, resolution: float, seed: int) -> ig.VertexClustering:
        if self.graph.is_bipartite():
            part, part0, part1 = la.CPMVertexPartition.Bipartite(
                self.graph, resolution_parameter_01=resolution, weights="weight"
            )
            opt = la.Optimiser()
            opt.set_rng_seed(seed)
            opt.optimise_partition_multiplex(
                [part, part0, part1], layer_weights=[1, -1, -1], n_iterations=-1
            )
        else:
            part = la.find_partition(
                self.graph,
                la.CPMVertexPartition,
                resolution_parameter=resolution,
                seed=seed,
                weights="weight",
            )
        return part

    def __repr__(self) -> str:
        type_counts: Counter = Counter(self.nodes["type"])
        return (
            f"""<{self.__class__.__name__} with {type_counts["doc"]} documents, """
            + f"""{type_counts["term"]} terms, and {self.ecount()} edges>"""
        )

    def _repr_html_(self) -> str:
        type_counts: Counter = Counter(self.nodes["type"])
        return f"""
            <style scoped>
              .full-width {{ width: 100%; }}
            </style>
            <table class="full-width">
              <tr style="font-weight: 600;">
                <td style="text-align: left;">
                  <kbd>{self.__class__.__name__}</kbd>
                </td>
                <td style="color: dodgerblue;">
                  <svg width="1ex" height="1ex">
                    <rect width="1ex" height="1ex" fill="dodgerblue">
                  </svg>
                  Docs: {type_counts["doc"]}
                </td>
                <td style="color: orangered;">
                  <svg width="1ex" height="1ex">
                    <circle cx="50%" cy="50%" r="50%" fill="orangered">
                  </svg>
                  Terms: {type_counts["term"]}
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
            </table>"""


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
    data: DataFrame
        * DataFrame of tokens with per-document counts, as created by
          `Corpus.tokenized` `Corpus.ngrams`, and `Corpus.noun_phrases`.
        * An incidence matrix relating documents to terms.
    sublinear : bool, optional
        Apply sublinear scaling to *tf-idf* values (default: True).
    min_docs : int, optional
        Minimum number of documents a term must appear in to be included
        in the network (default: 2).
    connected : bool, optional
        Keep only the largest connected component of the network (default:
        False).
    doc_attrs : dict of dict, optional
        Additional attributes of document nodes.

    Attributes
    ----------
    im : `IncidenceMatrix`
        Incidence matrix of the bipartite graph.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sublinear: bool = True,
        min_docs: int = 2,
        connected: bool = False,
        doc_attrs: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self._connected = connected
        self._doc_attrs = doc_attrs
        if data.empty:
            raise ValueError("Data is empty.")
        if isinstance(data, IncidenceMatrix):
            self.im = data
        elif isinstance(data, TidyText) or isinstance(data, pd.DataFrame):
            self.im = _im_from_tidy_text(data, sublinear, min_docs)

    @cached_property
    def graph(self):
        """Direct access to the underlying igraph object."""
        g = _graph_from_im(self.im)
        if self._doc_attrs:
            for name, attr in self._doc_attrs.items():
                g.vs[name] = [attr.get(doc) for doc in g.vs["id"]]
        if self._connected:
            return _giant_component(g)
        else:
            return g

    def project(
        self, *, node_type: Literal["doc", "term"], connected: Optional[bool] = False
    ) -> ProjectedTextnet:
        """
        Project to one-mode network.

        Parameters
        ----------
        node_type : {"doc", "term"}
            Either ``doc`` or ``term``, depending on desired node type.
        connected : bool, optional
            Keep only the largest connected component of the projected network
            (default: False).

        Returns
        -------
        `ProjectedTextnet`
            A one-mode textnet.
        """
        if node_type not in {"doc", "term"}:
            raise ValueError("No valid node_type specified.")
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
        g.es["cost"] = [
            1 / pow(w, tn.params["tuning_parameter"]) for w in g.es["weight"]
        ]
        if connected:
            g = _giant_component(g)
        return ProjectedTextnet(g)

    def plot(self, **kwargs) -> ig.Plot:
    def save(self, target: os.PathLike) -> None:
        """
        Save a textnet to file.

        Parameters
        ----------
        target : str or path
            File to save the corpus to. If the file exists, it will be
            overwritten.
        """
        conn = sqlite3.connect(Path(target))
        meta = {"connected": self._connected, "doc_attrs": json.dumps(self._doc_attrs)}
        with conn as c, warnings.catch_warnings():
            warnings.simplefilter("ignore")  # catch warning from pandas.to_sql
            self.im.T.to_sql("textnet_im", c, if_exists="replace")
            pd.Series(meta, name="values").to_sql(
                "textnet_meta", c, if_exists="replace", index_label="keys"
            )

    @classmethod
    def load(cls, source: os.PathLike) -> Textnet:
        """
        Load a textnet from file.

        Parameters
        ----------
        source : str or path
            File to read the corpus from. This should be a file created by
            `Textnet.save`.

        Returns
        -------
        `Textnet`
        """
        if not Path(source).exists():
            raise FileNotFoundError(f"File '{source}' does not exist.")
        conn = sqlite3.connect(Path(source))
        with conn as c:
            im = pd.read_sql("SELECT * FROM textnet_im", c, index_col="term")
            meta = pd.read_sql("SELECT * FROM textnet_meta", c, index_col="keys")[
                "values"
            ]
        connected = meta["connected"] == "1"
        doc_attrs = json.loads(meta["doc_attrs"])
        return cls(IncidenceMatrix(im.T), connected=connected, doc_attrs=doc_attrs)

    def plot(
        self,
        *,
        color_clusters: Union[bool, ig.VertexClustering] = False,
        show_clusters: Union[bool, ig.VertexClustering] = False,
        bipartite_layout: bool = False,
        sugiyama_layout: bool = False,
        circular_layout: bool = False,
        kamada_kawai_layout: bool = False,
        drl_layout: bool = False,
        node_opacity: Optional[float] = None,
        edge_opacity: Optional[float] = None,
        label_term_nodes: bool = False,
        label_doc_nodes: bool = False,
        label_nodes: bool = False,
        label_edges: bool = False,
        node_label_filter: Optional[Callable[[ig.Vertex], bool]] = None,
        edge_label_filter: Optional[Callable[[ig.Edge], bool]] = None,
        scale_nodes_by: Optional[str] = None,
        **kwargs,
    ) -> ig.Plot:
        """
        Plot the bipartite graph.

        Parameters
        ----------
        color_clusters : bool or VertexClustering, optional
            Color nodes according to clusters detected by the Leiden algorithm
            (default: False). Alternately a clustering object generated by
            another community detection algorithm can be passed.
        show_clusters : bool or VertexClustering, optional
            Mark clusters detected by the Leiden algorithm (default: False).
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
        kamada_kawai_layout : bool, optional
            Use a layout created by the Kamada-Kawai algorithm (default: False;
            a weighted Fruchterman-Reingold layout is used unless another
            layout is specified).
        drl_layout : bool, optional
            Use the DrL layout, suitable for large networks (default: False;
            a weighted Fruchterman-Reingold layout is used unless another
            layout is specified).
        node_opacity : float, optional
            Opacity (between 0 and 1) to apply to nodes (default: no
            transparency).
        edge_opacity : float, optional
            Opacity (between 0 and 1) to apply to edges (default: no
            transparency).
        label_term_nodes : bool, optional
            Label term nodes (default: False).
        label_doc_nodes : bool, optional
            Label document nodes (default: False).
        label_nodes : bool, optional
            Label term and document nodes (default: False).
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

        Other Parameters
        ----------------
        target : str or file, optional
            File or path that the plot should be saved to (e.g., ``plot.png``).
        kwargs
            Additional arguments to pass to `igraph.drawing.plot`.

        Returns
        -------
        `igraph.drawing.Plot`
            The plot can be directly displayed in a Jupyter notebook or saved
            as an image file.
        """
        return self._plot(**kwargs)


class ProjectedTextnet(TextnetBase):
    """
    One-mode projection of a textnet.

    Created by calling `Textnet.project()` with the desired ``node_type``.

    Attributes
    ----------
    graph : `igraph.Graph`
        Direct access to the igraph object.
    """

    def alpha_cut(self, alpha: float) -> ProjectedTextnet:
        """
        Return graph backbone.

        Parameters
        ----------
        alpha : float
            Threshold for edge elimination. Must be between 0 and 1. Edges with
            an alpha value above the specified threshold are removed.

        Returns
        -------
        `ProjectedTextnet`
            New textnet sans pruned edges.

        Notes
        -----
        Provided the package was installed properly, a compiled extension will
        be used for a significant speedup.

        References
        ----------
        :cite:`Serrano2009`
        """
        if "alpha" not in self.graph.vertex_attributes():
            self.graph.es["alpha"] = list(_disparity_filter(self.graph))
        pruned = self.graph.copy()
        pruned.delete_edges(pruned.es.select(alpha_ge=alpha))
        return ProjectedTextnet(_giant_component(pruned))

    def plot(self, *, alpha: Optional[float] = None, **kwargs) -> ig.Plot:
        """
        Plot the projected graph.

        Parameters
        ----------
        alpha : float, optional
            Threshold for edge elimination. Must be between 0 and 1. Edges with
            an alpha value above the specified threshold are removed. This is
            useful when plotting "hairball" graphs.

        Returns
        -------
        `igraph.drawing.Plot`
            The plot can be directly displayed in a Jupyter notebook or saved
            as an image file.

        Other Parameters
        ----------------
        target : str or file, optional
            File or path that the plot should be saved to (e.g., ``plot.png``).
        kwargs
            Additional arguments to pass to `igraph.drawing.plot`.
        """
        if alpha is not None:
            to_plot = self.alpha_cut(alpha)
        else:
            to_plot = self
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
        integral_s = quad(integrand, 0, norm_weight_s, args=(degree_s))
        degree_t = target.degree()
        sum_weights_t = target.strength(weights="weight")
        norm_weight_t = edge["weight"] / sum_weights_t
        if degree_t <= 1:
            yield 0
        else:
            integral_t = quad(integrand, 0, norm_weight_t, args=(degree_t))
        yield min(
            1 - (degree_s - 1) * integral_s[0], 1 - (degree_t - 1) * integral_t[0]
        )


def _giant_component(g: ig.Graph) -> ig.Graph:
    """Return the subgraph corresponding to the giant component."""
    size = max(g.components().sizes())
    pos = g.components().sizes().index(size)
    return g.subgraph(g.components()[pos])


def _im_from_tidy_text(
    tidy_text: TidyText, sublinear: bool, min_docs: int
) -> pd.DataFrame:
    df = _tf_idf(tidy_text, sublinear, min_docs)
    im = df.pivot(values="tf_idf", columns="term").fillna(0)
    return IncidenceMatrix(im)


def _graph_from_im(im: pd.DataFrame) -> ig.Graph:
    g = ig.Graph.Incidence(im.to_numpy().tolist(), directed=False)
    g.vs["id"] = np.append(im.index, im.columns).tolist()
    g.es["weight"] = im.to_numpy().flatten()[np.flatnonzero(im)]
    g.es["cost"] = [1 / pow(w, tn.params["tuning_parameter"]) for w in g.es["weight"]]
    g.vs["type"] = ["term" if t else "doc" for t in g.vs["type"]]
    return g


class IncidenceMatrix(pd.DataFrame):
    """Matrix relating documents to terms."""
