# -*- coding: utf-8 -*-

"""Implements the features relating to networks."""

from typing import Dict, Optional, List, Literal, Union
from functools import cached_property

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la


#: Tuning parameter (alpha) for inverse edge weights
#: (Opsahl et al. 2010, 10.1016/j.socnet.2010.03.006)
TUNING_PARAMETER = 0.5

#: Resolution parameter (gamma) for community detection
#: (Reichardt & Bornnholdt 2006, 10.1103/PhysRevE.74.016110;
#: Traag et al. 2019, 10.1038/s41598-019-41695-z)
RESOLUTION_PARAMETER = 0.5

#: Membership degree threshold (alpha) for concept lattice
#: (Tho et al. 2006, 10.1109/TKDE.2006.87)
FFCA_CUTOFF = 0.3


class Textnet:
    """
    Textnet for the relational analysis of meanings.

    A textnet is a bipartite network of documents and terms. Links exist
    only between the two types of nodes. Documents have a tie with terms
    they contain; the tie is weighted by tf-idf.

    The bipartite network can be projected into two different kinds of
    single-mode networks: document-to-document, and term-to-term.

    Parameters
    ----------
    tidy_text : DataFrame
        DataFrame of tokens with per-document counts, as created by
        `Corpus` methods `tokenized()` and `noun_phrases()`.
    sublinear : bool, optional
        Apply sublinear scaling to tf-idf values (default: True).
    doc_attrs : dict of dict, optional
        Additional attributes of document nodes.
    min_docs : int, optional
        Minimum number of documents a term must appear in to be included
        in the network (default: 2).
    """

    def __init__(
        self,
        tidy_text: pd.DataFrame,
        sublinear: bool = True,
        doc_attrs: Optional[Dict[str, Dict[str, str]]] = None,
        min_docs: int = 2,
    ):
        df = self._tf_idf(tidy_text, sublinear, min_docs)
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

    @cached_property
    def node_types(self) -> List[bool]:
        """Returns boolean list to distinguish node types."""
        return [True if t == "term" else False for t in self.graph.vs["type"]]

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
        graph = self.graph.bipartite_projection(
            types=self.node_types, which=graph_to_return
        )
        for i in graph.es.indices:
            edge = graph.es[i]
            source, target = edge.source_vertex["id"], edge.target_vertex["id"]
            if source == target:
                edge["weight"] = 0
            else:
                edge["weight"] = weights.loc[source, target]
        return graph

    def plot(
        self,
        show_clusters: bool = False,
        bipartite_layout: bool = False,
        label_term_nodes: bool = False,
        label_doc_nodes: bool = False,
        label_edges: bool = False,
        **kwargs,
    ):
        """Plot the bipartite graph.

        Parameters
        ----------
        show_clusters: bool, optional
            Mark clusters detected by Leiden algorithm (default: False).
        bipartite_layout : bool, optional
            Use a bipartite graph layout (default: False, in which case a
            weighted Fruchterman-Reingold layout is used).
        label_term_nodes : bool, optional
            Label term nodes (default: False)
        label_doc_nodes : bool, optional
            Label document nodes (default: False)
        label_edges : bool, optional
            Show edge weights in plot.

        Returns
        -------
        ig.drawing.Plot
            The plot can be directly displayed in a Jupyter network or saved as
            an image file.
        """
        if bipartite_layout:
            layout = self.graph.layout_bipartite(types=self.node_types)
        else:
            layout = self.graph.layout_fruchterman_reingold(
                weights="weight", grid=False
            )
        kwargs.setdefault("layout", layout)
        kwargs.setdefault("autocurve", True)
        kwargs.setdefault("margin", 50)
        kwargs.setdefault("edge_color", "lightgray")
        kwargs.setdefault(
            "vertex_shape", ["circle" if v else "square" for v in self.node_types]
        )
        kwargs.setdefault(
            "vertex_color",
            ["orangered" if v else "dodgerblue" for v in self.node_types],
        )
        kwargs.setdefault(
            "vertex_frame_color", ["black" if v else "white" for v in self.node_types]
        )
        kwargs.setdefault("vertex_frame_width", 0.2)
        kwargs.setdefault(
            "vertex_label",
            [
                v["id"]
                if (v["type"] == "doc" and label_doc_nodes)
                or (v["type"] == "term" and label_term_nodes)
                else None
                for v in self.graph.vs
            ],
        )
        kwargs.setdefault("vertex_label_size", 10)
        kwargs.setdefault(
            "edge_label",
            [f"{e['weight']:.2f}" if label_edges else None for e in self.graph.es],
        )
        kwargs.setdefault("edge_label_size", 8)
        kwargs.setdefault("mark_groups", self.clusters if show_clusters else False)
        return ig.plot(self.graph, **kwargs)

    @cached_property
    def clusters(self):
        """Return partition of bipartite graph detected by Leiden algorithm."""
        return self._partition_graph(self.graph, resolution=RESOLUTION_PARAMETER)

    @cached_property
    def context(self):
        """Return formal context of terms and documents."""
        return self._formal_context(self.im, alpha=FFCA_CUTOFF)

    @staticmethod
    def _partition_graph(graph, resolution):
        part, part0, part1 = la.CPMVertexPartition.Bipartite(
            graph, resolution_parameter_01=resolution
        )
        opt = la.Optimiser()
        opt.optimise_partition_multiplex(
            [part, part0, part1], layer_weights=[1, -1, -1], n_iterations=100
        )
        return part

    @staticmethod
    def _formal_context(im, alpha):
        # The incidence matrix is a "fuzzy formal context." We can binarize it
        # by using a cutoff. This is known as an alpha-cut.
        crisp = im.applymap(lambda x: True if x >= alpha else False)
        reduced = crisp[crisp.any(axis=1)].loc[:, crisp.any(axis=0)]
        objects = reduced.index.tolist()
        properties = reduced.columns.tolist()
        bools = reduced.to_numpy()
        return objects, properties, bools

    @staticmethod
    def _tf_idf(tidy_text: pd.DataFrame, sublinear: bool, min_docs: int):
        """Calculate term frequency/inverse document frequency."""
        if sublinear:
            tidy_text["tf"] = tidy_text["n"].map(_sublinear_scaling)
        else:
            totals = (
                tidy_text.groupby(tidy_text.index).sum().rename(columns={"n": "total"})
            )
            tidy_text = tidy_text.merge(totals, right_index=True, left_index=True)
            tidy_text["tf"] = tidy_text["n"] / tidy_text["total"]
        idfs = np.log10(len(set(tidy_text.index)) / tidy_text["term"].value_counts())
        tt = tidy_text.merge(
            pd.DataFrame(idfs), left_on="term", right_index=True
        ).rename(columns={"term_y": "idf"})
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
