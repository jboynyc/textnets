# -*- coding: utf-8 -*-

"""Extends visualization features."""

from __future__ import annotations

from functools import wraps
from itertools import repeat
from math import ceil
from typing import Callable, Iterator, List

import igraph as ig
import numpy as np
from igraph.drawing.colors import (
    color_name_to_rgba,
    darken,
    lighten,
    PrecalculatedPalette,
)
from pandas import Series

import textnets as tn

#: Base colors for textnets color palette.
BASE_COLORS = [
    "tomato",
    "darkseagreen",
    "slateblue",
    "gold",
    "orchid",
    "springgreen",
    "dodgerblue",
]


class TextnetPalette(PrecalculatedPalette):
    """Color palette for textnets."""

    def __init__(self, n: int):
        base_colors = [color_name_to_rgba(c) for c in BASE_COLORS]

        num_base_colors = len(base_colors)
        colors = base_colors[:]

        blocks_to_add = ceil((n - num_base_colors) / num_base_colors)
        ratio_increment = 1.0 / (ceil(blocks_to_add / 2.0) + 1)

        adding_darker = False
        ratio = ratio_increment
        while len(colors) < n:
            if adding_darker:
                new_block = [darken(color, ratio) for color in base_colors]
            else:
                new_block = [lighten(color, ratio) for color in base_colors]
                ratio += ratio_increment
            colors.extend(new_block)
            adding_darker = not adding_darker

        colors = colors[:n]
        super().__init__(colors)


def decorate_plot(plot_func: Callable) -> Callable:
    """Style the plot produced by igraph's plot function."""

    @wraps(plot_func)
    def wrapper(net: tn.network.TextnetBase, **kwargs) -> ig.Plot:
        graph = net.graph
        # Rewrite node_* arguments as vertex_* arguments
        node_opts = [k for k, _ in kwargs.items() if k.startswith("node_")]
        for opt in node_opts:
            val = kwargs.pop(opt)
            kwargs[opt.replace("node_", "vertex_")] = val
        # Marking and coloring clusters
        show_clusters = kwargs.pop("show_clusters", False)
        color_clusters = kwargs.pop("color_clusters", False)
        if show_clusters:
            if isinstance(show_clusters, ig.VertexClustering):
                markers = zip(
                    _cluster_node_indices(show_clusters),
                    repeat(_add_opacity("limegreen", 0.4)),
                )
            else:
                markers = zip(
                    _cluster_node_indices(net.clusters),
                    repeat(_add_opacity("limegreen", 0.4)),
                )
            kwargs.setdefault("mark_groups", markers)
        if color_clusters:
            if isinstance(color_clusters, ig.VertexClustering):
                kwargs["vertex_color"] = [
                    TextnetPalette(color_clusters._len)[c]
                    for c in color_clusters.membership
                ]
            else:
                kwargs["vertex_color"] = [
                    TextnetPalette(net.clusters._len)[c]
                    for c in net.clusters.membership
                ]
        # Default appearance
        kwargs.setdefault("autocurve", True)
        kwargs.setdefault("edge_color", "lightgray")
        kwargs.setdefault("edge_label_size", 6)
        kwargs.setdefault("edge_width", 2)
        kwargs.setdefault("margin", 50)
        kwargs.setdefault("vertex_frame_width", 0.25)
        kwargs.setdefault("vertex_label_size", 9)
        kwargs.setdefault("vertex_size", 20)
        kwargs.setdefault("wrap_labels", True)
        kwargs.setdefault(
            "layout", graph.layout_fruchterman_reingold(weights="weight", grid=False)
        )
        kwargs.setdefault(
            "vertex_color",
            ["orangered" if v else "dodgerblue" for v in net.node_types],
        )
        kwargs.setdefault(
            "vertex_shape", ["circle" if t else "square" for t in net.node_types]
        )
        kwargs.setdefault(
            "vertex_frame_color",
            ["black" if t else "white" for t in net.node_types],
        )
        # Layouts
        bipartite_layout = kwargs.pop("bipartite_layout", False)
        sugiyama_layout = kwargs.pop("sugiyama_layout", False)
        circular_layout = kwargs.pop("circular_layout", False)
        kamada_kawai_layout = kwargs.pop("kamada_kawai_layout", False)
        drl_layout = kwargs.pop("drl_layout", False)
        if bipartite_layout:
            layout = graph.layout_bipartite(types=net.node_types)
            layout.rotate(90)
            kwargs["wrap_labels"] = False
            kwargs["layout"] = layout
        elif sugiyama_layout:
            layout = graph.layout_sugiyama(weights="weight", hgap=50, maxiter=100000)
            layout.rotate(270)
            kwargs["wrap_labels"] = False
            kwargs["layout"] = layout
        elif circular_layout:
            kwargs["layout"] = graph.layout_reingold_tilford_circular()
        elif kamada_kawai_layout:
            kwargs["layout"] = graph.layout_kamada_kawai()
        elif drl_layout:
            kwargs["layout"] = graph.layout_drl(weights="weight")
        # Node and edge scaling
        PHI = 1.618
        scale_nodes_by = kwargs.pop("scale_nodes_by", None)
        if scale_nodes_by is not None:
            try:
                dist = getattr(net, scale_nodes_by)
            except AttributeError:
                dist = Series(net.nodes[scale_nodes_by])
            except TypeError:
                dist = Series(scale_nodes_by)
            if abs(dist.skew()) < 2:
                dist **= 2
            norm = (dist - dist.mean()) / dist.std()
            basesize = np.array(kwargs.pop("vertex_size"))
            mult = basesize / abs(norm).max()
            sizes = (norm * mult / PHI + basesize).fillna(0)
            kwargs["vertex_size"] = sizes
        scale_edges_by = kwargs.pop("scale_edges_by", None)
        if scale_edges_by is not None:
            if scale_edges_by in net.graph.edge_attributes():
                dist = Series(net.edges[scale_edges_by])
            else:
                dist = Series(scale_edges_by)
            if abs(dist.skew()) < 2:
                dist **= 2
            norm = (dist - dist.mean()) / dist.std()
            basewidth = np.array(kwargs.pop("edge_width"))
            mult = basewidth / abs(norm).max()
            widths = (PHI / 2 * norm * mult + (basewidth * PHI / 2)).fillna(0)
            kwargs["edge_width"] = widths
        # Node and edge opacity
        node_opacity = kwargs.pop("vertex_opacity", None)
        edge_opacity = kwargs.pop("edge_opacity", None)
        if node_opacity is not None:
            kwargs["vertex_color"] = [
                _add_opacity(c, node_opacity) for c in kwargs["vertex_color"]
            ]
        if edge_opacity is not None:
            kwargs["edge_color"] = [_add_opacity(kwargs["edge_color"], edge_opacity)]
        # Node and edge labels
        label_doc_nodes = kwargs.pop("label_doc_nodes", False)
        label_term_nodes = kwargs.pop("label_term_nodes", False)
        label_nodes = kwargs.pop("label_nodes", False)
        label_edges = kwargs.pop("label_edges", False)
        kwargs.setdefault(
            "vertex_label",
            [
                node["id"]
                if (node["type"] == "doc" and label_doc_nodes)
                or (node["type"] == "term" and label_term_nodes)
                or label_nodes
                else None
                for node in net.nodes
            ],
        )
        kwargs.setdefault(
            "edge_label",
            [f"{edge['weight']:.2f}" if label_edges else None for edge in net.edges],
        )
        # Node and edge label filters
        node_label_filter = kwargs.pop("vertex_label_filter", False)
        edge_label_filter = kwargs.pop("edge_label_filter", False)
        if node_label_filter and "vertex_label" in kwargs:
            node_labels = kwargs.pop("vertex_label")
            filtered_node_labels = map(node_label_filter, net.nodes)
            kwargs["vertex_label"] = [
                lbl if keep else None
                for lbl, keep in zip(node_labels, filtered_node_labels)
            ]
        if edge_label_filter and "edge_label" in kwargs:
            edge_labels = kwargs.pop("edge_label")
            filtered_edge_labels = map(edge_label_filter, net.edges)
            kwargs["edge_label"] = [
                lbl if keep else None
                for lbl, keep in zip(edge_labels, filtered_edge_labels)
            ]
        return plot_func(net, **kwargs)

    return wrapper


def _add_opacity(color: str, alpha: float) -> tuple:
    """Turn a color name into a RGBA tuple with specified opacity."""
    return (*color_name_to_rgba(color)[:3], alpha)


def _cluster_node_indices(vc: ig.VertexClustering) -> Iterator[List[int]]:
    """Return node indices for nodes in each cluster."""
    for n in range(vc._len):
        yield [i for i, x in enumerate(vc.membership) if x == n]
