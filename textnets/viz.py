# -*- coding: utf-8 -*-

"""Extends visualization features."""

from __future__ import annotations

from functools import wraps
from itertools import repeat
from math import ceil
from typing import Iterator, List

import igraph as ig
from igraph.drawing.colors import (
    PrecalculatedPalette,
    color_name_to_rgba,
    darken,
    lighten,
)

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


def add_opacity(color: str, alpha: float) -> tuple:
    """Turns a color name into a RGBA tuple with specified opacity."""
    return tuple([*color_name_to_rgba(color)[:3], alpha])


def decorate_plot(plot_func):
    """Style the plot produced by igraph's plot function."""

    @wraps(plot_func)
    def wrapper(*args, **kwargs):
        textnet = args[0]
        graph = textnet.graph
        # Marking and coloring clusters
        show_clusters = kwargs.pop("show_clusters", False)
        color_clusters = kwargs.pop("color_clusters", False)
        if show_clusters:
            if isinstance(show_clusters, ig.VertexClustering):
                markers = zip(
                    _cluster_node_indices(show_clusters),
                    repeat(add_opacity("limegreen", 0.4)),
                )
            else:
                markers = zip(
                    _cluster_node_indices(textnet.clusters),
                    repeat(add_opacity("limegreen", 0.4)),
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
                    TextnetPalette(textnet.clusters._len)[c]
                    for c in textnet.clusters.membership
                ]
        # Some sane defaults
        kwargs.setdefault("autocurve", True)
        kwargs.setdefault("edge_color", "lightgray")
        kwargs.setdefault(
            "layout", graph.layout_fruchterman_reingold(weights="weight", grid=False)
        )
        kwargs.setdefault("margin", 50)
        kwargs.setdefault("wrap_labels", True)
        kwargs.setdefault("vertex_frame_width", 0.2)
        kwargs.setdefault("vertex_label_size", 10)
        kwargs.setdefault("edge_label_size", 8)
        kwargs.setdefault(
            "vertex_color",
            ["orangered" if v else "dodgerblue" for v in textnet.node_types],
        )
        kwargs.setdefault(
            "vertex_shape", ["circle" if v else "square" for v in textnet.node_types]
        )
        kwargs.setdefault(
            "vertex_frame_color",
            ["black" if v else "white" for v in textnet.node_types],
        )
        # Layouts
        bipartite_layout = kwargs.pop("bipartite_layout", False)
        sugiyama_layout = kwargs.pop("sugiyama_layout", False)
        circular_layout = kwargs.pop("circular_layout", False)
        kamada_kawai_layout = kwargs.pop("kamada_kawai_layout", False)
        drl_layout = kwargs.pop("drl_layout", False)
        if bipartite_layout:
            layout = graph.layout_bipartite(types=textnet.node_types)
            layout.rotate(90)
            kwargs["wrap_labels"] = False
            kwargs["layout"] = layout
        elif sugiyama_layout:
            kwargs["layout"] = graph.layout_sugiyama(
                weights="weight", hgap=50, maxiter=100000
            )
        elif circular_layout:
            kwargs["layout"] = graph.layout_reingold_tilford_circular()
        elif kamada_kawai_layout:
            kwargs["layout"] = graph.layout_kamada_kawai()
        elif drl_layout:
            kwargs["layout"] = layout = graph.layout_drl(weights="weight")
        # Node scaling
        scale_nodes_by = kwargs.pop("scale_nodes_by", False)
        if scale_nodes_by:
            dist = getattr(textnet, scale_nodes_by)
            if abs(dist.skew()) < 2:
                dist **= 2
            norm = (dist - dist.mean()) / dist.std()
            mult = 20 / abs(norm).max()
            sizes = (norm * mult + 25).fillna(0)
            kwargs.setdefault("vertex_size", sizes)
        # Node and edge opacity
        node_opacity = kwargs.pop("node_opacity", None)
        edge_opacity = kwargs.pop("edge_opacity", None)
        if node_opacity is not None:
            kwargs["vertex_color"] = [
                add_opacity(c, node_opacity) for c in kwargs["vertex_color"]
            ]
        if edge_opacity is not None:
            kwargs["edge_color"] = [add_opacity(kwargs["edge_color"], edge_opacity)]
        # Node and edge labels
        label_doc_nodes = kwargs.pop("label_doc_nodes", False)
        label_term_nodes = kwargs.pop("label_term_nodes", False)
        label_nodes = kwargs.pop("label_nodes", False)
        label_edges = kwargs.pop("label_edges", False)
        kwargs.setdefault(
            "vertex_label",
            [
                v["id"]
                if (v["type"] == "doc" and label_doc_nodes)
                or (v["type"] == "term" and label_term_nodes)
                or label_nodes
                else None
                for v in textnet.nodes
            ],
        )
        kwargs.setdefault(
            "edge_label",
            [f"{e['weight']:.2f}" if label_edges else None for e in textnet.edges],
        )
        # Node and edge label filters
        node_label_filter = kwargs.pop("node_label_filter", False)
        edge_label_filter = kwargs.pop("edge_label_filter", False)
        if node_label_filter and "vertex_label" in kwargs:
            node_labels = kwargs.pop("vertex_label")
            filtered_node_labels = map(node_label_filter, textnet.nodes)
            kwargs["vertex_label"] = [
                lbl if keep else None
                for lbl, keep in zip(node_labels, filtered_node_labels)
            ]
        if edge_label_filter and "edge_label" in kwargs:
            edge_labels = kwargs.pop("edge_label")
            filtered_edge_labels = map(edge_label_filter, textnet.edges)
            kwargs["edge_label"] = [
                lbl if keep else None
                for lbl, keep in zip(edge_labels, filtered_edge_labels)
            ]
        # Rewrite remaining node_* arguments as vertex_* arguments
        node_opts = [k for k, _ in kwargs.items() if k.startswith("node_")]
        for opt in node_opts:
            val = kwargs.pop(opt)
            kwargs[opt.replace("node_", "vertex_")] = val
        print(f"{args=} {kwargs=}")
        return plot_func(*args, **kwargs)

    return wrapper


def _cluster_node_indices(vc: ig.VertexClustering) -> Iterator[List[int]]:
    """Return node indices for nodes in each cluster."""
    for n in range(vc._len):
        yield [i for i, x in enumerate(vc.membership) if x == n]
