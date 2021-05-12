# -*- coding: utf-8 -*-

"""Implements visualization features."""

from __future__ import annotations

from math import ceil

from igraph.drawing.colors import (
    PrecalculatedPalette,
    lighten,
    darken,
    color_name_to_rgba,
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

        blocks_to_add = ceil(float(n - num_base_colors) / num_base_colors)
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

        colors = colors[0:n]
        super().__init__(colors)
