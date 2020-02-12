# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la


class Textnets:
    def __init__(self, tidy_text, sublinear=True, min_docs=2):
        assert set(tidy_text.columns) == {'word', 'n'}
        self._df = _tf_idf(tidy_text, sublinear, min_docs)
        im = self._df.pivot(values='tf_idf',
                            columns='word').fillna(0)
        g = ig.Graph.Incidence(np.array(im).tolist(),
                               directed=False)
        g.vs['id'] = np.append(im.index, im.columns).tolist()
        g.es['weight'] = np.array(im).flatten()[np.flatnonzero(im)]
        g.vs['type'] = ['term' if t else 'doc' for t in g.vs['type']]
        self.graph = g


    def project(self, node_type):
        assert node_type in ('doc', 'term'), \
            'No valid node_type specified.'
        node_types = [True if t == 'term' else False for t in self.graph.vs['type']]
        graph_to_return = 0
        if node_type == 'term':
            graph_to_return = 1
        return self.graph.bipartite_projection(types=node_types,
                                               multiplicity=True,
                                               which=graph_to_return)


    def cluster(self, resolution=0.5):
        part, part0, part1 = la.CPMVertexPartition.Bipartite(self.graph,
                                                             resolution_parameter_01=resolution)
        opt = la.Optimiser()
        opt.optimise_partition_multiplex([part, part0, part1],
                                         layer_weights=[1,-1,-1],
                                         n_iterations=100)
        return part


def _tf_idf(tidy_text, sublinear, min_docs):
    if sublinear:
        tidy_text['tf'] = tidy_text['n'].map(_sublinear_scaling)
    else:
        totals = tidy_text.groupby(tidy_text.index).sum()\
            .rename(columns={'n': 'total'})
        tidy_text = tidy_text.merge(totals, right_index=True, left_index=True)
        tidy_text['tf'] = tidy_text['n'] / tidy_text['total']
    idfs = pd.np.log10(len(set(tidy_text.index)) / tidy_text['word'].value_counts())
    tt = tidy_text.merge(pd.DataFrame(idfs),
                         left_on='word',
                         right_index=True)\
                     .rename(columns={'word_y': 'idf'})
    tt['tf_idf'] = tt['tf'] * tt['idf']
    wc = tt.groupby('word').count()['tf']
    tt = tt.reset_index().merge(wc >= min_docs, on='word', how='left')\
                     .rename(columns={'tf_y': 'keep'})\
                     .set_index('index')
    return tt[tt['keep']][['word', 'n', 'tf_idf']]


def _sublinear_scaling(n):
    return 1 + pd.np.log10(n) if n > 0 else 0
