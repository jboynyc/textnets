# -*- coding: utf-8 -*-

from functools import cached_property
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la


class Textnet:
    def __init__(self, tidy_text, sublinear=True, min_docs=2):
        self._df = _tf_idf(tidy_text, sublinear, min_docs)
        im = self._df.pivot(values='tf_idf',
                            columns='term').fillna(0)
        self.im = im
        g = ig.Graph.Incidence(im.to_numpy().tolist(),
                               directed=False)
        g.vs['id'] = np.append(im.index, im.columns).tolist()
        g.es['weight'] = im.to_numpy().flatten()[np.flatnonzero(im)]
        g.vs['type'] = ['term' if t else 'doc' for t in g.vs['type']]
        self.graph = g

    @cached_property
    def node_types(self):
        return [True if t == 'term' else False for t in self.graph.vs['type']]


    def project(self, node_type):
        assert node_type in ('doc', 'term'), \
            'No valid node_type specified.'
        graph_to_return = 0
        if node_type == 'term':
            graph_to_return = 1
            weights = self.im.T.dot(self.im).to_numpy()
        else:
            weights = self.im.dot(self.im.T).to_numpy()
        np.fill_diagonal(weights, 0)
        weights = np.triu(weights)
        weights = weights.flatten()[np.flatnonzero(weights)]
        graph = self.graph.bipartite_projection(types=self.node_types,
                                                which=graph_to_return)
        graph.es['weight'] = weights
        return graph


    def plot(self, mark_groups=False, bipartite_layout=False, label_nodes=('term')):
        if bipartite_layout:
            layout = self.graph.layout_bipartite(types=self.node_types)
        else:
            layout = self.graph.layout_fruchterman_reingold(weights='weight',
                                                            grid=False)
        return ig.plot(self.graph,
                       layout=layout,
                       margin=50,
                       autocurve=True,
                       vertex_shape=['circle' if v else 'square'
                                     for v in self.node_types],
                       vertex_color=['orangered' if v else 'dodgerblue'
                                     for v in self.node_types],
                       vertex_frame_color=['black' if v else 'white'
                                           for v in self.node_types],
                       vertex_frame_width=.2,
                       mark_groups=self.clusters if mark_groups else False,
                       vertex_label=[v['id'] if v['type'] in label_nodes else None
                                     for v in self.graph.vs])


    @cached_property
    def clusters(self):
        return self._partition_graph()


    @cached_property
    def context(self):
        return self._formal_context(cutoff=0.3)


    def _partition_graph(self, resolution=0.5):
        #TODO: fix random seed
        part, part0, part1 = la.CPMVertexPartition.Bipartite(self.graph,
                                                             resolution_parameter_01=resolution)
        opt = la.Optimiser()
        opt.optimise_partition_multiplex([part, part0, part1],
                                         layer_weights=[1,-1,-1],
                                         n_iterations=100)
        return part


    def _formal_context(self, cutoff=0.3):
        # The incidence matrix is a "fuzzy formal context." We can binarize it
        # by using a cutoff. This is known as the alpha-cut.
        # See doi:10.1016/j.knosys.2012.10.005 and doi:10.1016/j.asoc.2017.05.028
        objects = self.im.index.tolist()
        properties = self.im.columns.tolist()
        bools = self.im.applymap(lambda x: True if x > cutoff else False).to_numpy()
        return objects, properties, bools


def _tf_idf(tidy_text, sublinear, min_docs):
    if sublinear:
        tidy_text['tf'] = tidy_text['n'].map(_sublinear_scaling)
    else:
        totals = tidy_text.groupby(tidy_text.index).sum()\
            .rename(columns={'n': 'total'})
        tidy_text = tidy_text.merge(totals, right_index=True, left_index=True)
        tidy_text['tf'] = tidy_text['n'] / tidy_text['total']
    idfs = np.log10(len(set(tidy_text.index)) / tidy_text['term'].value_counts())
    tt = tidy_text.merge(pd.DataFrame(idfs),
                         left_on='term',
                         right_index=True)\
                     .rename(columns={'term_y': 'idf'})
    tt['tf_idf'] = tt['tf'] * tt['idf']
    wc = tt.groupby('term').count()['tf']
    tt = tt.reset_index().merge(wc >= min_docs, on='term', how='left')\
                     .rename(columns={'tf_y': 'keep'})\
                     .set_index('label')
    return tt[tt['keep']][['term', 'n', 'tf_idf']]


def _sublinear_scaling(n):
    return 1 + np.log10(n) if n > 0 else 0
