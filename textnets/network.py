# -*- coding: utf-8 -*-

import pandas as pd
import igraph as ig
import leidenalg


class Textnets:
    def __init__(self, tidy_text, sublinear=True):
        assert set(tidy_text.columns) == {'word', 'n'}
        self._df = _tf_idf(tidy_text, sublinear)

    def graph(self, node_type):
        assert node_type in ('groups', 'words'), \
            'No valid node_type specified.'
        m = pd.pivot_table(self._df,
                           values='tf_idf',
                           index=self._df.index,
                           columns='word',
                           aggfunc=sum,
                           fill_value=0)
        if node_type == 'groups':
            prod = m.dot(m.T)
        elif node_type == 'words':
            prod = m.T.dot(m)
        g = ig.Graph.Weighted_Adjacency(pd.np.array(prod).tolist(),
                                        attr='weight',
                                        mode='undirected',
                                        loops=False)
        g.vs['label'] = g.vs['id'] = list(prod.index)
        return g


def _tf_idf(tidy_text, sublinear):
    if sublinear:
        tidy_text['tf'] = tidy_text['n'].map(_sublinear_scaling)
    else:
        totals = tidy_text.groupby(tidy_text.index).sum()\
            .rename(columns={'n': 'total'})
        tidy_text = tidy_text.merge(totals, right_index=True, left_index=True)
        tidy_text['tf'] = tidy_text['n'] / tidy_text['total']
    idfs = pd.np.log10(len(tidy_text.index) / tidy_text['word'].value_counts())
    tt = tidy_text.merge(pd.DataFrame(idfs),
                         left_on='word',
                         right_index=True)\
                     .rename(columns={'word_y': 'idf'})
    tt['tf_idf'] = tt['tf'] * tt['idf']
    wc = tt.groupby('word').count()['total']
    tt.merge(wc > 1, on='word').rename(columns={'total_y': 'keep'})
    return tt[tt['keep'][['word', 'n', 'tf_idf']]


def _sublinear_scaling(n):
    return 1 + pd.np.log10(n) if n > 0 else 0


def _sort_by_centrality(vs, graph):
    sub = graph.subgraph(vs)
    lookup = {i: n for n, i in enumerate(vs)}
    centrality = sub.strength(weights='weight')
    centrality_values = [centrality[lookup[v]] for v in vs]
    sorted_values = reversed(sorted(zip(centrality_values, vs)))
    return [v for _, v in sorted_values]


def _cluster_labels(graph, partition, min_size):
    label_lists = []
    for cluster in partition:
        if len(cluster) >= min_size:
            sorted_vs = _sort_by_centrality(cluster, graph)
            label_lists.append([graph.vs['label'][n] for n in sorted_vs])
    labels = ['\x1f'.join(lbl) for lbl in label_lists]
    return labels


def cluster_graph(graph, method=leidenalg.ModularityVertexPartition, min_size=1):
    '''Create cluster graph using Leiden modularity algorithm by default.'''
    methods = {'Modularity': leidenalg.ModularityVertexPartition,
               'CPM': leidenalg.CPMVertexPartition,
               'Surprise': leidenalg.SurpriseVertexPartition}
    if isinstance(method, str):
        method = methods[method]
    part = leidenalg.find_partition(graph, method, weights='weight')
    cluster_g = part.cluster_graph(combine_edges=sum)
    pruned_cluster_vs = cluster_g.vs.select(
        [v for v, s in enumerate(part.sizes()) if s >= min_size])
    pruned_cluster_g = cluster_g.subgraph(pruned_cluster_vs)
    pruned_cluster_g.vs['size'] = [s for s in part.sizes() if s >= min_size]
    pruned_cluster_g.vs['label'] = _cluster_labels(graph, part, min_size)
    return pruned_cluster_g
