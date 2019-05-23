import pandas as pd
import numpy as np
import logging, copy


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Graph(object):
    
    def __init__(self):
        self.nodes = {}
        self.idx_to_name = {}
        self.num_nodes = 0
        self.edges = pd.DataFrame()

    def add_node(self, name):
        self.num_nodes += 1
        self.nodes[name] = self.num_nodes
        self.idx_to_name[self.num_nodes] = name
        logger.debug('created node %s' % name)
        return self.num_nodes

    def add_nodes(self, names):
        return list(map(lambda n: self.add_node(n), names))

    def init_edge(self, idx):
        self.edges.loc[idx, idx] = np.nan

    def add_directed_edges(self, names1, names2):
        for n1, n2 in zip(names1, names2):
            self.add_directed_edge(n1, n2)

    def add_undirected_edges(self, names1, names2):
        for n1, n2 in zip(names1, names2):
            self.add_undirected_edge(n1, n2)

    def add_undirected_edge(self, name1, name2):
        idx = self.add_nodes([name1, name2])
        self.add_undirected_edge_from_nodes(idx[0], idx[1])

    def add_directed_edge(self, name1, name2):
        idx = self.add_nodes([name1, name2])
        self.add_directed_edge_from_nodes(idx[0], idx[1])

    def add_undirected_edge_from_nodes(self, idx1, idx2):
        if idx1 < idx2:
            self.add_directed_edge_from_nodes(idx1, idx2)
        else:
            self.add_directed_edge_from_nodes(idx2, idx1)

    def add_directed_edge_from_nodes(self, idx1, idx2):
        self.edges.loc[idx1, idx2] = 1
        logger.debug('added directed edge {} -> {}'.format(self.idx_to_name[idx1], self.idx_to_name[idx2]))

    def exist_undirected_edge(self, idx1, idx2):
        if idx1 < idx2:
            return self.exist_edge(idx1, idx2)
        else:
            return self.exist_edge(idx2, idx1)

    def exist_edge(self, idx1, idx2):
        return self.edges.loc[idx1, idx2] == 1

    def print_edges(self):
        row, col = np.where(np.asanyarray(~np.isnan(self.edges)))
        for r, c in zip(row, col):
            print((self.edges.index[r], self.edges.columns[c]))

    def delete_node(self, idx):
        del self.nodes[self.idx_to_name[idx]]
        self.edges[idx] = []
        self.edges.loc[idx, :] = []

    def get_neighbors(self, idx):
        return np.unique(np.concatenate([self.get_parents(idx), self.get_children(idx)]))

    def get_children(self, idx):
        return self.edges.index.values[self.edges.loc[idx, :] == 1]

    def get_parents(self, idx):
        return self.edges.columns.values[self.edges[idx] == 1]

    
class DirectedGraph(Graph):
    
    def __init__(self):
        super(DirectedGraph, self).__init__()
        self.in_degrees = pd.DataFrame()
        self.out_degrees = pd.DataFrame()

    def add_node(self, name):
        if name not in self.nodes:
            idx = super(DirectedGraph, self).add_node(name)
            self.init_edge(idx)
            return idx

    def init_edge(self, idx):
        super(DirectedGraph, self).init_edge(idx)
        self.in_degrees.loc[idx, 'degree'] = 0
        self.out_degrees.loc[idx, 'degree'] = 0
        
    def add_directed_edge_from_nodes(self, idx1, idx2):
        if not self.exist_edge(idx1, idx2):
            self.in_degrees.loc[idx2] += 1
            self.out_degrees.loc[idx1] += 1
            super(DirectedGraph, self).add_directed_edge_from_nodes(idx1, idx2)

    def delete_node(self, idx):
        super(DirectedGraph, self).delete_node(idx)
        self.in_degrees.loc[idx] = []
        self.out_degrees.loc[idx] = []


class UndirectedGraph(Graph):

    def __init__(self):
        super(UndirectedGraph, self).__init__()
        self.degrees = pd.DataFrame()

    def add_node(self, name):
        if name not in self.nodes:
            idx = super(UndirectedGraph, self).add_node(name)
            self.init_edge(idx)
            return idx
        else:
            return self.nodes[name]

    def init_edge(self, idx):
        super(UndirectedGraph, self).init_edge(idx)
        self.degrees.loc[idx, 'degree'] = 0

    def add_undirected_edge_from_nodes(self, idx1, idx2):
        if not self.exist_edge(idx1, idx2):
            self.degrees.loc[idx1] += 1
            self.degrees.loc[idx2] += 1
            super(UndirectedGraph, self).add_undirected_edge_from_nodes(idx1, idx2)

    def delete_node(self, idx):
        super(UndirectedGraph, self).delete_node(idx)
        self.degrees.loc[idx] = []


class Tree(DirectedGraph):

    def __init__(self):
        super(Tree, self).__init__()
        self.root = None

    def add_root(self, name):
        if name not in self.nodes:
            idx = super(Tree, self).add_node(name)
            self.root = idx
        else:
            return self.nodes[name]

    def set_root_from_node(self, idx):
        self.root = idx

    def add_subtree(self, T):
        pass


def get_min_degree_ordering(G):
    # sort degree in ascending order
    ordering = G.degrees.sort_values(by='degree')
    ordering['order'] = range(ordering.shape[0])
    return ordering


def eliminate_node(G, ordering, v):
    H = copy.deepcopy(G)
    # for each pair of neighbors w, x of v, ordering[w] > ordering[v] and ordering[x] > ordering[v]
    nbr = sorted(G.get_neighbors(v), key=lambda x: ordering.loc[x, 'order'], reverse=True)
    for j in nbr[:-1]:
        if j <= ordering.loc[v, 'order']:
            continue
        for k in nbr[j+1:]:
            if k <= ordering.loc[v, 'order']:
                continue
            # add edges for all pair of neighbors
            H.add_undirected_edge_from_nodes(nbr[j], nbr[k])
            # eliminate the node
            #H.delete_node(v)
    return H, nbr[-1]


def perm_to_tree_decomp(G, ordering, pi):
    v1 = ordering.index.values[0]
    T = Tree()
    if ordering.shape[0] == 1:
        # return decomp with one bag
        bag = {v1: frozenset([v1])}
        T.add_node(v1)
        return bag, T
    H, vj = eliminate_node(G, pi, v1)
    # vj is v1's neighbor with smallest order
    bags, T_prime = perm_to_tree_decomp(H, ordering.iloc[1:, :], pi)
    # construct a bag for neighbor of v1
    bags[vj] = frozenset(G.get_neighbors(v1))
    # nodes = G.nodes
    T.edges = T_prime.edges
    T.add_nodes(list(G.nodes.keys()))
    # union edges in T_prime with {v1,vj}
    T.add_undirected_edge_from_nodes(v1, vj)
    return bags, T
