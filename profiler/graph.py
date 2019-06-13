from heapq import heappush, heappop, heapify
from profiler.globalvar import *
import pandas as pd
import numpy as np
import logging
import itertools


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Graph(object):
    
    def __init__(self):
        self.idx_to_name = {}
        self.largest_idx = -1
        self.edges = pd.DataFrame()

    def add_node(self, name, idx=None):
        if idx is None:
            self.largest_idx += 1
            idx = self.largest_idx
        else:
            self.largest_idx = max(idx, self.largest_idx)
        self.idx_to_name[idx] = name
        logger.debug("added new nodes {} ({})".format(name, idx))
        return idx

    def add_nodes_with_idx(self, pair):
        """

        :param pair: (idx, name)
        :return:
        """
        return list(map(lambda n: self.add_node(n[1], idx=n[0]), pair))

    def add_nodes(self, names):
        return list(map(lambda n: self.add_node(n), names))

    def init_edge(self, idx):
        self.edges.loc[idx, idx] = np.nan

    def add_directed_edges(self, idx1, idx2):
        for n1, n2 in zip(idx1, idx2):
            self.add_directed_edge(n1, n2)

    def add_undirected_edges(self, idx1, idx2):
        for n1, n2 in zip(idx1, idx2):
            self.add_undirected_edge(n1, n2)

    def add_undirected_edge(self, idx1, idx2):
        self.add_directed_edge(idx1, idx2)
        self.add_directed_edge(idx2, idx1)

    def add_directed_edge(self, idx1, idx2):
        self.edges.loc[idx1, idx2] = 1
        logger.debug('added directed edge {}:{} -> {}:{}'.format(idx1, self.idx_to_name[idx1],
                                                                 idx2, self.idx_to_name[idx2]))

    def exist_undirected_edge(self, idx1, idx2):
        return self.exist_edge(idx1, idx2) and self.exist_edge(idx2, idx1)

    def exist_edge(self, idx1, idx2):
        return self.edges.loc[idx1, idx2] == 1

    def print_edges(self, undirected):
        for e in self.get_edges(undirected):
            print(e)

    def get_edges(self, undirected=False):
        row, col = np.where(np.asanyarray(~np.isnan(self.edges)))
        for r, c in zip(row, col):
            if undirected:
                if r >= c:
                    continue
            yield(self.edges.index[r], self.edges.columns[c])

    def delete_node(self, idx):
        logger.debug("deleted node %s"%idx)
        self.edges.drop(idx, axis=1, inplace=True)
        self.edges.drop(idx, axis=0, inplace=True)

    def get_neighbors(self, idx):
        return np.unique(np.concatenate([self.get_parents(idx), self.get_children(idx)]))

    def get_children(self, idx):
        return self.edges.index.values[self.edges.loc[idx, :] == 1]

    def get_parents(self, idx):
        return self.edges.columns.values[self.edges[idx] == 1]

    def remove_edge(self, idx1, idx2):
        logger.debug("remove directed edge %d - %d"%(idx1, idx2))
        self.edges.loc[idx1, idx2] = np.nan

    def remove_undirected_edge(self, idx1, idx2):
        self.remove_edge(idx1, idx2)
        self.remove_edge(idx2, idx1)

    
class DirectedGraph(Graph):
    
    def __init__(self):
        super(DirectedGraph, self).__init__()
        self.in_degrees = pd.DataFrame()
        self.out_degrees = pd.DataFrame()

    def add_node(self, name, idx=None):
        idx = super(DirectedGraph, self).add_node(name, idx)
        self.init_edge(idx)
        return idx

    def init_edge(self, idx):
        super(DirectedGraph, self).init_edge(idx)
        if idx not in self.in_degrees.index:
            self.in_degrees.loc[idx, 'degree'] = 0
        if idx not in self.out_degrees.index:
            self.out_degrees.loc[idx, 'degree'] = 0
        
    def add_directed_edge(self, idx1, idx2):
        if not self.exist_edge(idx1, idx2):
            self.in_degrees.loc[idx2, 'degree'] += 1
            self.out_degrees.loc[idx1, 'degree'] += 1
            super(DirectedGraph, self).add_directed_edge(idx1, idx2)

    def add_undirected_edge(self, idx1, idx2):
        self.add_directed_edge(idx1, idx2)
        self.add_directed_edge(idx2, idx1)

    def remove_edge(self, idx1, idx2):
        if self.exist_edge(idx1, idx2):
            super(DirectedGraph, self).remove_edge(idx1, idx2)
            self.in_degrees.loc[idx2, 'degree'] -= 1
            self.out_degrees.loc[idx1, 'degree'] -= 1

    def delete_node(self, idx):
        for child in self.get_children(idx):
            self.in_degrees.loc[child, 'degree'] -= 1
        for parent in self.get_parents(idx):
            self.out_degrees.loc[parent, 'degree'] -= 1
        self.in_degrees.drop(idx, axis=0)
        self.out_degrees.drop(idx, axis=0)
        super(DirectedGraph, self).delete_node(idx)

    def to_undirected(self):
        G = UndirectedGraph()
        G.add_nodes_with_idx(G.idx_to_name.items())
        G.edges = pd.DataFrame()
        for e in self.get_edges():
            G.add_undirected_edge(e[0], e[1])
        return G


class UndirectedGraph(Graph):

    def __init__(self):
        super(UndirectedGraph, self).__init__()
        self.degrees = pd.DataFrame()

    def add_node(self, name, idx=None):
        idx = super(UndirectedGraph, self).add_node(name, idx)
        self.init_edge(idx)
        return idx

    def init_edge(self, idx):
        super(UndirectedGraph, self).init_edge(idx)
        self.degrees.loc[idx, 'degree'] = 0

    def add_undirected_edge(self, idx1, idx2):
        if not self.exist_undirected_edge(idx1, idx2):
            self.degrees.loc[idx1, 'degree'] += 1
            self.degrees.loc[idx2, 'degree'] += 1
            super(UndirectedGraph, self).add_undirected_edge(idx1, idx2)

    def delete_node(self, idx):
        self.degrees.drop(idx, axis=0, inplace=True)
        for nbr in self.get_neighbors(idx):
            self.degrees.loc[nbr, 'degree'] -= 1
        super(UndirectedGraph, self).delete_node(idx)


    def remove_undirected_edge(self, idx1, idx2):
        if self.exist_undirected_edge(idx1, idx2):
            super(UndirectedGraph, self).remove_undirected_edge(idx1, idx2)
            self.degrees.loc[idx1, 'degree'] -= 1
            self.degrees.loc[idx2, 'degree'] -= 1

    def to_directed(self):
        G = DirectedGraph()
        G.idx_to_name = self.idx_to_name
        G.out_degrees = self.degrees
        G.in_degrees = self.degrees
        G.edges = self.edges
        return G

    def to_tree(self):
        G = Tree()
        G.idx_to_name = self.idx_to_name
        G.out_degrees = self.degrees
        G.in_degrees = self.degrees
        G.edges = self.edges
        return G

    def get_undirected_connected_components(self):

        visited = np.zeros((self.largest_idx+1,))
        Gs = []

        def recursive_add_children(G, start):
            for c in self.get_neighbors(start):
                if visited[c] == 0:
                    G.add_node(self.idx_to_name[c], idx=c)
                    visited[c] = 1
                    G = recursive_add_children(G, c)
            return G

        def get_component(start):
            G = UndirectedGraph()
            # add nodes
            G.add_node(self.idx_to_name[start], idx=start)
            visited[start] = 1
            G = recursive_add_children(G, start)
            nodes = list(G.idx_to_name.keys())
            # extract edge
            G.edges = self.edges.loc[nodes, nodes]
            G.degrees = self.degrees.loc[nodes]
            return G

        # pick a random node as start
        to_visit = np.where(visited == 0)[0]
        while to_visit.shape[0] > 0:
            if to_visit[0] not in self.idx_to_name:
                visited[to_visit[0]] = 1
                to_visit = to_visit[1:]
                continue
            G = get_component(to_visit[0])
            if len(G.idx_to_name) > 1:
                Gs.append(G)
            to_visit = np.where(visited == 0)[0]

        return Gs


class Tree(DirectedGraph):

    def __init__(self):
        super(Tree, self).__init__()
        self.root = None

    def add_root(self, name, idx=None):
        idx = super(Tree, self).add_node(name, idx)
        self.root = idx
        return idx

    def set_root_from_node(self, idx):
        self.root = idx


class MinDegreeHeuristic:
    """ Implements the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree
    (number of neighbours), i.e., first the node with the lowest degree is
    chosen, then the graph is updated and the corresponding node is
    removed. Next, a new node with the lowest degree is chosen, and so on.

    Copyright (C) 2004-2019, NetworkX Developers
    Aric Hagberg <hagberg@lanl.gov>
    Dan Schult <dschult@colgate.edu>
    Pieter Swart <swart@lanl.gov>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

      * Redistributions in binary form must reproduce the above
        copyright notice, this list of conditions and the following
        disclaimer in the documentation and/or other materials provided
        with the distribution.

      * Neither the name of the NetworkX Developers nor the names of its
        contributors may be used to endorse or promote products derived
        from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    """
    def __init__(self, graph):
        self._graph = graph

        # nodes that have to be updated in the heap before each iteration
        self._update_nodes = []

        self._degreeq = []  # a heapq with 2-tuples (degree,node)

        # build heap with initial degrees
        for n in graph:
            self._degreeq.append((len(graph[n]), n))
        heapify(self._degreeq)

    def best_node(self, graph):
        # update nodes in self._update_nodes
        for n in self._update_nodes:
            # insert changed degrees into degreeq
            heappush(self._degreeq, (len(graph[n]), n))

        # get the next valid (minimum degree) node
        while self._degreeq:
            (min_degree, elim_node) = heappop(self._degreeq)
            if elim_node not in graph or len(graph[elim_node]) != min_degree:
                # outdated entry in degreeq
                continue
            elif min_degree == len(graph) - 1:
                # fully connected: abort condition
                return None

            # remember to update nodes in the heap before getting the next node
            self._update_nodes = graph[elim_node]
            return elim_node

        # the heap is empty: abort
        return None


def treewidth_decomp(G):
    """ Returns a treewidth decomposition using the passed heuristic.

    Copyright (C) 2004-2019, NetworkX Developers
    Aric Hagberg <hagberg@lanl.gov>
    Dan Schult <dschult@colgate.edu>
    Pieter Swart <swart@lanl.gov>
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

      * Redistributions in binary form must reproduce the above
        copyright notice, this list of conditions and the following
        disclaimer in the documentation and/or other materials provided
        with the distribution.

      * Neither the name of the NetworkX Developers nor the names of its
        contributors may be used to endorse or promote products derived
        from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Parameters
    ----------
    G : NetworkX graph
    heuristic : heuristic function

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """

    # make dict-of-sets structure
    graph = {n: set(G.get_neighbors(n)) - set([n]) for n in G.idx_to_name.keys()}
    min_degree = MinDegreeHeuristic(graph)

    # stack containing nodes and neighbors in the order from the heuristic
    node_stack = []

    # get first node from heuristic
    elim_node = min_degree.best_node(graph)

    while elim_node is not None:
        # connect all neighbours with each other
        nbrs = graph[elim_node]
        for u, v in itertools.permutations(nbrs, 2):
            if v not in graph[u]:
                graph[u].add(v)

        # push node and its current neighbors on stack
        node_stack.append((elim_node, nbrs))

        # remove node from graph
        for u in graph[elim_node]:
            graph[u].remove(elim_node)

        del graph[elim_node]
        elim_node =  min_degree.best_node(graph)

    # the abort condition is met; put all remaining nodes into one bag
    decomp = Tree()
    first_bag = frozenset(graph.keys())
    decomp.add_node(first_bag)

    treewidth = len(first_bag) - 1
    decomp.width = treewidth

    while node_stack:
        # get node and its neighbors from the stack
        (curr_node, nbrs) = node_stack.pop()

        # find a bag all neighbors are in
        old_bag = None
        for bag in decomp.idx_to_name.values():
            if nbrs <= bag:
                old_bag = bag
                break

        if old_bag is None:
            # no old_bag was found: just connect to the first_bag
            old_bag = first_bag

        # create new node for decomposition
        nbrs.add(curr_node)
        new_bag = frozenset(nbrs)

        # update treewidth
        treewidth = max(treewidth, len(new_bag) - 1)

        # add edge to decomposition (implicitly also adds the new node)
        old_bag_idx = -1
        new_bag_idx = -1
        for idx, s in decomp.idx_to_name.items():
            if s == old_bag:
                old_bag_idx = idx
            elif s == new_bag:
                new_bag_idx = idx

        if old_bag_idx == -1:
            print(old_bag)
            old_bag_idx = decomp.add_node(old_bag)
        if new_bag_idx == -1:
            print(new_bag)
            new_bag_idx = decomp.add_node(new_bag)
        decomp.add_undirected_edge(old_bag_idx, new_bag_idx)
        decomp.width = treewidth

    return decomp


def add_forget(T, current, to_forget):
    T.node_types[current] = FORGET
    for n in to_forget:
        # create new node
        idx = T.add_node(T.idx_to_name[current].union(frozenset([n])))
        T.node_types[idx] = FORGET
        # link the node to current
        T.add_directed_edge(current, idx)
        current = idx
    return T, current


def add_intro(T, current, to_intro):
    T.node_types[current] = INTRO
    for n in to_intro:
        # create new node
        idx = T.add_node(T.idx_to_name[current] - frozenset([n]))
        T.node_types[idx] = INTRO
        # link the node to current
        T.add_directed_edge(current, idx)
        current = idx
    return T, current


def nice_tree_decompose(T, node):
    if node == T.root:
        # add forget nodes till root
        to_forget = list(T.idx_to_name[node])[:-1]
        root = T.add_node(frozenset([]))
        T, current = add_forget(T, root, to_forget)
        T.add_directed_edge(current, node)
        T.set_root_from_node(root)
    nbr = T.get_children(node)
    logger.debug("node {} neighbors: {}".format(node, nbr))
    if len(nbr) > 2:
        T.node_types[node] = JOIN
        logger.debug("nbr > 2")
        # if first child has different bag from current,
        # add one node above it as left with same bag
        if T.idx_to_name[nbr[0]] != T.idx_to_name[node]:
            # remove node - child1
            T.remove_undirected_edge(node, nbr[0])
            # add new node same as current node
            newl = T.add_node(T.idx_to_name[node])
            # parent of new node is current
            T.add_directed_edge(node, newl)
            # child of new node is 1st child
            T.add_directed_edge(newl, nbr[0])
            T = nice_tree_decompose(T, newl)
        else:
            # keep directed edge
            T.remove_edge(nbr[0], node)
            T = nice_tree_decompose(T, nbr[0])
        # add new node as right share same bag with current
        newr = T.add_node(T.idx_to_name[node])
        T.add_directed_edge(node, newr)
        # for other children, make it to be newr's children
        for n in nbr[1:]:
            # cut the connection with current
            T.remove_undirected_edge(node, n)
            # add parents to be newr
            T.add_directed_edge(newr, n)
        T = nice_tree_decompose(T, newr)
    elif len(nbr) == 2:
        T.node_types[node] = JOIN
        logger.debug("nbr = 2")
        for n in nbr:
            # if child has different bag from current, add one node above it with same bag
            if T.idx_to_name[n] != T.idx_to_name[node]:
                new = T.add_node(T.idx_to_name[node])
                # parent of new node is current
                T.add_directed_edge(node, new)
                # child of new node is 1st child
                T.add_directed_edge(new, n)
                T.remove_undirected_edge(node, n)
                T = nice_tree_decompose(T, new)
            else:
                # keep directed edge
                T.remove_edge(n, node)
                # search subnodes
                T = nice_tree_decompose(T, n)
    elif len(nbr) == 1:
        logger.debug("nbr = 1")
        # check relationship with the only child
        s_parent = T.idx_to_name[node]
        s_child = T.idx_to_name[nbr[0]]
        common = s_child.intersection(s_parent, s_child)
        # remove connection with child
        T.remove_undirected_edge(node, nbr[0])
        if len(common) == len(s_parent):
            if len(common) == len(s_child):
                # exactly the same, skip child
                for n in T.get_children(nbr[0]):
                    T.add_directed_edge(node, n)
                # recursive call to current
                T = nice_tree_decompose(T, node)
            else:
                # common < child, add one each time
                # no need introduce but need to forget, parent is common
                to_forget = list(s_child - common)[:-1]
                T, current = add_forget(T, node, to_forget)
                # link to the child
                T.add_directed_edge(current, nbr[0])
                T = nice_tree_decompose(T, nbr[0])
        else:
            # common < parent, need to introduce
            to_intro = list(s_parent - common)[:-1]
            # reduce one node each time until reach common
            T, current = add_intro(T, node, to_intro)
            if len(common) == len(s_child):
                # no need to forget, link to child
                T.add_directed_edge(current, nbr[0])
                # recursive call to child
                T = nice_tree_decompose(T, nbr[0])
            else:
                # need to forget link to common
                cm_idx = T.add_node(common)
                T.add_directed_edge(current, cm_idx)
                # from common add forget nodes
                to_forget = list(s_child - common)[:-1]
                T, current = add_forget(T, cm_idx, to_forget)
                # link to the child
                T.add_directed_edge(current, nbr[0])
                # recursive call to child
                T = nice_tree_decompose(T, nbr[0])
    else:
        logger.debug("leaf")
        # is leaf, has no children, no recursive call in the end
        s_curr = T.idx_to_name[node]
        if len(s_curr) == 1:
            # bag of size 1
            T.node_types[node] = LEAF
            pass
        else:
            # need to add introduce node
            to_intro = list(s_curr)[:-1]
            T, current = add_intro(T, node, to_intro)
            T.node_types[current] = LEAF
    return T

"""
def get_min_degree_ordering(G):
    # sort degree in ascending order
    ordering = G.degrees.sort_values(by='degree')
    ordering['order'] = range(ordering.shape[0])
    return ordering


def fill_in_graph(G, ordering, v):
    H = copy.deepcopy(G)
    # for each pair of neighbors w, x of v, ordering[w] > ordering[v] and ordering[x] > ordering[v]
    nbr = sorted(H.get_neighbors(v), key=lambda x: ordering.loc[x, 'order'], reverse=True)
    for j in nbr[:-1]:
        for k in nbr[j+1:]:
            # add edges for all pair of neighbors
            H.add_undirected_edge(nbr[j], nbr[k])
    if len(nbr) > 0:
        return H, nbr[-1]
    return H


def eliminate_node(G, v):
    H = copy.deepcopy(G)
    # eliminate the node
    H.delete_node(v)
    return H


def perm_to_tree_decomp(G, ordering):
    v1 = ordering.index.values[0]
    logger.debug("\n\nv1: %s"%v1)
    T = Tree()
    if ordering.shape[0] == 1:
        # return decomp with one bag
        bag = {v1: frozenset([v1])}
        logger.debug("return subtree with bag for v1: {}".format(bag))
        T.add_node(v1)
        return bag, T
    logger.debug("eliminated v1: %s"%v1)
    H = eliminate_node(G, v1)
    bags, T_prime = perm_to_tree_decomp(H, ordering.iloc[1:, :])
    # construct a bag for neighbor of v1 and find vj -- v1's neighbor with smallest order
    vj = None
    bags[v1] = frozenset(np.concatenate([G.get_neighbors(v1), [v1]]))
    for nbr in ordering.iloc[1:, :].index.values:
        if nbr in bags[v1]:
            vj = nbr
    logger.debug("bag for {}: {}".format(v1, bags[v1]))
    # nodes = G.nodes
    T.edges = T_prime.edges
    T.in_degrees = T_prime.in_degrees
    T.out_degrees = T_prime.out_degrees
    T.add_nodes_with_idx(G.idx_to_name.items())
    # union edges in T_prime with {v1,vj}
    if vj is not None:
        T.add_undirected_edge(v1, vj)
        logger.debug("added v1 %s - vj %s"%(v1, vj))
    return bags, T
"""
