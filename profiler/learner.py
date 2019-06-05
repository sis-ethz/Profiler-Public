from sklearn.covariance import graphical_lasso
from profiler.utility import find_all_subsets, visualize_heatmap
from copy import deepcopy
from profiler.graph import *
import pandas as pd


class StructureLearner(object):

    def __init__(self, env, ds):
        self.env = env
        self.ds = ds
        self.param = {
            'sparsity': 0.01,
            'solver': 'cd',
            'max_iter': 500,
            'lower_triangular': 0,
            'threshold': -1,
            'visualize': True,
        }
        self.width = -1
        self.Gs = None
        self.idx = None
        self.B = None
        self.p = -1
        self.n = -1
        self.s_p = -1
        self.R = {}

    def learn(self, data, null_pb, sample_size, **kwargs):
        self.param.update(kwargs)
        self.n = sample_size
        self.est_cov = self.estimate_covariance(data.values, null_pb, data.columns.values)
        self.inv_cov = self.estimate_inverse_covariance(self.est_cov.values, data.columns.values)
        return self.recover_dag()

    @staticmethod
    def get_df(matrix, columns):
        df = pd.DataFrame(data=matrix, columns=columns)
        df.index = columns
        return df

    def estimate_inverse_covariance(self, est_cov, columns):
        """
        estimate inverse covariance matrix
        :param data: dataframe
        :return: dataframe with attributes as both index and column names
        """
        # estimate inverse_covariance
        _, inv_cov = graphical_lasso(est_cov, alpha=self.param['sparsity'], mode=self.param['solver'],
                                     max_iter=self.param['max_iter'])
        self.s_p = np.count_nonzero(inv_cov)
        # apply threshold
        if self.param['threshold'] == -1:
            self.param['threshold'] = np.sqrt(np.log(self.p)*(self.s_p)/self.n)
        logger.info("use threshold %.4f" % self.param['threshold'])
        inv_cov[inv_cov > self.param['threshold']] = 0
        # add index/column names
        inv_cov = StructureLearner.get_df(inv_cov, columns)
        if self.param['visualize']:
            visualize_heatmap(inv_cov)
        return inv_cov

    def estimate_covariance(self, X, null_pb, columns):
        self.p = X.shape[1]
        # centralize data
        X = X - np.mean(X, axis=0)
        # standardize data
        #X = X / np.linalg.norm(X, axis=0)
        # with missing value
        cov = np.dot(X.T, X) / X.shape[0]
        self.cov = cov
        m = np.ones((cov.shape[0], cov.shape[1])) * (1/np.square(1-null_pb))
        np.fill_diagonal(m, 1/(1-null_pb))
        est_cov = np.multiply(cov, m)
        est_cov = StructureLearner.get_df(est_cov, columns)
        return est_cov

    def recover_dag(self):
        G = self.construct_moral_graphs(self.inv_cov)
        self.Gs = G.get_undirected_connected_components()
        if self.param['visualize']:
            plot_graph(G)
        R = []
        for G in self.Gs:
            if self.param['visualize']:
                plot_graph(G)
            # step 1: tree decomposition
            TD = treewidth_decomp(G)
            if self.param['visualize']:
                plot_graph(TD, label=True)
            # step 2: nice tree decomposition
            NTD = self.nice_tree_decompose(TD)
            if self.param['visualize']:
                plot_graph(NTD, label=False, directed=True)
                print_tree(NTD, NTD.root)
            # step 3: dynamic programming
            self.R = {}
            R = self.dfs(G, NTD, NTD.root)
            R.append(R)
            if self.param['visualize']:
                print(R[0])
            break
        return R

    def construct_moral_graphs(self, inv_cov):
        G = UndirectedGraph()
        self.idx = pd.DataFrame(zip(np.array(G.add_nodes(inv_cov.columns)), inv_cov.columns),
                                columns=['idx','col']).set_index('col')
        for i, attr in enumerate(inv_cov):
            if i == 0:
                continue
            # do not consider a_op1 -> a_op2
            columns = np.array([c for c in inv_cov.columns.values if "_".join(attr.split('_')[0]) not in c])
            neighbors = columns[(inv_cov.loc[attr, columns]).abs() > self.env['tol']]
            G.add_undirected_edges([self.idx.loc[attr, 'idx']]*len(neighbors), self.idx.loc[neighbors, 'idx'])

        if self.param['visualize']:
            plot_graph(G)
        return G

    def nice_tree_decompose(self, TD):
        NTD = deepcopy(TD)
        # set a root with largest bag
        root = -1
        for idx in NTD.idx_to_name:
            if NTD.width+1 == len(NTD.idx_to_name[idx]):
                root = idx
                break
        NTD.set_root_from_node(root)
        # store types
        NTD.node_types = {}
        # decompose
        NTD = nice_tree_decompose(NTD, root)
        return NTD

    """
    def tree_decompose(self, G):
        # ordering = get_min_degree_ordering(G)
        # bags, T = perm_to_tree_decomp(G, ordering)
        # T.width = max([len(bags[i]) for i in bags]) - 1
        # # set bags
        # T.idx_to_name = bags
        return TD
    """

    def score(self, j, S):
        S = list(S)
        k = len(S)
        score = self.est_cov.iloc[j,j] - (self.est_cov.iloc[j,S].values.reshape(1,-1) *
                                          np.linalg.inv(self.est_cov.iloc[S,S].values.reshape(k,k)) *
                                          self.est_cov.iloc[S,j].values.reshape(-1,1))[0][0]
        return score

    def dfs(self, G, tree, t):
        if t in self.R:
            return self.R[t]
        # R(a,p,s): a - parent sets; p: directed path, s:score
        if tree.node_types[t] == JOIN:
            print("check node t = {} with X(t) = {} ".format(t, tree.idx_to_name[t]))
            candidates = {}
            # has children t1 and t2
            t1, t2 = tree.get_children(t)
            for (a1, p1, s1) in self.dfs(G, tree, t1):
                for (a2, p2, s2) in self.dfs(G, tree, t2):
                    if not is_eq_dict(a1, a2):
                        continue
                    a = a1
                    p = union_and_check_cycle([p1, p2])
                    if p is None:
                        continue
                    s = s1 + s2
                    if s not in candidates:
                        candidates[s] = []
                    candidates[s].append((a, p, s))
            Rt = candidates[min(list(candidates.keys()))]
            print("R for node t = {} with X(t) = {} candidate size: {}".format(t, tree.idx_to_name[t],
                                                                               len(tree.idx_to_name[t])))
            self.R[t] = Rt
        elif tree.node_types[t] == INTRO:
            # has only one child
            child = tree.get_children(t)[0]
            Xt = tree.idx_to_name[t]
            Xtc = tree.idx_to_name[child]
            v0 = list(Xt - tree.idx_to_name[child])[0]
            Rt = []
            print("check node t = {} with X(t) = {} ".format(t, Xt))
            for P in find_all_subsets(set(G.get_neighbors(v0))):
                for (aa, pp, ss) in self.dfs(G, tree, child):
                    # parent sets
                    a = {}
                    a[v0] = P
                    for v in Xtc:
                        a[v] = aa[v]
                    # directed path
                    p1 = {}
                    for u in P:
                        p1[u] = [v0]
                    p2 = {}
                    for u in Xtc:
                        for vv in aa[u]:
                            if vv not in p2:
                                p2[vv] = []
                            p2[vv].append(u)
                    p = union_and_check_cycle([pp, p1, p2])
                    if p is None:
                        continue
                    s = ss
                    # since score does not change, all should have same score
                    Rt.append((a, p, s))
            print("R for node t = {} with X(t) = {} candidate size: {}".format(t, Xt, len(Rt)))
            self.R[t] = Rt
        elif tree.node_types[t] == FORGET:
            # has only one child
            child = tree.get_children(t)[0]
            Xt = tree.idx_to_name[t]
            print("check node t = {} with X(t) = {} ".format(t, Xt))
            v0 = list(tree.idx_to_name[child] - Xt)[0]
            candidates = {}
            for (aa, pp, ss) in self.dfs(G, tree, child):
                a = {}
                for v in Xt:
                    a[v] = aa[v]
                p = {}
                for u in pp:
                    if u not in Xt:
                        continue
                    p[u] = [v for v in pp[u] if v in Xt]
                s = ss + self.score(v0, aa[v0])
                if s not in candidates:
                    candidates[s] = []
                candidates[s].append((a, p, s))
            Rt = candidates[min(list(candidates.keys()))]
            print("R for node t = {} with X(t) = {} candidate size: {}".format(t, Xt, len(Rt)))
            self.R[t] = Rt
        else:
            # leaf
            min_score = 100000
            # 1. P is a subset of all the neighbors of the vertex in leaf
            candidates = {}
            Xt = tree.idx_to_name[t]
            v = list(Xt)[0]
            for P in find_all_subsets(set(G.get_neighbors(v))):
                a = {v: P}
                s = self.score(list(Xt)[0], P)
                p = {}
                for u in P:
                    p[u] = [v]
                if s not in candidates:
                    candidates[s] = []
                candidates[s].append((a, p, s))
            # get minimal-score records
            Rt = candidates[min(list(candidates.keys()))]
            self.R[t] = Rt
            print("R for node t = {} with X(t) = {} candidate size: {}".format(t, Xt, len(Rt)))
        return Rt

def union_and_check_cycle(sets):
    s0 = sets[0]
    # each set is a dictionary with left: [all rights] s.t. there is a directed edge from left to right
    for s in sets[1:]:
        for (l, rights) in s.items():
            for r in rights:
                # try to add (l,r) to s0
                # if (r,l) in s0 as well, has a cycle
                if r in s0:
                    if l in s0[r]:
                        return None
                if l in s0:
                    if r in s0[l]:
                        # path already exists
                        continue
                else:
                    s0[l] = []
                # else, add the edge
                s0[l].append(r)
                # add transitive closure
                # everything pointing to l now, should also points to r
                for ll in s0:
                    if ll == l or ll == r:
                        continue
                    if l in s0[ll] and r not in s0[ll]:
                        if r in s0:
                            if ll in s0[r]:
                                # cycle
                                return None
                        s0[ll].append(r)
    return s0

def is_eq_dict(dic1, dic2):
    if len(dic1.keys()) != len(dic2.keys()):
        return False
    for k1 in dic1:
        if k1 not in dic2:
            return False
        if dic1[k1] != dic2[k1]:
            return False
    return True

def plot_graph(graph, label=False, directed=False, circle=False):
    import networkx as nx
    import matplotlib.pyplot as plt
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for e in graph.get_edges():
        if label:
            G.add_edge(graph.idx_to_name[e[0]], graph.idx_to_name[e[1]])
        else:
            G.add_edge(e[0], e[1])
    if circle:
        nx.draw(G, with_labels=True, pos=nx.circular_layout(G))
    else:
        nx.draw(G, with_labels=True)
    plt.draw()
    plt.show()
    return G

def print_tree(T, node, level=0):
    print("{}[{}]{}".format("--"*level, node, T.idx_to_name[node]))
    for c in T.get_children(node):
        print_tree(T, c, level+1)
