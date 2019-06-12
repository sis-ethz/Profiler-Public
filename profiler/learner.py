from sklearn.covariance import graphical_lasso
from profiler.utility import find_all_subsets, visualize_heatmap
from sksparse.cholmod import cholesky, analyze
from itertools import permutations
from scipy import sparse
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
from profiler.graph import *


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StructureLearner(object):

    def __init__(self, session, env, ds):
        self.session = session
        self.env = env
        self.ds = ds
        self.param = {
            'sparsity': 0.01,
            'solver': 'cd',
            'max_iter': 500,
            'lower_triangular': 0,
            'threshold': -1,
            'visualize': True,
            'take_neg': False,
        }
        self.width = -1
        self.cov = None
        self.inv_cov = None
        self.Bs = None
        self.B = None
        self.idx = None
        self.p = -1
        self.n = -1
        self.s_p = -1
        self.R = {}

    def learn(self, **kwargs):
        self.param.update(kwargs)
        self.cov = self.estimate_covariance()
        self.inv_cov, _ = self.estimate_inverse_covariance(self.cov.values)
        self.B = self.cholesky_decompose(self.inv_cov)
        return self.B

    def learn_separate(self, **kwargs):
        self.param.update(kwargs)
        self.cov = self.estimate_covariance()
        self.inv_cov, _ = self.estimate_inverse_covariance(self.cov.values)
        G = pf.session.struct_engine.recover_moral_graphs(self.inv_cov)
        Gs = G.get_undirected_connected_components()
        self.Bs = [self.cholesky_decompose(self.inv_cov.iloc[list(g.idx_to_name.keys()),
                                                             list(g.idx_to_name.keys())]) for g in Gs]
        return self.Bs

    def learn_dp(self, **kwargs):
        """
        Loh's algorithm
        1. inverse covariance estimation
        2. tree decomposition
        3. nice tree decomposition
        4. dynamic programming to find dag with minimal score
        :param kwargs:
        :return:
        """
        self.param.update(kwargs)
        self.cov = self.estimate_covariance()
        self.inv_cov, self.est_cov = self.estimate_inverse_covariance(self.cov.values)
        G = self.recover_moral_graphs(self.inv_cov)
        Gs = G.get_undirected_connected_components()
        Rs = [self.recover_dag(i, G) for i, G in enumerate(Gs)]

        return Rs

    def visualize_inverse_covariance(self):
        visualize_heatmap(self.inv_cov, title="Inverse Covariance Matrix")

    def visualize_covariance(self):
        visualize_heatmap(self.cov, title="Covariance Matrix")

    def visualize_autoregression(self):
        if self.B is not None:
            visualize_heatmap(self.B, title="Autoregression Matrix")
        else:
            for i, B in enumerate(self.Bs):
                visualize_heatmap(B, title="Autoregression Matrix (Part %d)"%(i+1))

    def training_data_fd_violation(self, pair):
        left, right = pair
        stat = self.session.trans_engine.training_data.reset_index().groupby(list(left)+[right])['index'].count()
        idx = list([1.0]*len(left))
        pos_idx = tuple(idx + [1.0])
        neg_idx = tuple(idx + [0.0])
        if pos_idx not in stat.index:
            return 1, stat
        if neg_idx not in stat.index:
            return 0, stat
        agree = stat.loc[pos_idx]
        disagree = stat.loc[neg_idx]
        ratio = disagree / float(agree+disagree)
        return ratio, stat

    def fit_error(self, pair):
        left, right = pair
        # dim: left * right
        coeff = self.B.loc[left, right].values
        offset = self.session.trans_engine.training_data[right] - \
                 np.dot(self.session.trans_engine.training_data[left].values, coeff)
        # normalize offsets
        bias = np.mean(offset)
        err = np.sum(np.square(offset - bias)) / float(self.session.trans_engine.training_data.shape[0])
        return err, bias

    def get_dependencies(self, heatmap, score):

        def get_dependencies_helper(U_hat, s_func):
            parent_sets = {}
            for i, attr in enumerate(U_hat):
                columns = U_hat.columns.values[0:i]
                parents = columns[(U_hat.iloc[0:i, i] != 0).values]
                parent_sets[attr] = parents
                if len(parents) > 0:
                    s, _ = s_func((parents, attr))
                    print("{} -> {} ({})".format(",".join(parents), attr, s))
            return parent_sets

        if score == "training_data_fd_vio_ratio":
            scoring_func = self.training_data_fd_violation
        elif score == "fit_error":
            scoring_func = self.fit_error
        else:
            scoring_func = (lambda x: ("n/a", None))

        if heatmap is None:
            if self.B is not None:
                parent_sets = get_dependencies_helper(self.B, scoring_func)
            else:
                parent_sets = {}
                for B in self.Bs:
                    parent_sets.update(get_dependencies_helper(B, scoring_func))
        else:
            parent_sets = get_dependencies_helper(heatmap, scoring_func)
        return parent_sets


    @staticmethod
    def get_df(matrix, columns):
        df = pd.DataFrame(data=matrix, columns=columns)
        df.index = columns
        return df

    def cholesky_decompose(self, inv_cov):
        # cholesky decomposition of invcov
        A = sparse.csc_matrix(inv_cov.values)
        factor = analyze(A)
        perm = factor.P()
        mat = inv_cov.iloc[perm, perm]
        A = sparse.csc_matrix(mat.values)
        factor = cholesky(A)
        L = factor.L_D()[0].toarray()
        B = np.eye(L.shape[0]) - np.transpose(L)
        B_hat = StructureLearner.get_df(B, inv_cov.columns.values[perm])
        return B_hat

    def estimate_inverse_covariance(self, cov):
        """
        estimate inverse covariance matrix
        :param data: dataframe
        :return: dataframe with attributes as both index and column names
        """
        # estimate inverse_covariance
        columns = self.session.trans_engine.training_data.columns
        est_cov, inv_cov = graphical_lasso(cov, alpha=self.param['sparsity'], mode=self.param['solver'],
                                           max_iter=self.param['max_iter'])
        self.s_p = np.count_nonzero(inv_cov)
        # apply threshold
        if self.param['threshold'] == -1:
            self.param['threshold'] = np.sqrt(np.log(self.p)*(self.s_p)/self.session.trans_engine.sample_size)
        logger.info("use threshold %.4f" % self.param['threshold'])
        if self.param['take_neg']:
            diag = inv_cov.diagonal()
            diag_idx = np.diag_indices(inv_cov.shape[0])
            mask = -inv_cov
            mask[diag_idx] = diag
        else:
            mask = np.abs(inv_cov)
        inv_cov[mask < self.param['threshold']] = 0
        # add index/column names
        inv_cov = StructureLearner.get_df(inv_cov, columns)
        est_cov = StructureLearner.get_df(est_cov, columns)
        return inv_cov, est_cov

    def estimate_covariance(self):
        X = self.session.trans_engine.training_data.values
        columns = self.session.trans_engine.training_data.columns
        self.p = X.shape[1]
        # centralize data
        X = X - np.mean(X, axis=0)
        # with missing value
        cov = np.dot(X.T, X) / X.shape[0]
        m = np.ones((cov.shape[0], cov.shape[1])) * (1/np.square(1-self.session.trans_engine.null_pb))
        np.fill_diagonal(m, 1/(1-self.session.trans_engine.null_pb))
        est_cov = np.multiply(cov, m)
        est_cov = StructureLearner.get_df(est_cov, columns)
        self.cov = est_cov
        return est_cov

    def recover_dag(self, i, G):
        if self.param['visualize']:
            plot_graph(G, title="%d.1 connected component"%i)
        # step 1: tree decomposition
        TD = treewidth_decomp(G)
        if self.param['visualize']:
            plot_graph(TD, label=True, title="%d.2 tree width decomposition"%i)
        # step 2: nice tree decomposition
        NTD = self.nice_tree_decompose(TD)
        if self.param['visualize']:
            plot_graph(NTD, label=False, directed=True, title="%d.3 nice tree decomposition"%i)
            print_tree(NTD, NTD.root)
        # step 3: dynamic programming
        self.R = {}
        R = self.dfs(G, NTD, NTD.root)[0]
        min_score = R[2]
        r = self.find_record(NTD, NTD.root, 0)
        # optional: visualize
        if self.param['visualize']:
            dag = self.construct_dag_from_record(r)
            plot_graph(dag, label=True, directed=True,
                       title="%d.4 1 possible dag (score=%.4f)"%(i, min_score))
        return r, min_score

    def construct_dag_from_record(self, R):
        a, p, _, _, _ = R
        nodes = set(a.keys())
        for v in a.values():
            nodes = nodes.union(set(v))
        dag = DirectedGraph()
        for n in nodes:
            dag.add_node(self.idx_to_col.loc[n, 'col'], idx=n)
        for child, parents in a.items():
            for p in parents:
                dag.add_directed_edge(p, child)
            if self.param['visualize']:
                score = self.score(child, parents)
                print("{} -> {} ({})".format(", ".join(self.idx_to_col.loc[parents, 'col'].values),
                                        self.idx_to_col.loc[child, 'col'], score))
        return dag

    def recover_moral_graphs(self, inv_cov):
        G = UndirectedGraph()
        idx_col = pd.DataFrame(zip(np.array(G.add_nodes(inv_cov.columns)), inv_cov.columns),
                                columns=['idx','col'])
        self.col_to_idx = idx_col.set_index('col')
        self.idx_to_col = idx_col.set_index('idx')
        for i, attr in enumerate(inv_cov):
            # do not consider a_op1 -> a_op2
            if self.session.env['inequality']:
                columns = np.array([c for c in inv_cov.columns.values if "_".join(attr.split('_')[:-1]) not in c])
            else:
                columns = np.array([c for c in inv_cov.columns.values if attr != c])
            neighbors = columns[(inv_cov.loc[attr, columns]).abs() > 0]
            if len(neighbors) == 0:
                continue
            G.add_undirected_edges([self.col_to_idx.loc[attr, 'idx']]*len(neighbors),
                                   self.col_to_idx.loc[neighbors, 'idx'])
            if self.param['visualize']:
                print("{} -> {}".format(",".join(neighbors), attr))
        if self.param['visualize']:
            plot_graph(G, title="all connected components")
        return G

    def nice_tree_decompose(self, TD):
        NTD = deepcopy(TD)
        # set a root with smallest bag
        root = -1
        min_width = NTD.width + 1 + 1
        for idx in NTD.idx_to_name:
            if len(NTD.idx_to_name[idx]) < min_width:
                min_width = len(NTD.idx_to_name[idx])
                root = idx
        NTD.set_root_from_node(root)
        # store types
        NTD.node_types = {}
        # decompose
        NTD = nice_tree_decompose(NTD, root)
        return NTD

    def find_record(self, NTD, node, from_idx):
        # find R with from_idx
        for r in self.R[node]:
            if r[3] == from_idx:
                break
        _, _, score, _, from_idx = r
        if score == 0:
            # find the record contain all nodes
            return r
        children = NTD.get_children(node)
        if len(children) == 1:
            return self.find_record(NTD, children[0], from_idx)
        else:
            r1 = self.find_record(NTD, children[0], from_idx)
            r2 = self.find_record(NTD, children[1], from_idx)
            return (r1, r2)

    def score(self, j, S):
        S = list(S)
        if len(S) == 0:
            return self.est_cov.iloc[j,j]
        k = len(S)
        score = self.est_cov.iloc[j,j] - (np.dot(np.dot(self.est_cov.iloc[j,S].values.reshape(1,-1),
                                          np.linalg.inv(self.est_cov.iloc[S,S].values.reshape(k,k))),
                                          self.est_cov.iloc[S,j].values.reshape(-1,1)))
        return float(score)

    def dfs(self, G, tree, t):
        if t in self.R:
            return self.R[t]
        # R(a,p,s): a - parent sets; p: directed path, s:score
        if tree.node_types[t] == JOIN:
            logger.debug("check node t = {} with X(t) = {} ".format(t, tree.idx_to_name[t]))
            candidates = {}
            # has children t1 and t2
            t1, t2 = tree.get_children(t)
            i = 0
            for (a1, p1, s1, idx1, _) in self.dfs(G, tree, t1):
                for (a2, p2, s2, idx2, _) in self.dfs(G, tree, t2):
                    if not is_eq_dict(a1, a2):
                        continue
                    a = deepcopy(a1)
                    p = union_and_check_cycle([p1, p2])
                    if p is None:
                        continue
                    s = s1 + s2
                    if s not in candidates:
                        candidates[s] = []
                    candidates[s].append((a, p, s, i, (idx1, idx2)))
                    i += 1
            if len(candidates.keys()) == 0:
                raise Exception("No DAG found")
            Rt = candidates[min(list(candidates.keys()))]
            logger.debug("R for join node t = {} with X(t) = {} candidate size: {}".format(t, tree.idx_to_name[t],
                                                                               len(tree.idx_to_name[t])))
            self.R[t] = Rt
        elif tree.node_types[t] == INTRO:
            # has only one child
            child = tree.get_children(t)[0]
            Xt = tree.idx_to_name[t]
            Xtc = tree.idx_to_name[child]
            v0 = list(Xt - tree.idx_to_name[child])[0]
            Rt = []
            #candidates = {}
            logger.debug("check node t = {} with X(t) = {} ".format(t, Xt))
            i = 0
            for P in find_all_subsets(set(G.get_neighbors(v0))):
                for (aa, pp, ss, idx, _) in self.dfs(G, tree, child):
                    # parent sets
                    a = {}
                    a[v0] = set(P)
                    for v in Xtc:
                        a[v] = set(aa[v])
                    # directed path
                    p1 = {}
                    # p1: parents of new node v0 point to v0
                    for u in P:
                        p1[u] = [v0]
                    p2 = {}
                    # p2: v0 is parent of existing node v0 -> exist
                    p2[v0] = [u for u in Xtc if v0 in aa[u]]
                    p = union_and_check_cycle([pp, p1, p2])
                    if p is None:
                        continue
                    s = ss
                    # since score does not change, all should have same score
                    Rt.append((a, p, s, i, idx))
                    i += 1
            logger.debug("R for intro node t = {} with X(t) = {} candidate size: {}".format(t, Xt, len(Rt)))
            self.R[t] = Rt
        elif tree.node_types[t] == FORGET:
            # has only one child
            child = tree.get_children(t)[0]
            Xt = tree.idx_to_name[t]
            logger.debug("check node t = {} with X(t) = {} ".format(t, Xt))
            v0 = list(tree.idx_to_name[child] - Xt)[0]
            candidates = {}
            i = 0
            for (aa, pp, ss, idx, _) in self.dfs(G, tree, child):
                a = {}
                for v in Xt:
                    a[v] = set(aa[v])
                p = {}
                for u in pp:
                    if u not in Xt:
                        continue
                    p[u] = [v for v in pp[u] if v in Xt]
                s = ss + self.score(v0, aa[v0])
                if s not in candidates:
                    candidates[s] = []
                candidates[s].append((a, p, s, i, idx))
                i += 1
            if len(candidates.keys()) == 0:
                raise Exception("No DAG found")
            Rt = candidates[min(list(candidates.keys()))]
            logger.debug("R for forget node t = {} with X(t) = {} candidate size: {}".format(t, Xt, len(Rt)))
            self.R[t] = Rt
        else:
            # leaf
            # 1. P is a subset of all the neighbors of the vertex in leaf
            candidates = {}
            Xt = tree.idx_to_name[t]
            v = list(Xt)[0]
            for i, P in enumerate(find_all_subsets(set(G.get_neighbors(v)))):
                a = {v: set(P)}
                s = 0
                p = {}
                if s not in candidates:
                    candidates[s] = []
                candidates[s].append((a, p, s, i, -1))
            # get minimal-score records
            Rt = candidates[min(list(candidates.keys()))]
            self.R[t] = Rt
            logger.debug("R for leaf node t = {} with X(t) = {} candidate size: {}".format(t, Xt, len(Rt)))
        return Rt


def union_and_check_cycle(sets, debug=False):
    s0 = None
    # each set is a dictionary with left: [all rights] s.t. there is a directed edge from left to right
    for s in sets:
        if debug:
            logger.debug("s: {}".format(s))
        if len(s) == 0:
            if debug:
                logger.debug("empty, continue")
            continue
        if s0 is None:
            s0 = deepcopy(s)
            if debug:
                logger.debug("assign to s0, continue")
            continue
        if debug:
            logger.debug("merge with s0")
        for (l, rights) in s.items():
            for r in rights:
                # try to add (l,r) to s0
                # if (r,l) in s0 as well, has a cycle
                if r in s0:
                    if l in s0[r]:
                        # cycle
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
    if debug:
        logger.debug("merged: {}".format(s0))
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


def plot_graph(graph, label=False, directed=False, circle=False, title=None):
    import networkx as nx
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,6))
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    e = None
    for e in graph.get_edges():
        if label:
            G.add_edge(graph.idx_to_name[e[0]], graph.idx_to_name[e[1]])
        else:
            G.add_edge(e[0], e[1])
    if e is None:
        for node in graph.idx_to_name.values():
            G.add_node(node)
    if circle:
        nx.draw(G, ax=ax, with_labels=True, pos=nx.circular_layout(G))
    else:
        nx.draw(G, ax=ax, with_labels=True)
    if title is not None:
        plt.title(title)
    plt.draw()
    plt.show()
    return G


def print_tree(T, node, level=0):
    print("{}[{}]{}:{}".format("--"*level, node, T.node_types[node], T.idx_to_name[node]))
    for c in T.get_children(node):
        print_tree(T, c, level+1)


