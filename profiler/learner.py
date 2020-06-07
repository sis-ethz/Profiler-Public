import matplotlib
# matplotlib.use("Agg")
from sklearn.covariance import graphical_lasso
from sklearn import covariance
from profiler.utility import find_all_subsets, visualize_heatmap
from sksparse.cholmod import cholesky, analyze
from scipy import sparse
from copy import deepcopy
from profiler.graph import *
from scipy.cluster.vq import vq, kmeans, whiten
import operator


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
            'visualize': False,
            'take_neg': False,
            'take_pos': False,
            'infer_order': False,
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
        # self.visualize_inverse_covariance()
        if np.all(np.linalg.eigvals(self.inv_cov) > 0) == False:
            return np.zeros([self.inv_cov.shape[0], self.inv_cov.shape[1]])
        if not self.param['infer_order']:
            self.B = self.upper_decompose(self.inv_cov)
        else:
            self.B = self.upper_decompose_ordered(self.inv_cov)
        return self.B

    def learn_separate(self, **kwargs):
        self.param.update(kwargs)
        self.cov = self.estimate_covariance()
        self.inv_cov, _ = self.estimate_inverse_covariance(self.cov.values)
        G = self.session.struct_engine.recover_moral_graphs(self.inv_cov)
        Gs = G.get_undirected_connected_components()
        self.Bs = [self.upper_decompose(self.inv_cov.iloc[list(g.idx_to_name.keys()),
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

    # modified by Yunjia on 10/07/2019
    # output all the attrs that are involed in the FD
    def visualize_inverse_covariance(self, filename="Inverse Covariance Matrix"):
        visualize_heatmap(self.inv_cov, title="Inverse Covariance Matrix", filename=filename)

        # clean up the diagnal for sum up 
        np_inv = self.inv_cov.values
        for i in range(len(np_inv)):
            np_inv[i,i] = 0
        
        np_abs_inv = np.abs(self.inv_cov.values)
        np_inv_sum = np.sum(np.abs(self.inv_cov.values), axis=0)
        np_inv_sum = np.nan_to_num(np_inv_sum).astype(np.float64)
        
        col_num = np_abs_inv.shape[0]
        dict1 = {}
        # print("none-zero pairs of abs inv")
        for i in range(col_num):
            for j in range(i+1, col_num):
                if np_abs_inv[i,j] > 0:
                    dict1[ "%s <-> %s" % (self.inv_cov.columns[i], self.inv_cov.columns[j]) ] = np_abs_inv[i,j]
        sorted_x = sorted(dict1.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
        
        # formated prints of the high value pairs
        # with open('./Inv_Cov_Attrs.txt','w') as f:
        #     for i in range(len(dict1)):
        #         f.write(sorted_x[i][0] + "\t" + str(sorted_x[i][1]) + "\n")

        # # manual threshold
        # threshold = len(np_inv_sum) * 2
        # # dynamic threshold using k-means to split it into small values and large values        
        # # centers, _ = kmeans(np_inv_sum, k_or_guess=2)
        # # threshold = np.mean(centers)
        # # traditioinally set all attributes to 0
        # # threshold = 1


        # print("threshold = ", threshold)
        # print("sum = ", np_inv_sum)
        # print("attr = ", self.inv_cov.columns)
        
        # print("Attr w/o dependency: \n",self.inv_cov.columns[np.argwhere(np_inv_sum <= threshold)])
        # print("\n\nAttr w/ dependency: \n",self.inv_cov.columns[np.argwhere(np_inv_sum > threshold)])


    def visualize_covariance(self, filename='Covariance Matrix', write_pairs_file=None):
        # print("===== \nCov: \n", self.cov)
        # print("columns: ", self.cov.columns)
        # print("idices: ", self.cov.index)
        if write_pairs_file is not None:
            threshold = 0.3
            with open(write_pairs_file, 'w') as g:
                for col in self.cov.columns:
                    for idx in self.cov.index:
                        if idx == col:
                            break
                        if self.cov[col].loc[idx] >= threshold:
                            g.write("{} -> {}\n".format(col, idx))
                            # print("get pair: {} -> {}".format(col, idx))

        visualize_heatmap(
            self.cov, title="Covariance Matrix", filename=filename)

    def visualize_autoregression(self, filename="Autoregression Matrix"):
        if self.B is not None:
            visualize_heatmap(self.B, title="Autoregression Matrix", filename=filename)
        else:
            for i, B in enumerate(self.Bs):
                visualize_heatmap(B, title="Autoregression Matrix (Part %d)"%(i+1), filename= filename + " (Part %d)"%(i+1))

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

    def get_dependencies(self, heatmap, score, write_to=None):

        def get_dependencies_helper(U_hat, s_func, write_to=None, by_col=True):
            parent_sets = {}
            if write_to is not None:
                if by_col:
                    file_name = write_to + "_by_col"
                else:
                    file_name = write_to + "_by_row"
                fd_file = open(file_name + ".txt", 'w')
                # attr_file = open(file_name + "_attr.txt", 'w')
            
            # for i, attr in enumerate(U_hat):
            for i in range(U_hat.shape[0]):
                if by_col:
                    attr = U_hat.columns[i]
                    columns = U_hat.columns.values[0:i]
                    parents = columns[(U_hat.iloc[0:i, i] > 0).values]
                    parent_sets[attr] = parents
                else:
                    attr = U_hat.columns[i] # columns are the same as index
                    columns = U_hat.columns.values[i+1 : ]
                    parents = columns[U_hat.iloc[i].values[i+1 :] > 0]
                    parent_sets[attr] = parents
                if len(parents) > 0:
                    s, _ = s_func((parents, attr))
                    fd_file.write("{} -> {}\n".format(",".join(parents), attr))
                    # attr_file.write(attr + "\n")
                    print("{} -> {} ({})".format(",".join(parents), attr, s))
            fd_file.close()
            # attr_file.close()
            return parent_sets

        if score == "training_data_fd_vio_ratio":
            scoring_func = self.training_data_fd_violation
        elif score == "fit_error":
            scoring_func = self.fit_error
        else:
            scoring_func = (lambda x: ("n/a", None))
        parent_sets = {}
        if heatmap is None:
            if self.B is not None:
                parent_sets = get_dependencies_helper(self.B, scoring_func, write_to=write_to)
            elif self.Bs is not None:
                parent_sets = {}
                for B in self.Bs:
                    parent_sets.update(get_dependencies_helper(B, scoring_func, write_to=write_to))
        else:
            parent_sets = get_dependencies_helper(heatmap, scoring_func, write_to=write_to)
        return parent_sets

    @staticmethod
    def get_df(matrix, columns):
        df = pd.DataFrame(data=matrix, columns=columns)
        df.index = columns
        return df

    def get_ordering(self, inv_cov):
        G = self.recover_moral_graphs(inv_cov)
        order = []
        while G.degrees.shape[0] > 0:
            to_delete = G.degrees.index.values[G.degrees.degree.values.argmin()]
            order.append(to_delete)
            G.delete_node(to_delete)
        return order

    def upper_decompose(self, inv_cov):
        I = np.eye(inv_cov.shape[0])
        P = np.rot90(I)
        PAP = np.dot(np.dot(P, inv_cov), P.transpose())
        PAP = sparse.csc_matrix(PAP)
        factor = cholesky(PAP)
        L = factor.L_D()[0].toarray()
        U = np.dot(np.dot(P, L), P.transpose())
        B = I - U
        B = StructureLearner.get_df(B, np.flip(np.flip(inv_cov.columns.values)[factor.P()]))
        return B

    def upper_decompose_ordered(self, inv_cov):
        order = self.get_ordering(inv_cov)
        K = inv_cov.iloc[order, order]
        I = np.eye(inv_cov.shape[0])
        P = np.rot90(I)
        PAP = np.dot(np.dot(P, K), P.transpose())
        PAP = sparse.csc_matrix(PAP)
        factor = cholesky(PAP)
        L = factor.L_D()[0].toarray()
        U = np.dot(np.dot(P, L), P.transpose())
        B = I - U
        B = StructureLearner.get_df(B, np.flip(np.flip(K.columns.values)[factor.P()]))
        return B

    def cholesky_decompose(self, inv_cov):
        # cholesky decomposition of invcov
        A = sparse.csc_matrix(inv_cov.values)
        factor = cholesky(A)
        perm = factor.P()
        L = np.transpose(factor.L_D()[0].toarray())
        B = np.eye(L.shape[0]) - L
        B_hat = StructureLearner.get_df(B, inv_cov.columns.values[perm])
        return B_hat

    def estimate_inverse_covariance(self, cov, shrinkage=0.0):
        """
        estimate inverse covariance matrix
        :param data: dataframe
        :return: dataframe with attributes as both index and column names
        """
        # estimate inverse_covariance
        columns = self.session.trans_engine.training_data.columns

        # shrink covariance for ill-formed datset
        if shrinkage > 0:
            print("[INFO]: using sklearn shrunk_covariance ", shrinkage)
            cov = covariance.shrunk_covariance(cov, shrinkage=shrinkage)

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
        # print("x type: ", X.dtype, "\n", X[:5,:])
        # if X.dtype is np.str:
        #     print("X has type str rather than float!")
        #     print(X[:5,:])
        # X = X - np.mean(X, axis=0)
        # with missing value
        cov = np.dot(X.T, X) / X.shape[0]
        m = np.ones((cov.shape[0], cov.shape[1])) * (1/np.square(1-self.session.trans_engine.null_pb))
        np.fill_diagonal(m, 1/(1-self.session.trans_engine.null_pb))
        est_cov = np.multiply(cov, m)
        if self.param['take_pos']:
            est_cov[est_cov < 0] = 0
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
        """
        helper method for loh's algorithm
        :param R:
        :return:
        """
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
        """
        helper method for loh's algorithm
        :param inv_cov:
        :return:
        """
        G = UndirectedGraph()
        idx_col = pd.DataFrame(list(zip(np.array(G.add_nodes(inv_cov.columns)), inv_cov.columns)),
                                columns=['idx','col']) # former version: without list(), due to higher version of pandas
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
        """
        helper method for loh's algorithm
        :param TD:
        :return:
        """
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
        """
        helper method for loh's algorithm
        :param NTD:
        :param node:
        :param from_idx:
        :return:
        """
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
        """
        helper method for loh's algorithm
        :param j:
        :param S:
        :return:
        """
        S = list(S)
        if len(S) == 0:
            return self.est_cov.iloc[j,j]
        k = len(S)
        score = self.est_cov.iloc[j,j] - (np.dot(np.dot(self.est_cov.iloc[j,S].values.reshape(1,-1),
                                          np.linalg.inv(self.est_cov.iloc[S,S].values.reshape(k,k))),
                                          self.est_cov.iloc[S,j].values.reshape(-1,1)))
        return float(score)

    def dfs(self, G, tree, t):
        """
        helper method for loh's algorithm
        :param G:
        :param tree:
        :param t:
        :return:
        """
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
    import matplotlib
    matplotlib.use("Agg")
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


