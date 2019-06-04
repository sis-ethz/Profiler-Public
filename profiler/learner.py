from sklearn.covariance import graphical_lasso
from sklearn import preprocessing
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
        }
        self.width = -1
        self.Gs = None
        self.idx = None
        self.B = None
        self.p = -1
        self.n = -1
        self.s_p = -1

    def learn(self, data, null_pb, sample_size, **kwargs):
        self.param.update(kwargs)
        self.n = sample_size
        self.p = data.shape[1]
        columns = data.columns.values
        self.est_cov = self.estimate_covariance(data.values, null_pb, columns)
        self.inv_cov = self.estimate_inverse_covariance(self.est_cov.values, columns)

    @staticmethod
    def get_df(matrix, columns):
        df = pd.DataFrame(data=matrix, columns=columns)
        df.index = columns

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
        return inv_cov

    def estimate_covariance(self, X, null_pb, columns):
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

    def recover_dag(self, inv_cov):
        G = self.construct_moral_graphs(inv_cov)
        self.Gs = G.get_undirected_connected_components()
        for G in self.Gs:
            # step 1: tree decomposition
            TD = treewidth_decomp(G)
            # step 2: nice tree decomposition
            NTD = self.nice_tree_decompose(TD)

    def construct_moral_graphs(self, inv_cov):
        G = UndirectedGraph()
        self.idx = pd.DataFrame(zip(np.array(G.add_nodes(inv_cov.columns)), inv_cov.columns),
                                columns=['idx','col']).set_index('col')
        for i, attr in enumerate(inv_cov):
            if i == 0:
                continue
            columns = np.array([c for c in inv_cov.columns.values[:i] if attr.split('_')[0] not in c])
            neighbors = columns[(inv_cov.loc[attr, columns]).abs() > self.env['tol']]
            G.add_undirected_edges([self.idx.loc[attr, 'idx']]*len(neighbors), self.idx.loc[neighbors, 'idx'])
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


