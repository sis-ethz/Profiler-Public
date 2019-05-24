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
            'max_iter': 300,
            'zero': 0,
        }
        self.width = -1
        self.Gs = None
        self.idx = None

    def recover_moral(self, data, **kwargs):
        self.param.update(kwargs)
        inv_cov = self.estimate_inverse_covariance(data)
        self.inv_cov = inv_cov
        return inv_cov

    def recover_dag(self, inv_cov):
        G = self.construct_moral_graphs(inv_cov)
        self.Gs = G.get_undirected_connected_components()
        for G in self.Gs:
            # step 1: tree decomposition
            TD = self.tree_decompose(G)
            # step 2: nice tree decomposition
            NTD = self.nice_tree_decompose(TD)


    def estimate_inverse_covariance(self, data):
        """

        :param data: dataframe
        :return: dataframe with attributes as both index and column names
        """
        # Scale dataset
        scaler = preprocessing.MinMaxScaler()
        d_scaled = pd.DataFrame(scaler.fit_transform(data))
        est_cov = d_scaled.cov().values
        _, inv_cov = graphical_lasso(est_cov, alpha=self.param['sparsity'], mode=self.param['solver'],
                                     max_iter=self.param['max_iter'])
        inv_cov = pd.DataFrame(data=inv_cov, columns=data.columns)
        inv_cov.index = inv_cov.columns.values
        return inv_cov

    def construct_moral_graphs(self, inv_cov):
        G = DirectedGraph()
        self.idx = pd.DataFrame(zip(np.array(G.add_nodes(inv_cov.columns)), inv_cov.columns),
                                columns=['idx','col']).set_index('col')
        for i, attr in enumerate(inv_cov):
            if i == 0:
                continue
            columns = np.array([c for c in inv_cov.columns.values[:i] if attr.split('_')[0] not in c])
            neighbors = columns[(inv_cov.loc[attr, columns]).abs() > self.env['tol']]
            G.add_directed_edges([self.idx.loc[attr, 'idx']]*len(neighbors), self.idx.loc[neighbors, 'idx'])
        return G

    def tree_decompose(self, G):
        ordering = get_min_degree_ordering(G)
        bags, T = perm_to_tree_decomp(G, ordering)
        T.width = max([len(bags[i]) for i in bags]) - 1
        # set bags
        T.idx_to_name = bags
        return T

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


