from sklearn.covariance import graphical_lasso
from networkx.algorithms.approximation.treewidth import treewidth_min_fill_in
from sklearn import preprocessing
from graph import *
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
        self.G = None

    def recover_moral(self, data, **kwargs):
        self.param.update(kwargs)
        inv_cov = self.estimate_inverse_covariance(data)
        self.inv_cov = inv_cov
        return inv_cov

    def recover_dag(self, inv_cov):
        self.G = self.construct_moral_graph(inv_cov)
        self.decomposed = self.tree_decompos()

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

    def construct_moral_graph(self, inv_cov):
        G = UndirectedGraph()
        for i, attr in enumerate(inv_cov):
            if i == 0:
                continue
            columns = inv_cov.columns.values[:i]
            columns = np.array([c for c in columns if attr.split('_')[0] not in c])
            neighbors = columns[(inv_cov.loc[attr, columns]).abs() > self.env['tol']]
            G.add_undirected_edges([attr]*len(neighbors), neighbors)
        return G

    def tree_decompos(self):
        ordering = get_min_degree_ordering(self.G)
        self.ordering = ordering
        bags, T = perm_to_tree_decomp(self.G, ordering, ordering)
        self.width = max([len(bags[i]) for i in bags]) - 1
        return bags, T


