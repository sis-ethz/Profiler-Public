from sklearn.covariance import graphical_lasso
from networkx.algorithms.approximation.treewidth import treewidth_min_fill_in
from sklearn import preprocessing
import pandas as pd
import networkx as nx


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

    def run(self, data, **kwargs):
        self.param.update(kwargs)
        inv_cov = self.estimate_inverse_covariance(data)
        self.G = self.construct_moral_graph(inv_cov)
        self.tree_decompos()

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
        return pd.DataFrame(data=inv_cov, columns=data.columns)

    def construct_moral_graph(self, inv_cov):
        columns = inv_cov.columns.values
        G = nx.Graph()
        for attr in inv_cov:
            G.add_node(attr)
            neighbors = columns[inv_cov[attr] > self.param['zero']]
            G.add_edges_from(zip(len(neighbors)*[attr], neighbors))
        return G

    def tree_decompos(self):
        self.width, self.G = treewidth_min_fill_in(self.G)


