from __future__ import division
from detector import Detector
from sklearn.covariance import graphical_lasso
from functools import partial
from multiprocessing import Pool
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.linalg import ldl
from tqdm import tqdm
from ..globalvar import *
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_undirected_difference(attr, self, df):
    if self.dataEngine.dtypes[attr] == NUMERIC:
        if self.param['euclidean_ball']:
            # numerical
            logger.info("Treat %s as numerical"%attr)
            diff = (np.square(df['left_'+attr] - df['right_'+attr]) <= self.param['error_bound'])*2 - 1
        else:
            logger.info("Treat %s as numerical use kmeans"%attr)
            kmeans = KMeans(n_clusters=2)
            val = (df['left_'+attr] - df['right_'+attr]).values
            is_nan = np.isnan(val)
            nan_id = np.array(range(val.shape[0]))[is_nan]
            val_nonan = val[~is_nan].reshape(-1, 1)
            kmeans.fit(val_nonan)
            # make sure 0 from same 1 for different
            labels = kmeans.labels_.astype('float')
            if kmeans.predict(np.zeros((1, 1)))[0] == 1:
                # 1 for same
                labels = labels*2 - 1
            else:
                # 0 for same
                labels = (1-labels)*2 - 1
            for nid in nan_id:
                labels = np.insert(labels, nid, np.nan)
            diff = labels
    else:
        # categorical
        if self.dataEngine.param['use_embedding'] and self.dataEngine.dtypes[attr] == EMBEDDABLE:
            logger.info("Treat %s as embeddable categorical"%attr)
            diff = (1-self.embedding.get_pair_distance(df['left_'+attr], df['right_'+attr], attr=attr))*2-1
        else:
            logger.info("Treat %s as categorical"%attr)
            # same: 1, different: -1
            diff = (df['left_'+attr] == df['right_'+attr])*2 - 1
    return attr, diff


def compute_directed_difference(attr, self, df):
    if self.dataEngine.dtypes[attr] == NUMERIC:
        if self.param['euclidean_ball']:
            # numerical
            diff = 1 - (np.abs(df['left_'+attr] - df['right_'+attr]) /
                        np.nanmax([df['left_'+attr], df['right_'+attr]], axis=0) <= self.param['error_bound'])*1
            logger.info("Treat %s as numerical, get differences in binary"%attr)
        else:
            kmeans = KMeans(n_clusters=2)
            val = (df['left_'+attr] - df['right_'+attr]).values
            is_nan = np.isnan(val)
            nan_id = np.array(range(val.shape[0]))[is_nan]
            val_nonan = val[~is_nan].reshape(-1, 1)
            kmeans.fit(val_nonan)
            # make sure 0 from same 1 for different
            labels = kmeans.labels_.astype('float')
            if kmeans.predict(np.zeros((1,1)))[0] == 1:
                # 1 for same, need to change
                labels = 1 - labels
            for nid in nan_id:
                labels = np.insert(labels, nid, np.nan)
            diff = labels
            logger.info("Treat %s as numerical use kmeans"%attr)
    else:
        # categorical
        if self.dataEngine.param['use_embedding'] and self.dataEngine.dtypes[attr] == EMBEDDABLE:
            diff = self.embedding.get_pair_distance(df['left_'+attr], df['right_'+attr], attr=attr)
            logger.info("Treat %s as embeddable categorical"%attr)
        else:
            # same - 0, different - 1
            diff = 1 - (df['left_'+attr] == df['right_'+attr])*1
            logger.info("Treat %s as categorical"%attr)
    return attr, diff


class GLassoDetector(Detector):

    def __init__(self, parent, **kwargs):
        super(GLassoDetector, self).__init__(parent)
        self.param = {
            'differences': False,
            'undirected': False,
            'decompose': True,
            'alpha_corr': 0,
            'alpha_cov': 0,
            'total_frac': 1,
            'overwrite': False,
            'multiplier': -1,
            'sort_training_data': True,
            'error_bound': 0.0001,
            'euclidean_ball': True,
            'sample_frac': 1,
            'use_cov': True,
            'use_corr': True,
        }
        self.param.update(kwargs)
        self.columns = []
        self.heatmap_name = {}
        self.init_name()
        self.data = None
        self.embedding = self.dataEngine.embedding

    def init_name(self):
        if self.param['undirected']:
            direction = "undirected_heatmap"
        else:
            direction = "heatmap"
        if self.param['decompose']:
            direction = 'decomposed_' + direction
        for name in ['cov', 'corr']:
            alpha = str(self.param['alpha_%s'%name]).replace('.','dot')
            if self.param['differences']:
                self.heatmap_name[name] = "GLD_{}_alpha{}_{}".format(name, alpha, direction)
            else:
                self.heatmap_name[name] = "GL_{}_alpha{}_{}".format(name, alpha, direction)

    @staticmethod
    def decompose(matrix, lower=True):
        lu, d, perm = ldl(matrix, lower=lower) # Use the upper part
        B = lu[perm,:]
        return B, perm

    def get_training_data(self):
        self.profiler.timer.time_start("Create Training Data")
        if self.param['differences']:
            # GLD
            if self.param['sort_training_data']:
                self.dataEngine.create_training_data_column(total_frac=self.param['total_frac'],
                                                            sample_frac=self.param['sample_frac'],
                                                            multiplier=self.param['multiplier'],
                                                            overwrite=self.param['overwrite'])
            else:
                self.dataEngine.create_training_data_row(total_frac=self.param['total_frac'],
                                                         sample_frac=self.param['sample_frac'],
                                                         multiplier=self.param['multiplier'],
                                                         overwrite=self.param['overwrite'])
            train_data = self.dataEngine.trainData
            df = train_data.get_dataframe()
            data = pd.DataFrame()
            self.columns = np.array(self.dataEngine.field)
            p = Pool(self.dataEngine.param['workers'])
            if self.param['undirected']:
                for (attr, diff) in tqdm(p.map(partial(compute_undirected_difference, self=self, df=df), self.columns)):
                    data[attr] = diff
            else:
                for (attr, diff) in tqdm(p.map(partial(compute_directed_difference, self=self, df=df), self.columns)):
                    data[attr] = diff
            # drop zero columns and rows
            drop_col = (data != 0).any(axis=0)
            self.columns = self.columns[drop_col.values]
            data = data.loc[:, (data != 0).any(axis=0)]
            data = data.loc[(data != 0).any(axis=1), :]
        else:
            # GL
            data = pd.DataFrame()
            df = self.dataEngine.get_dataframe()
            self.columns = df.columns.values
            if 'id' in self.columns:
                df.drop('id', axis=1)
            for attr in self.columns:
                data[attr] = df[attr].astype('category').cat.codes
        self.profiler.timer.time_end("Create Training Data")
        return data
    
    def run(self):
        # load training data
        self.data = self.get_training_data()
        self.profiler.timer.time_start("Train Graphical Lasso")
        # Scale dataset
        scaler = preprocessing.MinMaxScaler()
        d_scaled = pd.DataFrame(scaler.fit_transform(self.data))
        m_cov = d_scaled.cov().values
        m_corr = self.data.corr().values

        # Increasing alpha leads to more sparsity
        try:
            if self.param['use_cov']:
                c_cov = graphical_lasso(m_cov,alpha=self.param['alpha_cov'], mode='cd')
        except Exception as e:
            logger.error(" Error running GLD cov: {}".format(e))
            raise

        try:
            if self.param['use_corr']:
                c_corr = graphical_lasso(m_corr,alpha=self.param['alpha_corr'], mode='lars')
        except Exception as e:
            logger.error(" Error running GLD corr: {}".format(e))
            raise

        if self.param['undirected'] or (not self.param['decompose']):
            if self.param['use_cov']:
                self.cov_heatmap = pd.DataFrame(data=c_cov[0], columns=self.columns)
                self.cov_heatmap.index = self.columns
            if self.param['use_corr']:
                self.corr_heatmap = pd.DataFrame(data=c_corr[0], columns=self.columns)
                self.corr_heatmap.index = self.columns
            end = self.profiler.timer.time_end("Train Graphical Lasso")
        else:
            if self.param['use_cov']:
                # decomposition
                B1, perm1 = GLassoDetector.decompose(c_cov[0])
                # get tic
                ticks1 = [self.columns[i] for i in perm1]
                self.cov_heatmap = pd.DataFrame(data=B1, columns=ticks1)
                self.cov_heatmap.index = ticks1
            if self.param['use_corr']:
                B2, perm2 = GLassoDetector.decompose(c_corr[0])
                ticks2 = [self.columns[i] for i in perm2]
                self.corr_heatmap = pd.DataFrame(data=B2, columns=ticks2)
                self.corr_heatmap.index = ticks2
            
            end = self.profiler.timer.time_end("Train Graphical Lasso")                
        return end

