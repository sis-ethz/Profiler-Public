from __future__ import division
from profiler.detector.detector import Detector
from sklearn.covariance import graphical_lasso
from functools import partial
from multiprocessing import Pool
from sklearn import preprocessing
from scipy.linalg import ldl
from tqdm import tqdm
from profiler.globalvar import *
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_undirected_difference(attr, self, df):
    if self.dataEngine.dtypes[attr] == NUMERIC:
        if not self.param['user_defined']:
            # numerical
            logger.info("Treat %s as numerical"%attr)
            diff = np.abs(df['left_'+attr] - df['right_'+attr])
            # numerical
            diff = (diff / np.nanmax(diff) <= self.param['error_bound'])*2 - 1
        else:
            # same: 1, different: -1
            a = df['left_'+attr]
            b = df['right_'+attr]
            nan = np.isnan(a) | np.isnan(b)
            nan_id = np.array(range(a.shape[0]))[nan]
            try:
                diff = self.param['user_defined'](a[~nan], b[~nan])
                for nid in nan_id:
                    diff = np.insert(diff, nid, np.nan)
            except:
                raise
            logger.info("Treat %s as numerical, using user defined function"%attr)
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
        if not self.param['user_defined']:
            diff = np.abs(df['left_'+attr] - df['right_'+attr])
            # numerical
            diff = 1 - (diff / np.nanmax(diff) <= self.param['error_bound'])*1
            logger.info("Treat %s as numerical" % attr)
        else:
            # same - 0, different - 1
            a = df['left_'+attr]
            b = df['right_'+attr]
            nan = np.isnan(a) | np.isnan(b)
            diff = np.zeros(a.shape[0], dtype=float)
            diff[nan] = np.nan
            diff[~nan] = self.param['user_defined'](a[~nan], b[~nan])
            logger.info("Treat %s as numerical, using user defined function"%attr)
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
            'user_defined': None,
            'sample_frac': 1,
            'use_cov': True,
            'use_corr': True,
            'data': None,
            'max_iter':100,
        }
        self.param.update(kwargs)
        self.columns = []
        self.heatmap_name = {}
        self.init_name()
        self.data = self.param['data']
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
            logger.info('Taking Differences ... ')
            if self.dataEngine.param['workers'] <= 1:
                for attr in tqdm(self.columns):
                    if self.param['undirected']:
                        _, data[attr] = compute_undirected_difference(attr, self=self, df=df)
                    else:
                        _, data[attr] = compute_directed_difference(attr, self=self, df=df)
            else:
                p = Pool(self.dataEngine.param['workers'])
                if self.param['undirected']:
                    for (attr, diff) in p.map(partial(compute_undirected_difference, self=self, df=df), self.columns):
                        data[attr] = diff
                else:
                    for (attr, diff) in p.map(partial(compute_directed_difference, self=self, df=df), self.columns):
                        data[attr] = diff
            # drop zero columns and rows
            drop_col = (data != 0).any(axis=0)
            data = data.loc[:, (data != 0).any(axis=0)]
            data = data.loc[(data != 0).any(axis=1), :]
            # check singular
            to_drop = []
            for col in data:
                if len(data[col].unique()) == 1:
                    to_drop.append(col)
            if len(to_drop) != 0:
                data = data.drop(to_drop, axis=1)
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
        if self.data is None:
            self.data = self.get_training_data()
        self.columns = self.data.columns.values
        self.profiler.timer.time_start("Train Graphical Lasso")
        # Scale dataset
        scaler = preprocessing.MinMaxScaler()
        d_scaled = pd.DataFrame(scaler.fit_transform(self.data))
        m_cov = d_scaled.cov().values
        m_corr = self.data.corr().values

        # Increasing alpha leads to more sparsity
        try:
            if self.param['use_cov']:
                c_cov = graphical_lasso(m_cov,alpha=self.param['alpha_cov'], mode='cd', 
                                        max_iter=self.param['max_iter'])
        except Exception as e:
            logger.error(" Error running GLD cov: {}".format(e))
            return self.data
            raise

        try:
            if self.param['use_corr']:
                c_corr = graphical_lasso(m_corr,alpha=self.param['alpha_corr'], mode='lars', 
                                        max_iter=self.param['max_iter'])
        except Exception as e:
            logger.error(" Error running GLD corr: {}".format(e))
            return self.data
            raise

        if self.param['undirected'] or (not self.param['decompose']):
            if self.param['use_cov']:
                self.cov_heatmap = pd.DataFrame(data=c_cov[1], columns=self.columns)
                self.cov_heatmap.index = self.columns
            if self.param['use_corr']:
                self.corr_heatmap = pd.DataFrame(data=c_corr[1], columns=self.columns)
                self.corr_heatmap.index = self.columns
            end = self.profiler.timer.time_end("Train Graphical Lasso")
        else:
            if self.param['use_cov']:
                # decomposition
                B1, perm1 = GLassoDetector.decompose(c_cov[1])
                # get tic
                ticks1 = [self.columns[i] for i in perm1]
                self.cov_heatmap = pd.DataFrame(data=B1, columns=ticks1)
                self.cov_heatmap.index = ticks1
            if self.param['use_corr']:
                B2, perm2 = GLassoDetector.decompose(c_corr[1])
                ticks2 = [self.columns[i] for i in perm2]
                self.corr_heatmap = pd.DataFrame(data=B2, columns=ticks2)
                self.corr_heatmap.index = ticks2
            
            end = self.profiler.timer.time_end("Train Graphical Lasso")                
        return end

