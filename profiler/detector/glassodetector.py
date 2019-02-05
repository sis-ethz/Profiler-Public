from __future__ import division
from detector import Detector
from sklearn.covariance import graphical_lasso
from sklearn import preprocessing
from scipy.linalg import ldl
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
            'multiplier': 0,
            'sort_training_data': False,
            'error_bound': 1,
            'binary': True,
            'k': 5,
            'sample_frac': 1
        }
        self.param.update(kwargs)
        self.columns = []
        self.heatmap_name = {}
        self.init_name()
        self.data = None

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
            trainData = self.dataEngine.trainData
            df = trainData.get_dataframe()
            data = pd.DataFrame()
            self.columns = np.array(self.dataEngine.field)
            if self.param['undirected']:
                for attr in tqdm(self.columns):
                    if df['left_'+attr].dtype == np.float64 or df['left_'+attr].dtype == np.int64:
                        if self.param['binary']:
                            # numerical
                            #logger.info("Treat %s as numerical"%attr)
                            data['diff_'+attr] = (np.square(df['left_'+attr] - df['right_'+attr]) 
                                                  <= self.param['error_bound'])*2 - 1
                        else:
                            #logger.info("Treat %s as numerical, get differences in %d-ary"%(attr, self.param['k']))
                            from sklearn.cluster import KMeans
                            kmeans = KMeans(n_clusters=self.param['k'])
                            kmeans.fit(np.abs(df['left_'+attr] - df['right_'+attr]).values)
                            data['diff_'+attr] = kmeans.labels_
                    else:
                        # categorical
                        #logger.info("Treat %s as categorical"%attr)
                        data['diff_'+attr] = (df['left_'+attr] == df['right_'+attr])*2 - 1
            else:
                for attr in tqdm(self.columns):
                    if df['left_'+attr].dtype == np.float64 or df['left_'+attr].dtype == np.int64:
                        if self.param['binary']:
                            # numerical
                            # logger.info("Treat %s as numerical, get differences in binary"%attr)
                            data['diff_'+attr] = 1 - (np.abs(df['left_'+attr] - df['right_'+attr]) 
                                                      <= self.param['error_bound'])*1
                        else:
                            # logger.info("Treat %s as numerical, get differences in %d-ary"%(attr, self.param['k']))
                            from sklearn.cluster import KMeans
                            kmeans = KMeans(n_clusters=self.param['k'])
                            kmeans.fit((df['left_'+attr] - df['right_'+attr]).values)
                            data['diff_'+attr] = kmeans.labels_
                    else:
                        # categorical
                        # logger.info("Treat %s as categorical"%attr)
                        data['diff_'+attr] = 1 - (df['left_'+attr] == df['right_'+attr])*1
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
            c_cov = graphical_lasso(m_cov,alpha=self.param['alpha_cov'], mode='cd')
        except Exception as e:
            logger.error(" Error running GLD cov: {}".format(e))
            raise
        try:
            c_corr = graphical_lasso(m_corr,alpha=self.param['alpha_corr'], mode='lars')
        except Exception as e:
            c_corr = None
            logger.error(" Error running GLD corr: {}".format(e))

        if self.param['undirected'] or (not self.param['decompose']):
            self.cov_heatmap = pd.DataFrame(data=c_cov[0], columns=self.columns)
            self.cov_heatmap.index = self.columns
            if c_corr:
                self.corr_heatmap = pd.DataFrame(data=c_corr[0], columns=self.columns)
                self.corr_heatmap.index = self.columns
            end = self.profiler.timer.time_end("Train Graphical Lasso")
        else:
            # decomposition
            B1, perm1 = GLassoDetector.decompose(c_cov[0])
            # get tic
            ticks1 = [self.columns[i] for i in perm1]
            self.cov_heatmap = pd.DataFrame(data=B1, columns=ticks1)
            self.cov_heatmap.index = ticks1
            
            if c_corr:
                B2, perm2 = GLassoDetector.decompose(c_corr[0])
                ticks2 = [self.columns[i] for i in perm2]
                self.corr_heatmap = pd.DataFrame(data=B2, columns=ticks2)
                self.corr_heatmap.index = ticks2
            
            end = self.profiler.timer.time_end("Train Graphical Lasso")                
        return end

