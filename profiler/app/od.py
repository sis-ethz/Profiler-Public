from abc import ABCMeta, abstractmethod
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from profiler.utility import GlobalTimer
from profiler.data.embedding import OneHotModel
import matplotlib.pyplot as plt
from profiler.globalvar import *
from tqdm import tqdm
import numpy as np
import sklearn
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



class OutlierDetector(object):

    __metaclass__ = ABCMeta
    
    def __init__(self, df, gt_idx=None, method='std', workers=4, t=0.05, tol=1e-6):
        self.timer = GlobalTimer()
        self.method = method
        self.df = df
        self.gt_idx = gt_idx
        self.overall = None
        self.structured = None
        self.combined = None
        self.workers=workers
        self.t = t
        self.tol = tol
        self.structured_info = {}
        self.overall_info = {}

    def get_neighbors(self, left):
        X = self.df[left].values.reshape(-1,len(left))
        # calculate pairwise distance for each attribute
        distances = np.zeros((X.shape[0],X.shape[0]))
        for j in range(X.shape[1]):
            dis = sklearn.metrics.pairwise_distances(X[:,j].reshape(-1,1), 
                                                     metric='cityblock', n_jobs=self.workers)
            # normalize distance
            # avoid divided by zero
            maxdis = max(self.tol, np.nanmax(dis))
            dis = dis / maxdis
            distances = (dis <= self.tol)*1 + distances
        has_same_left = (distances == X.shape[1])
        return has_same_left

    @abstractmethod
    def get_outliers(self, data, right=None):
        # return a mask
        pass

    def run_attr(self, right):
        attr_outliers = self.df.index.values[self.get_outliers(self.df[right], right)]
        self.overall_info[right] = {
            'avg_neighbors': self.df.shape[0],
            'total_outliers': len(attr_outliers),
            'precision': self.compute_precision(outliers=attr_outliers, log=False)[0]
        }
        return attr_outliers

    def run_all(self, parent_sets, separate=True):
        self.run_overall(separate)
        self.run_structured(parent_sets)
        self.run_combined()
        print(self.timer.get_stat())

    def run_overall(self, separate=True):
        self.timer.time_start("naive")
        if separate:
            overall = []
            for attr in self.df:
                overall.extend(list(self.run_attr(attr)))
        else:
            overall = self.run_attr(self.df.columns.values)
        self.overall = overall
        self.timer.time_end("naive")

    def run_attr_structured(self, left, right):
        outliers = []
        if len(left) == 0:
            return outliers
        has_same_neighbors = self.get_neighbors(left)
        num_neighbors = np.zeros((len(has_same_neighbors, )))
        num_outliers = np.zeros((len(has_same_neighbors, )))
        #X = self.df[left].values
        #pbar= tqdm(total=100, leave=True)
        #iter = has_same_neighbors.shape[0] / 100
        for i, row in enumerate(has_same_neighbors):
            # indicies of neighbors
            nbr = self.df.index.values[row == len(left)]
            # has_same = np.sum([(X[:, i] <= row[i] + self.tol) & (X[:, i] >= row[i] - self.tol)
            #                    for i in range(X.shape[1])], axis=0) == 2
            # nbr = self.df.index.values[has_same]
            if len(nbr) == 0:
                continue
            if self.method != "std":
                outlier = nbr[self.get_outliers(self.df.loc[nbr, right], right)]
            else:
                outlier = nbr[self.get_outliers(self.df.loc[nbr, right], right, m='m2')]
            outliers.extend(outlier)

            # save outlier info
            num_neighbors[i] = len(nbr)
            num_outliers[i] = len(outlier)
            # if i % iter == 0 and i != 0:
            #     pbar.update(1)
        # save info
        self.structured_info[right] = {
            'determined_by': left,
            'num_neighbors': num_neighbors,
            'num_outliers': num_outliers,
            'avg_neighbors': np.nanmean(num_neighbors),
            'total_outliers': len(np.unique(outliers))
        }
        return outliers

    def run_structured(self, parent_sets):
        self.timer.time_start("structured")
        structured = []
        for i, child in enumerate(tqdm(parent_sets)):
            outlier = self.run_attr_structured(parent_sets[child], child)
            structured.extend(outlier)
            if child not in self.structured_info:
                continue
            self.structured_info[child]['precision'], _ = self.compute_precision(outlier, log=False)
        unique, count = np.unique(structured, return_counts=True)
        outliers = list(unique[count > self.t*self.df.shape[0]])
        self.structured = outliers
        self.timer.time_end("structured")

    def run_combined(self, parent_sets=None):
        if self.overall is None:
            self.run_overall()
        if self.structured is None:
            self.run_structured(parent_sets)
        combined = list(self.structured)
        combined.extend(self.overall)
        self.combined = combined

    def compute_precision(self, outliers, log=True):
        outliers = set(outliers)
        tp = 0.0
        # precision
        if len(outliers) == 0:
            if len(self.gt_idx) == 0:
                if log:
                    print("no outlier is found and no outlier is present in the ground truth as well, f1 is 1")
                return 1, 0
            if log:
                print("no outlier is found, f1: 0")
            return 0, 0
        for i in outliers:
            if i in self.gt_idx:
                tp += 1
        prec = tp / len(outliers)
        if log:
            print("with %d detected outliers, precision is: %.4f"%(len(outliers), prec))
        return prec, tp

    def compute_f1(self, outliers, title=None, log=True):
        if title is not None:
            print("Results for %s:"%title)
        prec, tp = self.compute_precision(outliers, log=log)
        rec = self.compute_recall(tp, outliers)
        if log:
            print("f1: %.4f"%(2 * (prec * rec) / (prec + rec)))

    def compute_recall(self, tp, outliers):
        if tp == 0:
            print("with %d outliers in gt, recall is: 0"%(len(self.gt_idx)))
            return 0
        if len(self.gt_idx) == 0:
            print("since no outliers in the groud truth, recall is: 1"%(len(self.gt_idx)))
            return 1
        recall = tp / len(self.gt_idx)
        print("with %d detected outliers, recall is: %.4f"%(len(outliers), recall))
        return recall

    def visualize_precision(self, dict, name):
        data = [dict[right]['precision'] for right in dict]
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(data)), data)
        ax.set_xticklabels(list(dict.keys()))
        ax.set_title("[%s] precision for every column"%name)

    def evaluate(self):
        self.compute_f1(self.overall, "naive approach")
        self.compute_f1(self.structured, "structure only")
        self.compute_f1(self.combined, "enhance naive with structured")
        self.visualize_precision(self.structured_info, 'structured')
        self.visualize_precision(self.overall_info, 'overall')

    def view_neighbor_info(self):

        for right in self.structured_info:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            data = self.structured_info[right]['num_neighbors']
            ax1.hist(data, density=True, bins=np.arange(data.min(), data.max()+1))
            ax1.set_title("histogram of num_neighbors\n for column %s"%right)
            width = 0.35
            rects1 = ax2.bar(np.arange(len(data)),self.structured_info[right]['num_neighbors'],width)
            rects2 = ax2.bar(np.arange(len(data))+width,self.structured_info[right]['num_outliers'],width)
            ax2.legend((rects1[0], rects2[0]),['num_neighbors', 'num_outliers'])
            ax2.set_title("num_neighbors and \nnum_outliers\n for column %s"%right)

        fig, ax = plt.subplots()
        width = 0.35
        rects1 = ax.bar(np.arange(len(self.structured_info))+width,
                        [self.overall_info[right]['avg_neighbors'] for right in self.structured_info], width)
        rects2 = ax.bar(np.arange(len(self.structured_info)),
                        [self.structured_info[right]['avg_neighbors'] for right in self.structured_info], width)
        ax.legend((rects1[0], rects2[0]),['overall', 'structured'])
        ax.set_xticklabels(list(self.structured_info.keys()))
        ax.set_title("average number of neighbors for every column")


class STDDetector(OutlierDetector):
    
    def __init__(self, df, gt_idx=None):
        super(STDDetector, self).__init__(df, gt_idx, "std")
        self.param = {
            'm1': 3,
            'm2': 5,
        }

    def get_outliers(self, data, right=None, m='m1'):
        return abs(data - np.nanmean(data)) > self.param[m] * np.nanstd(data)
        # else, categorical, find low frequency items


class SEVERDetector(OutlierDetector):
    def __init__(self, df, gt_idx=None):
        super(SEVERDetector, self).__init__(df, gt_idx, "sever")
        self.param = {
        }
        self.overall = None
        self.structured = None
        self.combined = None

    def get_outliers(self, gradient, right=None):
        size = gradient.shape[0]
        gradient_avg = np.sum(gradient, axis=0)/size
        gradient_avg = np.repeat(gradient_avg.reshape(1, -1), size, axis=0)
        G = gradient - gradient_avg

        decompose = np.linalg.svd(G)
        S = decompose[1]
        V = decompose[2]

        top_right_v = V[np.argmax(S)].T
        score = np.matmul(G, top_right_v)**2

        thred = np.percentile(score, 100-p*100)
        mask = (score < thred)
        #if it is going to remove all, then remove none
        if np.all(~mask):
            return ~mask
        return mask


class ScikitDetector(OutlierDetector):
    def __init__(self, df, method, attr=None, embed=None, gt_idx=None, embed_txt=False,
                 t=0.05, workers=4, tol=1e-6, **kwargs):
        super(ScikitDetector, self).__init__(df, gt_idx, method, t=t, workers=workers, tol=tol)
        self.embed = embed
        self.attributes = attr
        self.embed_txt = embed_txt
        self.overall = None
        self.structured = None
        self.combined = None
        self.algorithm = None
        self.param, self.algorithm = self.get_default_setting()
        self.param.update(kwargs)
        self.encoder = self.create_one_hot_encoder(df)

    def get_default_setting(self):
        if self.method == "isf":
            param = {
                'contamination': 0.1,
            }
            alg = IsolationForest
        elif self.method == "ocsvm":
            param = {
                'nu': 0.5,
                'kernel': "rbf",
                'gamma': 'auto'
            }
            alg = svm.OneClassSVM
        elif self.method == "lof":
            param = {
                'n_neighbors': 100,
                'contamination': 0.1,
            }
            alg = LocalOutlierFactor
        return param, alg

    def create_one_hot_encoder(self, df):
        encoders = {}
        for attr, dtype in self.attributes.items():
            if dtype == CATEGORICAL or (dtype == TEXT and (not self.embed_txt)):
                data = df[attr]
                if not isinstance(data, np.ndarray):
                    data = data.values
                if len(data.shape) == 1:
                    data = data.reshape(-1, 1)
                encoders[attr] = OneHotModel(data)
        return encoders

    def get_neighbors(self, left):
        X = self.df[left].values.reshape(-1,len(left))
        # calculate pairwise distance for each attribute
        distances = np.zeros((X.shape[0],X.shape[0]))
        for j, attr in enumerate(left):
            # validate type and calculate cosine distance
            if self.attributes[attr] == TEXT and self.embed_txt:
                data = self.embed[attr].get_embedding(X[:,j].reshape(-1,1))
                dis = sklearn.metrics.pairwise.cosine_distances(data)
            elif self.attributes[attr] == CATEGORICAL or self.attributes[attr] == TEXT:
                data = self.encoder[attr].get_embedding(X[:,j].reshape(-1,1))
                dis = sklearn.metrics.pairwise.cosine_distances(data)
            else:
                dis = sklearn.metrics.pairwise_distances(X[:,j].reshape(-1,1),
                                                         metric='cityblock', n_jobs=self.workers)
            # normalize distance
            # avoid divided by zero
            maxdis = max(self.tol, np.nanmax(dis))
            dis = dis / maxdis
            distances = (dis <= self.tol)*1 + distances
        has_same_left = (distances == X.shape[1])
        return has_same_left

    def get_outliers(self, data, right=None):
        mask = np.zeros((data.shape[0]))

        if not isinstance(data, np.ndarray):
            data = data.values
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        if self.attributes[right] == TEXT:
            if self.embed_txt:
                # take embedding
                data = self.embed[right].get_embedding(data)
            else:
                data = self.encoder[right].get_embedding(data)
        elif self.attributes[right] == CATEGORICAL:
            # take one hot encoding
            data = self.encoder[right].get_embedding(data)

        # remove nan:
        row_has_nan = np.isnan(data).any(axis=1)
        clean = data[~row_has_nan]
        model = self.algorithm(**self.param)
        y = model.fit_predict(clean)
        mask[~row_has_nan] = y
        mask = mask.astype(int)

        return mask == -1





    
