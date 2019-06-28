from abc import ABCMeta, abstractmethod
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from tqdm import tqdm
from profiler.utility import GlobalTimer
import numpy as np
import sklearn
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



class OutlierDetector(object):

    __metaclass__ = ABCMeta
    
    def __init__(self, df, gt_idx=None, method='std', workers=4, t=0.05):
        self.timer = GlobalTimer()
        self.method = method
        self.df = df
        self.gt_idx = gt_idx
        self.overall = None
        self.structured = None
        self.combined = None
        self.workers=workers
        self.t = t

    def get_neighbors(self, left):
        X = self.df[left].values.reshape(-1,len(left))
        # calculate pairwise distance for each attribute
        distances = np.zeros((X.shape[0],X.shape[0]))
        for j in range(X.shape[1]):
            dis = sklearn.metrics.pairwise_distances(X[:,j].reshape(-1,1), metric='cityblock', n_jobs=self.workers)
            # normalize distance
            # avoid divided by zero
            maxdis = max(1e-6, np.nanmax(dis))
            dis = dis / maxdis
            distances = (dis <= 1e-6)*1 + distances
        has_same_left = (distances == X.shape[1])
        return has_same_left

    @abstractmethod
    def get_outliers(self, data):
        # return a mask
        pass

    def run_attr(self, right):
        attr_outliers = self.df.index.values[self.get_outliers(self.df[right])]
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
        #X = self.df[left].values
        #pbar= tqdm(total=100, leave=True)
        iter = has_same_neighbors.shape[0] / 100
        for i, row in enumerate(has_same_neighbors):
            # indicies of neighbors
            nbr = self.df.index.values[row == len(left)]
            # has_same = np.sum([(X[:, i] <= row[i] + 1e-6) & (X[:, i] >= row[i] - 1e-6)
            #                    for i in range(X.shape[1])], axis=0) == 2
            # nbr = self.df.index.values[has_same]
            if len(nbr) == 0:
                continue
            if self.method != "std":
                outliers.extend(nbr[self.get_outliers(self.df.loc[nbr, right])])
            else:
                outliers.extend(nbr[self.get_outliers(self.df.loc[nbr, right], m='m2')])
            # if i % iter == 0 and i != 0:
            #     pbar.update(1)

        return outliers

    def run_structured(self, parent_sets):
        self.timer.time_start("structured")
        structured = []
        for child in tqdm(parent_sets):
            structured.extend(self.run_attr_structured(parent_sets[child], child))
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

    def compute_pr(self, outliers, title=None):
        if title is not None:
            print("Results for %s:"%title)
        outliers = set(outliers)
        tp = 0.0
        # precision
        if len(outliers) == 0:
            if len(self.gt_idx) == 0:
                print("no outlier is found and no outlier is present in the groud truth as well, f1 is 1")
                return
            print("no outlier is found, f1: 0")
            return
        for i in outliers:
            if i in self.gt_idx:
                tp += 1
        prec = tp / len(outliers)
        print("with %d detected outliers, precision is: %.4f"%(len(outliers), prec))
        rec = self.compute_recall(tp, outliers)
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

    def evaluate(self):
        self.compute_pr(self.overall, "naive approach")
        self.compute_pr(self.structured, "structure only")
        self.compute_pr(self.combined, "enhance naive with structured")


class STDDetector(OutlierDetector):
    
    def __init__(self, df, gt_idx=None):
        super(STDDetector, self).__init__(df, gt_idx, "std")
        self.param = {
            'm1': 3,
            'm2': 5,
        }

    def get_outliers(self, data, m='m1'):
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

    def get_outliers(self, gradient):
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


class PyDetector(OutlierDetector):
    def __init__(self, df, method, gt_idx=None, t=0.05):
        super(PyDetector, self).__init__(df, gt_idx, method, t)
        self.overall = None
        self.structured = None
        self.combined = None
        self.algorithm = None
        self.param = {}


    def get_outliers(self, data):
        mask = np.zeros((data.shape[0]))
        if not isinstance(data, np.ndarray):
            data = data.values
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        # remove nan:
        row_has_nan = np.isnan(data).any(axis=1)
        clean = data[~row_has_nan]
        model = self.algorithm(**self.param)
        y = model.fit_predict(clean)
        mask[~row_has_nan] = y
        mask = mask.astype(int)

        return mask == -1


class ISFDetector(PyDetector):
    def __init__(self, df, gt_idx=None, t=0.05, **kwargs):
        super(ISFDetector, self).__init__(df, "isf", gt_idx, t=t)
        self.param = {
            'contamination': 0.1,
        }
        self.overall = None
        self.structured = None
        self.combined = None
        self.param.update(kwargs)
        self.algorithm = IsolationForest


class OCSVMDetector(PyDetector):
    def __init__(self, df, gt_idx=None, t=0.05, **kwargs):
        super(OCSVMDetector, self).__init__(df, "ocsvm", gt_idx, t=t)
        self.param = {
            'nu': 0.1,
            'kernel': "rbf",
            'gamma': 0.1
        }
        self.overall = None
        self.structured = None
        self.combined = None
        self.param.update(kwargs)
        self.algorithm = svm.OneClassSVM


class LOFDetector(PyDetector):
    def __init__(self, df, gt_idx=None, t=0.05, **kwargs):
        super(LOFDetector, self).__init__(df, "lof", gt_idx, t=t)
        self.param = {
            'n_neighbors': 100,
            'contamination': 0.1,
        }
        self.overall = None
        self.structured = None
        self.combined = None
        self.param.update(kwargs)
        self.algorithm = LocalOutlierFactor





    
