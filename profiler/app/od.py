from abc import ABCMeta, abstractmethod
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from profiler.utility import GlobalTimer
from profiler.data.embedding import OneHotModel
import matplotlib.pyplot as plt
from profiler.globalvar import *
from sklearn.neighbors import BallTree
from tqdm import tqdm
import numpy as np
import sklearn
import warnings, logging


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OutlierDetector(object):

    __metaclass__ = ABCMeta
    
    def __init__(self, df, gt_idx=None, method='std', workers=4, t=0.05, tol=1e-6, neighbor_size=100,
                 knn=False, high_dim=False):
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
        self.eval = {}
        self.neighbors = {}
        self.neighbor_size = neighbor_size
        if knn:
            if not high_dim:
                self.get_neighbors = self.get_neighbors_knn
            else:
                self.get_neighbors = self.get_neighbors_knn_highdim
        else:
            self.get_neighbors = self.get_neighbors_threshold

    def get_neighbors_threshold(self, left):
        X = self.df[left].values.reshape(-1, len(left))
        # calculate pairwise distance for each attribute
        distances = np.zeros((X.shape[0],X.shape[0]))
        for j, attr in enumerate(left):
            # check if saved
            if attr in self.neighbors:
                distances = self.neighbors[attr] + distances
                continue
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
                maxdis = max(self.tol, np.nanmax(dis))
                dis = dis / maxdis
            self.neighbors[attr] = (dis <= self.tol)*1
            distances = self.neighbors[attr] + distances
        has_same_left = (distances == X.shape[1])
        return has_same_left

    def get_neighbors_knn(self, left):
        X = self.df[left].values.reshape(-1, len(left))
        # calculate pairwise distance for each attribute
        distances = np.zeros((X.shape[0],X.shape[0]))
        for j, attr in enumerate(left):
            # check if saved
            if attr in self.neighbors:
                distances = self.neighbors[attr] + distances
                continue
            # validate type and calculate cosine distance
            if self.attributes[attr] == TEXT and self.embed_txt:
                data = self.embed[attr].get_embedding(X[:,j].reshape(-1,1))
                # normalize each vector to take cosine distance
                data = data / np.linalg.norm(data, axis=1)
            elif self.attributes[attr] == CATEGORICAL or self.attributes[attr] == TEXT:
                data = self.encoder[attr].get_embedding(X[:,j].reshape(-1,1))
            else:
                data = X[:,j].reshape(-1,1)
            kdt = BallTree(data, metric='euclidean')
            # find knn
            indicies = kdt.query(data, k=self.neighbor_size, return_distance=False)
            self.neighbors[attr] = np.zeros((X.shape[0],X.shape[0]))
            for i in range(len(indicies)):
                self.neighbors[attr][i, indicies[i, :]] = 1
            distances = self.neighbors[attr] + distances
        has_same_left = (distances == X.shape[1])
        return has_same_left

    def get_neighbors_knn_highdim(self, left):
        X = self.df[left].values.reshape(-1, len(left))
        # calculate pairwise distance for each attribute
        distances = np.zeros((X.shape[0],X.shape[0]))
        data = []
        for j, attr in enumerate(left):
            # check if saved
            if attr in self.neighbors:
                data.append(self.neighbors[attr])
                continue
            # validate type and calculate cosine distance
            if self.attributes[attr] == TEXT and self.embed_txt:
                embedded = self.embed[attr].get_embedding(X[:,j].reshape(-1,1))
                # normalize each vector to take cosine distance
                data.append(embedded / np.linalg.norm(embedded, axis=1))
            elif self.attributes[attr] == CATEGORICAL or self.attributes[attr] == TEXT:
                embedded = self.encoder[attr].get_embedding(X[:,j].reshape(-1,1))
                data.append(embedded)
            else:
                data.append(X[:,j].reshape(-1,1))
            self.neighbors[attr] = data[-1]
        data = np.hstack(data)
        if data.shape[0] != X.shape[0]:
            print(data.shape)
            raise Exception
        kdt = BallTree(data, metric='euclidean')
        # find knn
        indicies = kdt.query(data, k=self.neighbor_size, return_distance=False)
        for i in range(len(indicies)):
            distances[i, indicies[i, :]] = 1
        has_same_left = (distances == 1)
        return has_same_left

    @abstractmethod
    def get_outliers(self, data, right=None):
        # return a mask
        pass

    def run_attr(self, right):
        attr_outliers = self.df.index.values[self.get_outliers(self.df[right], right)]
        prec, tp = self.compute_precision(outliers=attr_outliers, log=False)
        self.overall_info[right] = {
            'avg_neighbors': self.df.shape[0],
            'total_outliers': len(attr_outliers),
            'precision': prec,
            'recall': self.compute_recall(tp, outliers=attr_outliers, log=False)
        }
        return attr_outliers

    def run_all(self, parent_sets, separate=True):
        self.run_overall(separate)
        self.run_structured(parent_sets)
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
        return self.timer.time_end("naive")

    def run_attr_structured(self, left, right):
        outliers = []
        if len(left) == 0:
            return outliers
        has_same_neighbors = self.get_neighbors(left)
        num_neighbors = np.zeros((len(has_same_neighbors, )))
        num_outliers = np.zeros((len(has_same_neighbors, )))
        for i, row in enumerate(has_same_neighbors):
            # indicies of neighbors
            nbr = self.df.index.values[row]
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
            prec, tp = self.compute_precision(outlier, log=False)
            self.structured_info[child]['precision'] = prec
            self.structured_info[child]['recall'] = self.compute_recall(tp, outliers=outlier, log=False)
        self.structured = structured
        return self.timer.time_end("structured")

    def filter(self, structured, t=None):
        if t is None:
            t = self.t
        unique, count = np.unique(structured, return_counts=True)
        outliers = list(unique[count > t*self.df.shape[0]])
        return outliers

    def run_combined(self, structured):
        combined = list(structured)
        combined.extend(self.overall)
        return combined

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
        rec = self.compute_recall(tp, outliers, log=log)
        if rec*prec == 0:
            f1 = 0
        else:
            f1 = 2 * (prec * rec) / (prec + rec)
        if log:
            print("f1: %.4f" % f1)
        return "%.4f,%.4f,%.4f"%(prec, rec, f1)

    def compute_recall(self, tp, outliers, log=True):
        if tp == 0:
            if log:
                print("with %d outliers in gt, recall is: 0"%(len(self.gt_idx)))
            return 0
        if len(self.gt_idx) == 0:
            if log:
                print("since no outliers in the groud truth, recall is: 1"%(len(self.gt_idx)))
            return 1
        recall = tp / len(self.gt_idx)
        if log:
            print("with %d detected outliers, recall is: %.4f"%(len(outliers), recall))
        return recall

    def visualize_stat(self, dict, name, stat='precision'):
        data = [dict[right][stat] if right in dict else 0 for right in self.overall_info]
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(data)), data)
        ax.set_xticks(np.arange(len(data)))
        ax.set_yticks(np.arange(0,1,0.1))
        for i, v in enumerate(data):
            ax.text(i - 0.25, v + .03, "%.2f"%v)
        ax.set_xticklabels(list(self.overall_info.keys()))
        ax.set_xlabel('Column Name')
        ax.set_ylabel(stat)
        ax.set_title("[%s] %s for every column"%(name, stat))

    def evaluate(self, t=None, log=True):
        structured = self.filter(self.structured, t)
        self.eval['overall'] = self.compute_f1(self.overall, "naive approach")
        self.eval['structured'] = self.compute_f1(structured, "structure only")
        self.eval['combined'] = self.compute_f1(self.run_combined(structured), "enhance naive with structured")
        if log:
            self.visualize_stat(self.overall_info, 'overall', stat='precision')
            self.visualize_stat(self.structured_info, 'structured', stat='precision')
            self.visualize_stat(self.overall_info, 'overall', stat='recall')
            self.visualize_stat(self.structured_info, 'structured', stat='recall')

    def evaluate_structured(self, t):
        structured = self.filter(self.structured, t)
        self.eval['structured'] = self.compute_f1(structured, "structure only", log=False)
        self.eval['combined'] = self.compute_f1(self.run_combined(structured),
                                                "enhance naive with structured",
                                                log=False)

    def evaluate_overall(self):
        self.eval['overall'] = self.compute_f1(self.overall, "naive approach", log=False)

    def view_neighbor_info(self):
        for right in self.structured_info:
            fig, (ax1, ax2) = plt.subplots(1,2)
            data = self.structured_info[right]['num_neighbors']
            ax1.hist(data, bins=np.arange(data.min(), data.max()+1))
            ax1.set_title("histogram of num_neighbors\n for column %s"%right)
            ax1.set_xlabel('number of neighbors')
            ax1.set_ylabel('count')
            width = 0.35
            rects1 = ax2.bar(np.arange(len(data)),self.structured_info[right]['num_neighbors'],width)
            rects2 = ax2.bar(np.arange(len(data))+width,self.structured_info[right]['num_outliers'],width)
            ax2.legend((rects1[0], rects2[0]),['num_neighbors', 'num_outliers'])
            ax2.set_title("num_neighbors and \nnum_outliers\n for column %s"%right)
            ax2.set_xlabel('index of tuple')
            ax2.set_ylabel('count')

        fig, ax = plt.subplots()
        width = 0.35
        rects1 = ax.bar(np.arange(len(self.overall_info))+width,
                        [self.overall_info[right]['avg_neighbors'] for right in self.overall_info], width)
        rects2 = ax.bar(np.arange(len(self.overall_info)),
                        [self.structured_info[right]['avg_neighbors'] if right in self.structured_info else 0
                         for right in self.overall_info], width)
        ax.legend((rects1[0], rects2[0]),['overall', 'structured'])
        ax.set_xticks(np.arange(len(self.overall_info)))
        ax.set_xticklabels(list(self.overall_info.keys()))
        ax.set_title("average number of neighbors for every column")
        ax.set_xlabel('column name')
        ax.set_ylabel('count')


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
                 t=0.05, workers=4, tol=1e-6, min_neighbors=50, neighbor_size=100,
                 knn=False, high_dim=False, **kwargs):
        super(ScikitDetector, self).__init__(df, gt_idx, method, t=t, workers=workers, tol=tol,
                                             neighbor_size=neighbor_size, knn=knn, high_dim=high_dim)
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
        self.min_neighbors = min_neighbors

    def get_default_setting(self):
        if self.method == "isf":
            param = {
                'contamination': 0.1,
                'n_jobs': self.workers
            }
            alg = IsolationForest
        elif self.method == "ocsvm":
            param = {
                'nu': 0.1,
                'kernel': "rbf",
                'gamma': 'auto'
            }
            alg = svm.OneClassSVM
        elif self.method == "lof":
            param = {
                'n_neighbors': int(max(self.neighbor_size / 2, 2)),
                'contamination': 0.1,
            }
            alg = LocalOutlierFactor
        elif self.method == "ee":
            param = {
                'contamination': 0.1,
            }
            alg = EllipticEnvelope
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
        if len(clean) <= self.min_neighbors:
            return mask == -1
        y = model.fit_predict(clean)
        mask[~row_has_nan] = y
        mask = mask.astype(int)

        return mask == -1
