from __future__ import division
from profiler.globalvar import *
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
from sklearn import preprocessing
from scipy.cluster.vq import vq, kmeans, whiten
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def normalized_sim(diff, threshold_type=2):
    if threshold_type == 1:
        # standardizing and set all value below or above as zeros
        np_diff = diff.values
        scaler = preprocessing.StandardScaler()
        np_diff = scaler.fit_transform(np_diff.reshape(-1, 1))
        zero_trans = scaler.transform([[0]])
        for i in range(np_diff.shape[1]):
            if zero_trans[i] >= 0:
                np_diff[np_diff[:, i] >= 0] = 0
            else:
                np_diff[np_diff[:, i] < 0] = 0
        ret = pd.Series(data=np.abs(np_diff.flatten()),
                        index=diff.index, name=diff.name)
    elif threshold_type == 2:
        # standardizing and take abs value
        np_diff = diff.values
        scaler = preprocessing.StandardScaler()
        np_diff = scaler.fit_transform(np_diff.reshape(-1, 1))
        ret = pd.Series(data=np.abs(np_diff.flatten()),
                        index=diff.index, name=diff.name)
    elif threshold_type == 3:
        # k-means to seperate and assign zeros to values within the same class of 0
        np_diff = np.abs(diff.values.flatten())
        np_diff = np.nan_to_num(np_diff).astype(np.float64)
        centers, _ = kmeans(np_diff, k_or_guess=2)
        np_diff[np_diff <= np.mean(centers)] = 0
        ret = pd.Series(data=np.abs(np_diff.flatten()),
                        index=diff.index, name=diff.name)
    else:
        ret = 1 - np.abs(diff) / np.nanmax(np.abs(diff))
    return ret


def compute_differences(attr, dtype, env, operators, left, right, embed):
    if dtype == CATEGORICAL:
        df, c = compute_differences_categorical(env, attr, left, right)
    elif dtype == NUMERIC:
        df, c = compute_differences_numerical(
            env, attr, operators, left, right)
    elif dtype == DATE:
        df, c = compute_differences_date(env, attr, operators, left, right)
    elif dtype == TEXT:
        if env['embedtxt']:
            df, c = compute_differences_text(env, attr, left, right, embed)
        else:
            df, c = compute_differences_categorical(env, attr, left, right)
    else:
        df = None

    return df, c


def compute_differences_text(env, attr, left, right, embed, sim_type=1):
    # sim_type = 1: cosine similarity between text
    # sim_type = 2 / else: Euclidean distance between text

    if embed is None:
        raise Exception(
            'ERROR while creating training data. Embedding model is none')
    # handle null
    mask = left[(left == env['null']) | (right == env['null'])].index.values

    df = pd.DataFrame()

    left = embed.get_embedding(left, attr=attr).squeeze()
    right = embed.get_embedding(right, attr=attr).squeeze()

    if sim_type == 1:
        # Cosine similarity
        sim = np.sum(np.multiply(left, right), axis=1) / (np.sqrt(np.sum(np.square(left), axis=1)) *
                                                          np.sqrt(np.sum(np.square(right), axis=1)))
    else:
        # Euclidean distance
        sub = left - right
        sim = np.sqrt(np.sum(np.multiply(sub, sub), axis=1))
        scaler = preprocessing.MinMaxScaler()
        sim = scaler.fit_transform(sim.reshape([len(sim), 1])).flatten()

    if env['continuous']:
        df[attr] = sim
    else:
        df[attr] = (sim >= 1 - env['tol'])*1

    # handle null
    df.iloc[mask, :] = np.zeros((len(mask), df.shape[1]))
    return df, len(mask)


def compute_differences_categorical(env, attr, left, right):
    df = pd.DataFrame()
    mask = left[(left == env['null']) | (right == env['null'])].index.values
    df[attr] = np.equal(left, right)*1

    # if the values are categorially equal to each other
    # handle null
    df.iloc[mask, :] = np.zeros((len(mask), df.shape[1]))
    return df, len(mask)


def compute_differences_numerical(env, attr, operators, left, right):
    diff = left - right  # directly calculate the deduction of left and right
    return compute_differences_numerical_helper(env, attr, operators, left, diff)


def compute_differences_date(env, attr, operators, left, right):
    diff = ((left.values - right.values) /
            np.timedelta64(1, 's')).astype('float')
    return compute_differences_numerical_helper(env, attr, operators, left, diff)


def compute_differences_numerical_helper(env, attr, operators, left, diff):
    df = pd.DataFrame()
    if env['continuous']:
        if len(operators) == 1:
            df[attr] = normalized_sim(diff)
        else:
            df["%s_eq" % attr] = normalized_sim(diff)
        if GT in operators:
            gt = diff.copy()
            gt[diff <= 0] = 0
            df["%s_gt" % attr] = gt
        if LT in operators:
            lt = diff.copy()
            lt[diff >= 0] = 0
            df["%s_lt" % attr] = lt
    else:
        if len(operators) == 1:
            df[attr] = (normalized_sim(diff) >= 1 - env['tol'])*1
        else:
            df["%s_eq" % attr] = (normalized_sim(diff) >= 1 - env['tol'])*1
        if GT in operators:
            df["%s_gt" % attr] = (diff > 0)*1
        if LT in operators:
            df["%s_lt" % attr] = (diff < 0)*1

    # handle null
    mask = left[np.isnan(diff)].index.values
    df.iloc[mask, :] = np.zeros((len(mask), df.shape[1]))
    return df, len(mask)


class TransformEngine(object):
    """
    Transform input data to generate training data
    """

    def __init__(self, env, ds):
        self.env = env
        self.ds = ds
        self.training_data = None
        self.sample_size = -1
        self.null_pb = 0
        self.embed = None
        self.left_idx = None
        self.right_idx = None

    def check_singular(self, df):
        # check singular
        to_drop = [col for col in df if len(df[col].unique()) == 1]
        if len(to_drop) != 0:
            df.drop(to_drop, axis=1, inplace=True)
        self.ds.field = df.columns.values
        return df, np.unique(list(map(lambda x: "_".join(x.split('_')[0:-1]), to_drop))).tolist()

    def estimate_sample_size(self, sample_frac):
        # n > 1/eps^2*logp*(s+p) -> n > 1/eps^2*logp*((p-1)^2/2+p) = 1/eps^2*logp*(p^2/2 + 1/2)
        p = np.sum([len(op) for op in self.ds.operators.values()])
        min_n = 1/np.square(self.env['eps'])*np.log(p)*(np.square(p)/2+0.5)
        multiplier = int(
            np.ceil(min_n / (self.ds.df.shape[0] * self.ds.df.shape[1] * sample_frac)))
        # multiplier
        logger.info("needs multiplier = %d to bound the error in inv cov estimation <= %.8f" % (
            multiplier, self.env['eps']))
        multiplier = min(max(1, multiplier), self.ds.df.shape[0]-1)
        self.env['eps'] = (np.sqrt(np.square(p-1)/2+p) /
                           (multiplier*self.ds.df.shape[0]))
        logger.info("use multiplier = %d, and the bound is %.8f" %
                    (multiplier, self.env['eps']))
        return multiplier

    def create_training_data(self, multiplier=None, sample_frac=1, embed=None, difference=True):

        if not difference:
            data, _ = self.check_singular(self.ds.df)

            # change all data into numerical rather than text or cat
            self.embed = embed
            data = self.transfer_data_into_all_num(data)

            self.null_pb = 0
            self.sample_size = data.shape[0]
            self.training_data = data
            return

        self.embed = embed

        self.handle_nulls()

        multiplier = self.get_multiplier(multiplier, sample_frac)

        logger.info("Draw Pairs")
        left, right, self.left_idx, self.right_idx = self.create_pair_data(
            multiplier, sample_frac)

        logger.info("Computing Differences")
        data_count = self.compute_differences(left, right)
        data = pd.concat([attr[0] for attr in data_count], axis=1)

        # turn data into non-singualr matrix by dropping columns
        self.training_data, drop_cols = self.check_singular(data)

        # obtain count of nulls
        self.compute_null_pb(data_count, drop_cols)

    def compute_differences(self, left, right):
        if self.env['workers'] < 1:
            data_count = [compute_differences(attr, self.ds.dtypes[attr], self.env, self.ds.operators[attr], left[attr],
                                              right[attr], self.embed) for attr in self.ds.field]
        else:
            pool = ThreadPoolExecutor(self.env['workers'])
            data_count = list(pool.map(lambda attr: compute_differences(attr, self.ds.dtypes[attr], self.env,
                                                                        self.ds.operators[attr], left[attr],
                                                                        right[attr], self.embed), self.ds.field))
        return data_count

    def compute_null_pb(self, data_count, drop_cols):
        null_counts = np.sum([attr[1] for i, attr in enumerate(
            data_count) if self.ds.df.columns.values[i] not in drop_cols])
        self.null_pb = null_counts / \
            (self.training_data.shape[0] * len(self.ds.field))
        logger.info(
            "estimated missing data probability in training data is %.4f" % self.null_pb)

    def get_multiplier(self, multiplier, sample_frac):
        if multiplier is None:
            # set min multiplier to 10
            multiplier = max(self.estimate_sample_size(sample_frac), 5)
            logger.info("Using multiplier %d" % multiplier)
        # conservative sample size, did not multiply by number of attributes since there may have repeated samples
        self.sample_size = multiplier * \
            int(np.ceil(self.ds.df.shape[0] *
                        sample_frac / self.ds.df.shape[1]))
        return multiplier

    def handle_nulls(self):
        # handle nulls
        self.ds.replace_null()
        if self.env['null_policy'] == SKIP:
            self.ds.df.dropna(how="any", axis=0, inplace=True)

    def transfer_data_into_all_num(self, data):
        for col in data.columns:
            if self.ds.dtypes[col] in ['categorical', 'text']:
                frac_value = pd.factorize(data[col].values)[
                    0].astype(np.float64)
                frac_value = (np.abs(frac_value)) / \
                    (np.nanmax(np.abs(frac_value)))
                data.update(pd.DataFrame(frac_value, columns=[col]))
                data[col] = data[col].astype(np.float64)
                self.ds.dtypes[col] = 'numeric'
            elif self.ds.dtypes[col] in ['numeric']:
                frac_value = np.abs(data[col].values) / \
                    np.nanmax(np.abs(data[col].values))
                data.update(pd.DataFrame(frac_value, columns=[col]))
                data[col] = data[col].astype(np.float64)
        data = data.reset_index(drop=True)
        data = data.dropna()
        return data

    # Edited on 09/28/2019 by Yunjia
    # added a para for indicating if the training set should be sorted
    # the default attr_sort should be False (random permutation for every attr)
    # If attr_sort=False, should use lower sparsity configuration

    def create_pair_data(self, multiplier, sample_frac, attr_sort=True):
        multiplier = max(1, int(np.ceil(multiplier/self.ds.field.shape[0])))
        # shift and concate
        lefts = []
        rights = []
        # print("sample frac = ",sample_frac)
        for attr in tqdm(self.ds.field):
            if sample_frac == 1:
                if attr_sort:
                    base_table = self.ds.df.sort_values(by=attr)
                else:
                    base_table = self.ds.df.reindex(
                        np.random.permutation(self.ds.df.index))
            else:
                if attr_sort:
                    base_table = self.ds.df.sample(
                        frac=sample_frac).sort_values(by=attr)
                else:
                    base_table = self.ds.df.sample(frac=sample_frac)
                    base_table = base_table.reindex(
                        np.random.permutation(base_table.index))
            left = [base_table] * multiplier
            right = [base_table.iloc[list(range(i+1, base_table.shape[0])) + list(range(i+1)), :].reset_index(
                drop=True) for i in range(multiplier)]
            lefts.extend(left)
            rights.extend(right)
        lefts = pd.concat(lefts)
        left_idx = lefts.index.values
        lefts = lefts.reset_index(drop=True)
        rights = pd.concat(rights)
        right_idx = rights.index.values
        rights = rights.reset_index(drop=True)
        logger.info("Number of training samples: %d" % lefts.shape[0])
        return lefts, rights, left_idx, right_idx
