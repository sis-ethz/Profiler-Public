from profiler.globalvar import *
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np


def compute_differences(attr, dtype, env, operators, left, right, embed):
    if dtype == CATEGORICAL:
        df = compute_differences_categorical(attr, operators, left, right)
    elif dtype == NUMERIC:
        df = compute_differences_numerical(env, attr, operators, left, right)
    elif dtype == TEXT:
        if env['embedtxt']:
            df = compute_differences_text(env, attr, operators, left, right, embed)
        else:
            df = compute_differences_categorical(attr, operators, left, right)
    else:
        df = None
    if env['null_policy'] == NULL_EQ:
        mask = np.isnan(left) | np.isnan(right)
        df[mask, :] = np.ones((1, df.shape[1]))
    return df


def compute_differences_text(env, attr, operators, left, right, embed):
    if embed is None:
        raise Exception('ERROR while creating training data. Embedding model is none')
    df = pd.DataFrame()
    diff = 1 - np.sum(left * right, axis=1) / (np.sqrt(np.sum(np.square(left), axis=1)) *
                                               np.sqrt(np.sum(np.square(right), axis=1)))
    df["%s_eq" % attr] = (diff / np.nanmax(diff)) <= env['tol']
    if NEQ in operators:
        df["%s_neq" % attr] = 1 - df["%s_eq" % attr]
    return df


def compute_differences_categorical(attr, operators, left, right):
    df = pd.DataFrame()
    df["%s_eq" % attr] = np.equal(left, right)*1
    if NEQ in operators:
        df["%s_neq" % attr] = 1 - df["%s_eq"%attr]
    return df


def compute_differences_numerical(env, attr, operators, left, right):
    df = pd.DataFrame()
    diff = np.abs(left - right)
    df["%s_eq" % attr] = (diff / np.nanmax(diff)) <= env['tol']
    if NEQ in operators:
        df["%s_neq" % attr] = 1 - df["%s_eq"%attr]
    if GT in operators:
        df["%s_gt" % attr] = (left > right)*1
    if LT in operators:
        df["%s_lt" % attr] = (left < right)*1
    return df


class TransformEngine(object):
    """
    Transform input data to generate training data
    """

    def __init__(self, env, ds):
        self.env = env
        self.ds = ds
        self.embed = None
        # self.left_prefix = 'left_'
        # self.right_prefix = 'right_'

    @staticmethod
    def check_singular(df):
        # check singular
        to_drop = [col for col in df if len(df[col].unique()) == 1]
        if len(to_drop) != 0:
            df.drop(to_drop, axis=1, inplace=True)
        return df

    def create_training_data(self, multiplier=10, embed=None):
        self.embed = embed
        left, right = self.create_pair_data(multiplier=multiplier)
        if self.env['workers'] < 1:
            data = [compute_differences(attr, self.ds.dtypes[attr], self.env, self.ds.operators[attr], left[attr],
                                        right[attr], self.embed) for attr in self.ds.field]
        else:
            pool = ThreadPoolExecutor(self.env['workers'])
            data = list(pool.map(lambda attr: compute_differences(attr, self.ds.dtypes[attr], self.env,
                                                                  self.ds.operators[attr], left[attr],
                                                                  right[attr], self.embed), self.ds.field))
        return TransformEngine.check_singular(pd.concat(data, axis=1))

    def create_pair_data(self, multiplier):
        # TODO: allow different null policy
        if self.env['null_policy'] == SKIP:
            self.ds.df.dropna(how="any", axis=0, inplace=True)

        # shift and concate
        base_table = self.ds.df.sample(frac=1)
        left = [base_table] * multiplier
        right = [base_table.iloc[list(range(i+1, base_table.shape[0])) + list(range(i+1)), :].reset_index(
            drop=True) for i in range(multiplier)]
        left = pd.concat(left).reset_index(drop=True)
        right = pd.concat(right).reset_index(drop=True)
        return left, right
        # TODO: show how many data points are enough, i.e. how to set multiplier
