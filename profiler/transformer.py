from profiler.globalvar import *
import pandas as pd


class TransformEngine(object):
    """
    Transform input data to generate training data
    """

    def __init__(self, env, ds):
        self.env = env
        self.ds = ds
        # self.left_prefix = 'left_'
        # self.right_prefix = 'right_'

    def create_training_data(self):
        
        pass

    def create_pair_data(self, multiplier=10):
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
