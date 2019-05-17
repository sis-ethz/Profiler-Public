from profiler.globalvar import *


class TransformEngine(object):
    """
    Transform input data to generate training data
    """

    def __init__(self, env, ds):
        self.env = env
        self.ds = ds

    def create_training_data(self):
        pass

    def create_pair_data(self):
        # TODO: allow different null policy
        df = self.ds.df
        if self.env['null_policy'] == SKIP:
            df.dropna(how="any", axis=0, inplace=True)
        pass
