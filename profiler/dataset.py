from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Source(Enum):
    FILE = 1
    DF   = 2
    DB   = 3

class Dataset(object):

    def __init__(self, name, env):
        self.name = name
        self.env = env

    def load_data(self, name, src, fpath, df, **kwargs):
        param = {
            'na_values': {"?", "", "None", "none", "nan", "NaN", "unknown"},
            'sep': ',',
            'header': 'infer',
            'dropcol': None,
            'encoding': 'utf-8',
            'normalize': True,
            'nan': "_empty_",
        }
        param.update(kwargs)

        if src == Source.FILE:
            if fpath is None:
                raise Exception("ERROR while loading table. File path for CSV file name expected. Please provide <fpath> param.")
            self.df = pd.read_csv(fpath, encoding=param['encoding'], header=param['header'], sep=param['sep'],
                                  na_values=param['na_values'])
            # Normalize the dataframe: drop null columns, convert to lowercase strings, and strip whitespaces.
            for attr in self.df.columns.values:
                if self.df[attr].isnull().all():
                    logging.warning("Dropping the following null column from the dataset: '%s'", attr)
                    self.df.drop(labels=[attr], axis=1, inplace=True)
                    continue
        elif src == Source.DF:
            if df is None:
                raise Exception("ERROR while loading table. Dataframe expected. Please provide <df> param.")
            self.df = df
        elif src == Source.DB:
            raise Exception("Not Implemented")


class DataEngine(object):

    def __init__(self):
        pass

    def normalize(self, df):
        




