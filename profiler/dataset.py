from profiler.globalvar import *
import pandas as pd
import numpy as np
import logging, json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dataset(object):

    def __init__(self, name, env):
        self.session_name = name
        self.env = env
        self.null = "_empty_"
        self.df = None
        self.field = []
        self.original_dtypes = None
        self.dtypes = None

    def load_data(self, name, src, fpath, df, **kwargs):
        param = {
            'na_values': {"?", "", "None", "none", "nan", "NaN", "unknown"},
            'sep': ',',
            'header': 'infer',
            'dropcol': None,
            'encoding': 'utf-8',
            'normalize': True,
            'null': "_empty_",
            'min_categories_for_embedding': 10,
        }
        param.update(kwargs)
        setattr(self, 'null', param['null'])
        setattr(self, 'name', name)

        if src == FILE:
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
        elif src == DF:
            if df is None:
                raise Exception("ERROR while loading table. Dataframe expected. Please provide <df> param.")
            self.df = df
        elif src == DB:
            raise Exception("Not Implemented")

        if param['normalize']:
            self.normalize()

        self.infer_column_types(param['min_categories_for_embedding'])

    def normalize(self):
        """
        drop null columns, convert to lowercase strings, and strip whitespaces
        :param df:
        :return:
        """
        # drop empty columns
        self.df.dropna(axis=1, how='all', inplace=True)

        for i, t in enumerate(self.df.dtypes):
            # replace all nans non-numeric data to self.nan, strip whitespaces, and convert to lowercase
            if np.issubdtype(t, np.number):
                continue
            self.df.iloc[:, i] = self.df.iloc[:, i].replace(np.nan, self.null, regex=True).str.strip().str.lower()

    def infer_column_types(self, min_cate):
        types = {}
        self.field = self.df.columns.values
        for i, c in enumerate(self.df.dtypes):
            # test if it is numeric
            if np.issubdtype(c, np.number):
                types[self.field[i]] = NUMERIC
                continue
            # test if it is category with few types
            if self.df.iloc[:, i].unique().shape[0] >= min_cate:
                types[self.field[i]] = TEXT
                continue
            self.df[self.field[i]] = self.df[self.field[i]].astype('str')
            types[self.field[i]] = CATEGORICAL

        logger.info("inferred types of attributes: {}".format(json.dumps(types, indent=4)))
        logger.info("(possible types: %s)" % (", ".join(DATA_TYPES)))
        self.dtypes = types
        self.original_dtypes = types

    def change_dtypes(self, names, types):

        def validate_type(tp):
            if tp not in DATA_TYPES:
                raise ValueError("Invalid Attribute Type")
            return tp

        def validate_name(n):
            if n not in self.dtypes:
                raise ValueError("Invalid Attribute Name")
            return n

        def update(n, tp):
            self.dtypes[validate_name(n)] = validate_type(tp)
            if tp == NUMERIC:
                self.df[n] = pd.to_numeric(self.df[n], errors='coerce')
                logger.info("updated types of {} to 'numeric'".format(n))
            else:
                self.df[n] = self.df[n].astype('str')
                logger.info("updated types of {} to '{}'".format(n, tp))

        if isinstance(names, str):
            update(names, types)
        else:
            if isinstance(types, str):
                for name in names:
                    update(name, types)
            else:
                for name, t in zip(names, types):
                    update(name, t)

    def to_embed(self):
        return [attr for attr in self.dtypes if self.dtypes[attr] == TEXT]
