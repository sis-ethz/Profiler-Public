from profiler.globalvar import *
import pandas as pd
import numpy as np
import logging
import json

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
        self.operators = {}

    def load_data(self, name, src, fpath, df, check_param, **kwargs):
        param = {
            'na_values': {"?", "", "None", "none", "nan", "NaN", "unknown"},
            'sep': ',',
            'header': 'infer',
            'dropcol': None,
            'dropna': False,
            'encoding': 'utf-8',
            'normalize': True,
            'min_categories_for_embedding': 10,
        }
        param.update(kwargs)
        setattr(self, 'name', name)

        if check_param:
            logger.info("parameters used for data loading:\n {}".format(param))

        if src == FILE:
            if fpath is None:
                raise Exception(
                    "ERROR while loading table. File path for CSV file name expected. Please provide <fpath> param.")
            # print("In dataset: Reading: ", fpath)
            self.df = pd.read_csv(fpath, encoding=param['encoding'], header=param['header'], sep=param['sep'],
                                  na_values=param['na_values'], engine='python')
            # Normalize the dataframe: drop null columns, convert to lowercase strings, and strip whitespaces.
            for attr in self.df.columns.values:
                if self.df[attr].isnull().all():
                    logger.warning(
                        "Dropping the following null column from the dataset: '%s'", attr)
                    self.df.drop(labels=[attr], axis=1, inplace=True)
                    continue
        elif src == DF:
            if df is None:
                raise Exception(
                    "ERROR while loading table. Dataframe expected. Please provide <df> param.")
            self.df = df
        elif src == DB:
            raise Exception("Not Implemented")

        if param['normalize']:
            self.normalize(param['dropcol'], param['dropna'])

        self.infer_column_types(param['min_categories_for_embedding'])

    def normalize(self, dropcol, dropna):
        """
        drop null columns, convert to lowercase strings, and strip whitespaces
        :param df:
        :return:
        """
        # drop empty columns
        self.df.dropna(axis=1, how='all', inplace=True)

        # optional drop columns
        if dropcol is not None:
            self.df.drop(dropcol, axis=1, inplace=True)

        # optional drop null records
        if dropna:
            self.df.dropna(axis=0, how='any', inplace=True)

        for i, t in enumerate(self.df.dtypes):
            # strip whitespaces, and convert to lowercase
            if np.issubdtype(t, np.number) or np.issubdtype(t, np.datetime64):
                continue
            self.df.iloc[:, i] = self.df.iloc[:, i].astype(
                str).str.strip().str.lower()

    def replace_null(self, attr=None):
        def replace_null_helper(attr):
            # replace all nans non-numeric data to same value
            if self.dtypes[attr] not in [DATE, NUMERIC]:
                self.df[attr].replace(np.nan, self.env['null'], regex=True)
        if attr:
            replace_null_helper(attr)
        else:
            for attr in self.field:
                replace_null_helper(attr)

    def infer_column_types(self, min_cate):
        self.dtypes = {}
        self.field = self.df.columns.values
        for i, c in enumerate(self.df.dtypes):
            # test if it is numeric
            if np.issubdtype(c, np.number):
                self.dtypes[self.field[i]] = NUMERIC
                self.infer_operator(self.field[i])
                continue
            # test if it is category with few types
            if self.df.iloc[:, i].unique().shape[0] >= min_cate:
                self.df[self.field[i]] = self.df[self.field[i]].astype('str')
                self.dtypes[self.field[i]] = TEXT
                self.infer_operator(self.field[i])
                continue
            self.df[self.field[i]] = self.df[self.field[i]].astype('str')
            self.dtypes[self.field[i]] = CATEGORICAL
            self.infer_operator(self.field[i])

        logger.info("inferred types of attributes: {}".format(
            json.dumps(self.dtypes, indent=4)))
        logger.info("(possible types: %s)" % (", ".join(DATA_TYPES)))
        self.original_dtypes = self.dtypes
        logger.info(
            "inferred operators of attributes: {}".format(self.operators))
        logger.info("(possible operators: %s)" % (", ".join(OPERATORS)))

    def change_dtypes(self, names, types, regexs):

        def validate_type(tp):
            if tp not in DATA_TYPES:
                raise ValueError("Invalid Attribute Type")
            return tp

        def validate_name(n):
            if n not in self.dtypes:
                raise ValueError("Invalid Attribute Name")
            return n

        def update(n, tp, regex):
            self.dtypes[validate_name(n)] = validate_type(tp)
            if regex:
                df = self.df[n].str.extract(regex, expand=False)
            else:
                df = self.df[n]
            if tp == NUMERIC:
                self.df[n] = pd.to_numeric(df, errors='coerce')
                logger.info("updated types of {} to 'numeric'".format(n))
            elif tp == DATE:
                self.df[n] = pd.to_datetime(df, errors='coerce')
            else:
                self.df[n] = df.astype('str')
                logger.info("updated types of {} to '{}'".format(n, tp))
            self.infer_operator(n)
            logger.info("updated operators of {} to {}".format(
                n, self.operators[n]))

        if isinstance(names, str):
            update(names, types, regexs)
        else:
            for name, t, regex in zip(names, types, regexs):
                update(name, t, regex)
        logger.info(
            "updated inferred operators of attributes: {}".format(self.operators))

    def infer_operator(self, attr):
        if (self.dtypes[attr] in [NUMERIC, DATE]) and self.env['inequality']:
            # operators[attr] = [EQ, NEQ, GT, LT]
            self.operators[attr] = [EQ, GT, LT]
        else:
            # operators[attr] = [EQ, NEQ]
            self.operators[attr] = [EQ]

    def change_operators(self, names, operators):

        def validate_op(x):
            if isinstance(x, str):
                x = [x]
            for op in x:
                if op not in OPERATORS:
                    raise ValueError("Invalid Operator: %s" % op)
            return x

        def validate_name(n):
            if n not in self.operators:
                raise ValueError("Invalid Attribute Name")
            return n

        def update(n, op):
            self.operators[validate_name(n)] = validate_op(op)
            logger.info("updated operators of {} to {}".format(n, op))

        if isinstance(names, str):
            update(names, operators)
        else:
            assert(len(names) == len(operators))
            for name, ops in zip(names, operators):
                update(name, ops)

    def to_embed(self):
        return [attr for attr in self.dtypes if self.dtypes[attr] == TEXT]
