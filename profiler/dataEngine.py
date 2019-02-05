import pandas as pd
from tqdm import trange, tqdm
from collections import defaultdict
import numpy as np
import json
import math
import random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataEngine(object):

    def __init__(self, profiler, use_db):
        self.profiler = profiler
        self.use_db = use_db
        self.loaded = False
        self.embedding = None
        self.tableName = ''
        self.numAttr = 0
        self.numTuple = 0
        self.dtype = None
        self.trainData = None
        self.trainAttr = {}
        self.field = []
        self.df = None
        self.left = ''
        self.right = ''


    '''
    LOAD AND PREPROCESS DATA
    '''

    def load_data(self, input_type='file', input_df=None, name=None, **kwargs):
        '''
        if input type is table, print schema of the table by name
        else if input type is file, load data into database and print the schema of the table
        :param input: source for data input, default 'file', possible types: 'file', 'table'
        :param name: table name or file name
        :param header: if the first row is a header, use 'infer', default None
        :param sep: separator, default ','
        :param na: scalar, str, list-like, or dict, default None
        refers to https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
        :param preprocess:
        :param dropna: drop records contain null
        :param pre_trained: use pre-trained embedding
        :param embedding: name of embedding, default one-hot-encoding
        :param cache: path of downloaded embedding
        :param load_embedding: if set true, will load/train embedding model
        :return:
        '''
        
        param = {
            'pre_trained': False,
            'one_hot': True,
            'embedding': 'embedding.bin',
            'embedding_cache': './embedding_cache',
            'na_values': {"?","", "None", "none", "nan", "NaN"},
            'sep': ',',
            'header': 'infer',
            'dropna': False,
            'dropcol': None,
            'dropkey': False,
            'fillna': False,
            'dtype': None,
            'preprocess': True,
            'left_prefix': 'left_',
            'right_prefix': 'right_',
        }
        param.update(kwargs)
        self.left = param['left_prefix']
        self.right = param['right_prefix']
        if isinstance(param['dtype'], str):
            try:
                param['dtype'] = json.load(open(param['dtype']))
            except:
                logger.warn('Cannot load dtype, set to auto')
                param['dtype'] = None
        self.key = self.profiler.key

        if input_type == "dataframe":
            # config
            self.tableName = name

            # preprocessing
            raw_df = input_df
            if param['preprocess']:
                df = self.preprocess(raw_df, param['dropna'], param['dropkey'], param['dropcol'], param['fillna'])
            else:
                df = raw_df

            if self.use_db:
                # create table in db
                self.profiler.create_table_df(df, self.tableName)

            # save fieldname
            self.field = list(df.columns)

            logger.info("Loaded table {}".format(self.tableName))

        elif input_type == 'file':
            # input type is file, load file into database
            if '/' in name:
                self.tableName = name.split('/')[-1].split('.')[-2]
            else:
                self.tableName = name.split('.')[-2]

            # setup header:
            if param['header'].lower() == 'none':
                param['header'] = None

            # load data from file
            raw_df = pd.read_csv(name, encoding='utf8', header=param['header'], sep=param['sep'],
                                 na_values=param['na_values'], dtype=param['dtype'])
            self.dtype = {}
            for col, t in zip(raw_df.columns.values, raw_df.dtypes.values):
                self.dtype[col] = t

            # prerocess
            if param['preprocess']:
                df = self.preprocess(raw_df, param['dropna'], param['dropkey'], param['dropcol'], param['fillna'])
            else:
                df = raw_df

            # create table in db
            if self.use_db:
                self.profiler.create_table_df(df,self.tableName)
            else:
                self.df = df

            # save fieldname
            self.field = list(df.columns)

            logger.info("Loaded table {}".format(self.tableName))

        elif input_type == 'table':
            if not self.use_db:
                raise Exception('SET TO NOT USING DB')

            # assume data is preprocessed

            # input type is a table in the database, connect to db
            self.tableName = name

            # save field
            df = self.profiler.query_df("SELECT * FROM {}".format(self.tableName))
            self.field = list(df.columns)

            logger.info("Loaded table".format(self.tableName))
        else:
            raise ValueError("Please specify either \'file\' or \'table\' or \'dataframe\' as a source for data input")

        # log basic data information
        if self.use_db:
            self.numTuple = self.profiler.query('SELECT COUNT(*) FROM {}'.format(self.tableName))[0]
        else:
            self.numTuple = df.shape[0]
        logger.info("Table has {} attributes and {} tuples".format(len(self.field), self.numTuple))

        # load embedding
        if param['load_embedding']:
            logger.info("Creating Embedding ...")
            raise Exception("not implemented")

        self.loaded = True

    def preprocess(self, df, dropna, dropkey, dropcol, fillna):

        logger.info("Preprocessing Data...")

        # (optional) drop specified columns
        if dropcol is not None:
            df = df.drop(dropcol, axis=1)

        # (optional) drop column that is key
        if self.key in list(df.columns) and dropkey:
            df = df.drop([self.key], axis=1)

        # (optional) drop rows with empty values
        if dropna:
            df.dropna(axis=0, how='any', inplace=True)

        # (optional) replace empty cells
        if fillna:
            df = df.replace(np.nan, "_empty_", regex=True)

        # drop empty columns
        df.dropna(axis=1, how='all', inplace=True)

        # drop columns with only one distinct value
        def drop_single_value(df):
            res = df
            for col in df.columns:
                if len(df[df[col].notnull()][col].unique()) == 1:
                    res = res.drop(col,axis=1)
            return res
        df = drop_single_value(df)

        # drop columns with all distinct value
        def drop_distinct_value(df):
            res = df
            for col in df.columns:
                if len(df[df[col].notnull()][col].unique()) == df[df[col].notnull()].shape[0]:
                    res = res.drop(col,axis=1)
            return res
        df = drop_distinct_value(df)

        return df

    def get_data(self, mode="row"):
        '''
        used for training word embedding
        :return:
        '''
        data = []
        df = self.get_dataframe()
        if mode == "row":
            for rid in range(df.shape[0]):
                row = df.iloc[rid,:].values
                sentence = u" ".join(np.core.defchararray.array(row).encode("utf-8"))
                data.append(sentence)
        elif mode == "cell":
             for rid in range(df.shape[0]):
                row = df.iloc[rid, :].values
                row = np.core.defchararray.array(row).encode("utf-8").tolist()
                data += row
        return data


    '''
    CREATE TRAINING DATA FOR ATTENTION MODEL
    '''

    def create_training_data_row(self, total_frac=1, multiplier=1, sample_frac=1, overwrite=False):
        '''

        :param total_frac: sample after obtaining all training data
        :param multiplier:
        :param sample_frac: sample for each iteration in the shifting process
        :param overwrite:
        :return:
        '''
        table = self.get_dataframe()
        columns = self.field
        left_columns = []
        right_columns = []
        sampled_train_field = {}
        logger.info("Original data size: {}".format(table.shape))
        for col in columns:
            left_columns.append(self.left + col)
            right_columns.append(self.right + col)
            sampled_train_field[col] = (self.left + col,self.right + col)
        try:
            if overwrite:
                raise Exception("Require Overwriting")
            sampled_train = self.profiler.query_df("select * from {}".format(self.tableName+"_train"))
            assert (len(list(sampled_train.columns))-1 == len(columns)*2)
            logger.info("Loaded Training Data")
        except:
            logger.info("Creating Training Data (multiplier = %f, frac = %f, sample_frac = %f)" % (
                multiplier, total_frac, sample_frac))
            all_merged = []
            if multiplier == 0:
                multiplier = table.shape[0]-1
                logger.info("use all pairs with multiplier = %d" % multiplier)
            elif 0 < multiplier < 1:
                multiplier = int(max(math.floor((table.shape[0]-1)*multiplier), 1))
                logger.info("multiplier = %d" % multiplier)
            else:
                multiplier = int(multiplier)
                logger.info("multiplier = %d" % multiplier)
            # shuffle the table
            base_table = table.sample(frac=1)
            for i in trange(multiplier):
                if sample_frac != 1:
                    base_table = table.sample(frac=sample_frac)
                shifted = base_table.iloc[range(i+1,base_table.shape[0])+range(i+1),:].reset_index(drop=True)
                main = base_table.reset_index(drop=True)
                shifted.columns = right_columns
                main.columns = left_columns
                merged = pd.concat([main, shifted], axis=1, join_axes=[main.index])
                all_merged.append(merged)
            merged = pd.concat(all_merged)
            merged = merged.reset_index(drop=True)
            merged['label'] = 0
            merged.index.name = 'id'
            # if data is still too large, take a sample
            if total_frac != 1:
                sampled_train = merged.sample(frac=total_frac)
                sampled_train = sampled_train.reset_index(drop=True)
            else:
                sampled_train = merged
            logger.info("Training data size: {}".format(sampled_train.shape))
            if self.use_db:
                self.profiler.create_table_df(sampled_train, self.tableName+"_train")
        if self.use_db:
            self.trainData = DataSample(
                self.profiler,
                self,
                self.tableName+"_train",
                list(sampled_train.columns),
                sampled_train_field,
                sampled_train.shape[0]
            )
        else:
            self.trainData = DataSample(
                self.profiler,
                self,
                sampled_train,
                list(sampled_train.columns),
                sampled_train_field,
                sampled_train.shape[0]
            )

    def create_training_data_column(self, total_frac=1, single=False, sample_frac=1, overwrite=False, multiplier=-1):
        '''
        create training data for joint model and single model
        :param sample_frac:
        :param attr_frac:
        :param joint:
        :param balance:
        :param overwrite:
        :return:
        '''
        table = self.get_dataframe()
        columns = self.field
        left_columns = []
        right_columns = []
        sampled_train_field = {}
        for col in columns:
            left_columns.append(self.left + col)
            right_columns.append(self.right + col)
            sampled_train_field[col] = (self.left + col,self.right + col)

        # check if existed
        try:
            if overwrite:
                raise Exception("Require Overwriting")
            sampled_train = self.profiler.query_df("select * from {}".format(self.tableName+"_train"))
            assert (len(list(sampled_train.columns))-1 == len(columns)*2)
            logger.info("Loaded Joint Training Data")
        except:
            logger.info("Creating Joint Training Data (total_frac = %f, sample_frac = %f)" % (total_frac, sample_frac))
            all_merged = []
            if multiplier != -1 and multiplier != 1:
                if multiplier < 1:
                    # fraction
                    multiplier = int(multiplier*len(columns))
            else:
                multiplier = len(columns)
            if isinstance(multiplier, float):
                multiplier = min(int(multiplier), len(columns))
            prog_bar = tqdm(total=multiplier)
            for i, column in enumerate(columns):
                if i+1 > multiplier:
                    break
                if sample_frac != 1:
                    shuffled_table = table.sample(frac=sample_frac)
                else:
                    shuffled_table = table
                sorted_table = shuffled_table.sort_values(by=column)
                shifted = sorted_table.iloc[1:].reset_index(drop=True)
                shifted.columns = right_columns
                main = sorted_table.iloc[:-1].reset_index(drop=True)
                main.columns = left_columns
                merged = pd.concat([main, shifted], axis=1, join_axes=[main.index])
                all_merged.append(merged)
                prog_bar.update(1)
            prog_bar.close()
            merged = pd.concat(all_merged)
            merged = merged.reset_index(drop=True)
            merged['label'] = 0
            merged.index.name = 'id'
            if total_frac != 1:
                sampled_train = merged.sample(frac=total_frac)
                sampled_train = sampled_train.reset_index(drop=True)
            else:
                sampled_train = merged
            if self.use_db:
                self.profiler.create_table_df(sampled_train,self.tableName+"_train")
        logger.info("Training data size: {}".format(sampled_train.shape))
        if self.use_db:
            self.trainData = DataSample(
                self.profiler,
                self,
                self.tableName+"_train",
                list(sampled_train.columns),
                sampled_train_field,
                sampled_train.shape[0]
            )
        else:
            self.trainData = DataSample(
                self.profiler,
                self,
                sampled_train,
                list(sampled_train.columns),
                sampled_train_field,
                sampled_train.shape[0]
            )
        if single:
            return sampled_train_field

    def get_dataframe(self):
        if self.use_db:
            return self.profiler.query_df_with_index("SELECT * FROM {}".format(self.tableName))
        return self.df

    # def get_dataframe_with_key(self):
    #     return self.profiler.query_df("SELECT * FROM {}".format(self.tableName))


class DataSample(object):

    def __init__(self, pf, engine, df, header, field_dic, tuples):
        self.profiler = pf
        self.dataEngine = engine
        self.df = df
        self.canonical_field = self.dataEngine.field
        self.text_fields = field_dic
        self.header = header
        self.numTuples = tuples

    def __len__(self):
        return self.numTuples

    def __getitem__(self, idx):
        if isinstance(self.df, pd.DataFrame):
            return self.df[idx].values
        return self.profiler.query_df("SELECT {} FROM {}".format(idx, self.df)).values

    def get_dataframe(self):
        if isinstance(self.df, pd.DataFrame):
            return self.df
        return self.profiler.query_df_with_index("SELECT * FROM {}".format(self.df))

    def get_column(self, col):
        return self.get_dataframe()[col]



