import pandas as pd
from tqdm import trange, tqdm
from profiler.wordembedding import Embedding
from profiler.globalvar import *
import numpy as np
import json
import math
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataEngine(object):

    def __init__(self, profiler, use_db):
        self.profiler = profiler
        self.use_db = use_db
        self.loaded = False
        self.tableName = ''
        self.numAttr = 0
        self.numTuple = 0
        self.trainData = None
        self.trainAttr = {}
        self.field = []
        self.df = None
        self.embedding = None
        self.param = {}
        self.key = None
        self.left_prefix = 'left_'
        self.right_prefix = 'right_'
        self.dtypes = None
        self.original_dtypes = None

    '''
    LOAD AND PREPROCESS DATA
    '''

    def load_data(self, input_type='file', input_df=None, path=None, **kwargs):
        '''
        if input type is table, print schema of the table by name
        else if input type is file, load data into database and print the schema of the table
        :param input_type: source for data input, default 'file', possible types: 'file', 'table'
        :param path: file name
        :param name: table name
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
        :param embedding_type: "attribute", "tuple", "one-hot", "pre-trained"
        :return:
        '''
        
        param = {
            'use_embedding': False,
            'embedding_type': ATTRIBUTE_EMBEDDING,
            'embedding_file': 'local_embedding.bin',
            'embedding_size': 100,
            'na_values': {"?", "", "None", "none", "nan", "NaN"},
            'sep': ',',
            'header': 'infer',
            'dropna': False,
            'dropcol': None,
            'dropkey': False,
            'fillna': True,
            'left_prefix': 'left_',
            'right_prefix': 'right_',
            'min_categories_for_embedding': 10,
            'nan': "_empty_",
            'workers': 1,
        }
        param.update(kwargs)
        self.param = param
        setattr(self, 'left_prefix', param['left_prefix'])
        setattr(self, 'right_prefix', param['right_prefix'])

        self.key = self.profiler.key
        self.tableName = "t{}".format(self.profiler.ID)

        if input_type == "dataframe":
            self.df = input_df
        elif input_type == "file":
            # setup header:
            if param['header'].lower() == 'none':
                param['header'] = None
            # load data from file
            self.df = pd.read_csv(path, encoding='utf8', header=param['header'], sep=param['sep'],
                                  na_values=param['na_values'])
        elif input_type == 'table':
            if not self.use_db:
                raise Exception('SET TO NOT USING DB')
            self.df = self.profiler.query_df("SELECT * FROM {}".format(self.tableName))
        else:
            raise ValueError("Please specify either \'file\' or \'table\' or \'dataframe\' as a source for data input")
        logger.info("Loaded table".format(self.tableName))

        # preprocessing
        self.df = self.preprocess_df(param['dropna'], param['dropkey'], param['dropcol'])

        # log basic data information
        self.infer_column_types(param['min_categories_for_embedding'])
        self.numTuple = self.df.shape[0]
        logger.info("Table has {} attributes and {} tuples".format(len(self.field), self.numTuple))
        self.loaded = True

    def load_embedding(self):
        if self.param['embedding_type'] not in EMBEDDING_TYPES:
            raise Exception("Embedding Type {} is not supported. "
                            "\nSupported types are: {}".format(self.param['embedding_type'], EMBEDDING_TYPES))
        self.embedding = Embedding(self, self.param['embedding_file'], self.param['embedding_type'],
                                   embedding_size=self.param['embedding_size'])

    def infer_column_types(self, min_cate):
        types = {}
        self.field = list(self.df.columns)
        for i, c in enumerate(self.df.dtypes):
            # test if it is numeric
            if np.issubdtype(c, np.number):
                types[self.field[i]] = "numeric"
                continue
            # test if it is category with few types
            if self.df.iloc[:, i].unique().shape[0] >= min_cate and self.param['use_embedding']:
                types[self.field[i]] = "textual"
                continue
            self.df[self.field[i]] = self.df[self.field[i]].astype('str')
            types[self.field[i]] = "categorical"

        logger.info("inferred types of attributes: {}".format(json.dumps(types, indent=4)))
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

    def preprocess_df(self, dropna, dropkey, dropcol):

        logger.info("Preprocessing Data...")

        # (optional) drop specified columns
        if dropcol is not None:
            self.df = self.df.drop(dropcol, axis=1)

        # (optional) drop column that is key
        if self.key in list(self.df.columns) and dropkey:
            self.df = self.df.drop([self.key], axis=1)

        # (optional) drop rows with empty values
        if dropna:
            self.df.dropna(axis=0, how='any', inplace=True)

        # (optional) replace empty cells in non-numeric columns
        for i, c in enumerate(self.df.dtypes):
            if np.issubdtype(c, np.number):
                continue
            self.df.iloc[:,i] = self.df.iloc[:,i].replace(np.nan, self.param['nan'], regex=True)

        # drop empty columns
        self.df.dropna(axis=1, how='all', inplace=True)

        # drop columns with only one distinct value
        def drop_single_value(df):
            res = df
            for col in df.columns:
                if len(df[df[col].notnull()][col].unique()) == 1:
                    res = res.drop(col, axis=1)
            return res
        #self.df = drop_single_value(self.df)

        # drop columns with all distinct value
        def drop_distinct_value(df):
            res = df
            for col in df.columns:
                if len(df[df[col].notnull()][col].unique()) == df[df[col].notnull()].shape[0]:
                    res = res.drop(col, axis=1)
            return res
        #self.df = drop_distinct_value(self.df)

        return self.df

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
    HELPER METHODS FOR CREATING LOCALLY-TRAINED WORD EMBEDDINGS
    '''
    def get_embedded_columns(self):
        return [f for f in self.field if self.dtypes[f] == TEXT]

    def get_embedding_source(self, embedding_type="attribute"):
        '''
        used for training word embedding
        :return:
        '''
        data = []
        df = self.get_dataframe()
        if embedding_type == "attribute":
            data = {}
            for c in self.get_embedded_columns():
                col = df[c].values
                data[c] = [x.split(" ") for x in col]
        elif embedding_type == "row":
            for rid in range(df.shape[0]):
                row = df.iloc[rid, :].values
                sentence = np.core.defchararray.array(row).encode("utf-8")
                data.append(sentence)
        elif embedding_type == "one-hot":
            for rid in range(df.shape[0]):
                row = df.iloc[rid, :].values
                row = row.tolist()
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
            left_columns.append(self.left_prefix + col)
            right_columns.append(self.right_prefix + col)
            sampled_train_field[col] = (self.left_prefix + col, self.right_prefix + col)
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
            left_columns.append(self.left_prefix + col)
            right_columns.append(self.right_prefix + col)
            sampled_train_field[col] = (self.left_prefix + col,self.right_prefix + col)

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
            if multiplier > 0:
                multiplier = min(int(multiplier), len(columns))
            else:
                multiplier = len(columns)
            logger.info("multiplier: %d" % multiplier)
            prog_bar = tqdm(total=multiplier)
            for i, column in enumerate(columns):
                if self.dtypes[column] != NUMERIC:
                    this_table = table.copy().loc[table[column] != self.param['nan'], :]
                else:
                    this_table = table.copy().loc[np.isnan(table[column].values), :]
                if i+1 > multiplier:
                    break
                if sample_frac != 1:
                    shuffled_table = this_table.sample(frac=sample_frac)
                else:
                    shuffled_table = this_table
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



