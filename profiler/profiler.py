from detector.glassodetector import GLassoDetector
from sqlalchemy import create_engine
from dataEngine import DataEngine
from utility import GlobalTimer
import pandas as pd
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Profiler(object):
    def __init__(self, use_db=True, db='profiler', host='localhost', 
                 user="profileruser", password="abcd1234", port=5432):
        '''
        constructor for profiler, create a database or connect to database if existed
        create a data engine object
        :param db: database name, default None
        '''
        self.use_db = use_db
        if use_db:
            self.connect_db(db, host, user, password, port)
        self.dataEngine = DataEngine(self, use_db)

        # init params
        self.key = None
        self.heatmap = None
        self.heatmap_name = None
        self.heatmap_csv_name = None
        self.detector = None

        # start timer
        self.timer = GlobalTimer()
        self.timer.time_point("START")

    def load_data(self,
                  key='id',
                  # load data
                  input_type='file',
                  input_df=None,
                  name=None,
                  **kwargs
                  ):
        # load data
        self.key = key
        self.timer.time_start("Preprocess Data and Create Embedding")
        self.dataEngine.load_data(input_type=input_type, name=name, input_df=input_df, **kwargs)
        self.timer.time_end("Preprocess Data and Create Embedding")
        
    # def run_mix_attention(self,
    #         # training data
    #         overwrite=False,
    #         # attention model
    #         frac=1,
    #         model="joint",
    #         device="cpu",
    #         save_heatmap="csv",
    #         hm_path="./",
    #         ID="",
    #         sparse=False,
    #         **kwargs):
    #     # save heatmap
    #     if 'factor' in kwargs.keys() and sparse:
    #         l = str(kwargs['factor']).replace(".","dot")
    #         self.heatmap_name = "{}_{}_{}_osc{}_heatmap".format(ID, self.dataEngine.tableName, model, l)
    #     else:
    #         self.heatmap_name = "{}_{}_{}_heatmap".format(ID, self.dataEngine.tableName, model)
    #
    #     # train attention model
    #     self.attention = MixAttentionDetector(self, frac=frac, mode=model, device=device, overwrite=overwrite,
    #                                           sparse=sparse, **kwargs)
    #     time = self.attention.run()
    #
    #     # save heatmap
    #     self.heatmap = self.attention.heatmap
    #     self.heatmap_csv_name = "{}.csv".format(hm_path+self.heatmap_name)
    #     self.save_heatmap(save_heatmap=save_heatmap)
    #
    #     self.timer.time_point("DONE")
    #     return time

    # def run_attention(self, save_heatmap="csv", hm_path="./", ID="", sparse=False, **kwargs):
    #     '''
    #     method for run attention model with mi estimator as an option
    #     # attention model
    #     :param attention_only:
    #     :param frac:
    #     :param model:[joint, single, vanilla]
    #     :param device:
    #     :param drop:
    #     :param epochs:
    #     :param visual:
    #     :param save_heatmap:
    #     :param ID: identifier for heatmap
    #
    #     :return:
    #     '''
    #     # save heatmap
    #     if 'factor' in kwargs.keys() and sparse:
    #         l = str(kwargs['factor']).replace(".","dot")
    #         self.heatmap_name = "{}_{}_{}_osc{}_heatmap".format(ID, self.dataEngine.tableName, model, l)
    #     else:
    #         self.heatmap_name = "{}_{}_{}_heatmap".format(ID, self.dataEngine.tableName, model)
    #
    #     # train attention model
    #     self.attention = AttentionDetector(self, sparse=sparse, **kwargs)
    #     time = self.attention.run()
    #
    #     # save heatmap
    #     self.heatmap = self.attention.heatmap
    #     self.heatmap_csv_name = "{}.csv".format(hm_path+self.heatmap_name)
    #     self.save_heatmap(save_heatmap=save_heatmap)
    #
    #     self.timer.time_point("DONE")
    #     return time

    def run_graphical_lasso(self, save_heatmap="csv", ID="", hm_path="./", **kwargs):

        # train LR model
        self.gl = GLassoDetector(self, **kwargs)
        time = self.gl.run()

        # save heatmap
        self.heatmap = {}
        try:
            self.heatmap['corr'] = self.gl.corr_heatmap
        except:
            logger.warn('did not produce corr')
        self.heatmap['cov'] = self.gl.cov_heatmap
        self.heatmap_name = {}
        self.heatmap_csv_name = {}
        for name in ['cov', 'corr']:
            self.heatmap_name[name] = "{}_{}_{}".format(ID, self.dataEngine.tableName, self.gl.heatmap_name[name])
            self.heatmap_csv_name[name] = "{}.csv".format(hm_path+self.heatmap_name[name])
        self.save_heatmap(save_heatmap=save_heatmap)

        self.timer.time_point("DONE")
        return time

    def data_loaded(self):
        return self.dataEngine.loaded
    
    def load_heatmap(self, heatmap):
        if isinstance(heatmap, dict):
            self.heatmap = {}
            self.heatmap_csv_name = {}
            for name in ['corr', 'cov']:
                try:
                    self.heatmap[name] = pd.read_csv(heatmap[name], index_col=0)
                    self.heatmap_csv_name[name] = heatmap[name]
                except Exception as e:
                    if name not in heatmap:
                        pass
                    elif heatmap[name] != '' and heatmap[name]:
                        logger.warn('Fail to load heatmap %s: %s'%(name, e))
        else:
            self.heatmap = pd.read_csv(heatmap, index_col=0)
            self.heatmap_csv_name = heatmap

    def visualize_heatmap(self, heatmap, title=None, save=False, filename="heatmap.png"):
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure()
        snsplt = sns.heatmap(heatmap, cmap=sns.color_palette("RdBu_r", 1000), center=0)
        if title:
            snsplt.set_title(title)
        if save:
            snsplt.get_figure().savefig(filename, bbox_inches='tight')

    def save_heatmap(self, save_heatmap='csv'):
        def save_heatmap_helper(heatmap, heatmap_name, heatmap_csv_name):
            if save_heatmap == "db":
                self.create_table_df(heatmap, heatmap_name)
                logger.info("heatmap stored in db: {}".format(heatmap_name))
            elif save_heatmap == "both":
                heatmap.to_csv(heatmap_csv_name)
                logger.info("heatmap stored in csv: {}".format(heatmap_csv_name))
                self.create_table_df(heatmap, heatmap_name)
                logger.info("heatmap stored in db: {}".format(heatmap_name))
            elif save_heatmap == "csv":
                heatmap.to_csv(heatmap_csv_name)
                logger.info("heatmap stored in csv: {}".format(heatmap_csv_name))
            else:
                logger.warn("did not save heatmap")
        if isinstance(self.heatmap, dict):
            for name in self.heatmap.keys():
                save_heatmap_helper(self.heatmap[name], self.heatmap_name[name], self.heatmap_csv_name[name])
        else:
            save_heatmap_helper(self.heatmap, self.heatmap_name, self.heatmap_csv_name)

    '''
    DATABASE HELPER
    '''

    def connect_db(self, db, host, user, password, port):
        '''
        Connects to database
        :param db: database name, default None
        :return:
        '''
        # postgres
        #self.conn = psycopg2.connect(database=db, host=host, user=user, password=password)
        self.engine = create_engine('postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'.format(
                                     user=user, password=password, database=db, host=host, port=port))
        self.conn = self.engine.raw_connection()
        logger.info("Connected to database {}".format(db))

    def query(self, q, value=[]):
        '''
        Exectutes an query
        :param q: query statement, e.g. "SELECT * from table WHERE column1 = ?"
        :param value: used to replace ?, e.g. value = ('id',)
        :return:
        '''
        #self.conn.row_factory = lambda cursor, row: row[0]
        c = self.conn.cursor()
        if len(value) != 0:
            c.execute(q,value)
        else:
            c.execute(q)
        logger.debug("Executed Query [ {} ]".format(q))
        return c.fetchall()

    def query_df(self,q, value=[]):
        if len(value) != 0:
            df = pd.read_sql_query(q, self.conn, params=value)
        else:
            df = pd.read_sql_query(q, self.conn)
        logger.debug("Executed Query [ {} ]".format(q))
        return df

    def query_df_with_index(self,q,index="id"):
        df = pd.read_sql_query(q, self.engine, index_col=index)
        logger.debug("Executed Query [ {} ]".format(q))
        return df

    def insert(self, q, value):
        c = self.conn.cursor()
        c.execute(q,value)
        self.conn.commit()
        logger.debug("Executed Insert Query [ {} with values {}]".format(q,value))
        return c.lastrowid

    def modify(self,q):
        c = self.conn.cursor()
        c.execute(q)
        self.conn.commit()
        logger.debug("Executed Query [ {} ]".format(q))

    def create_table(self, q, tableName):
        c = self.conn.cursor()
        c.execute("DROP TABLE IF EXISTS {};".format(tableName))
        c.execute(q)
        self.conn.commit()
        logger.debug("Create Table [ {} ]".format(q))

    def create_table_df(self, df, tableName):
        df.index.name = "id"
        df.to_sql(tableName, self.engine, if_exists='replace')
        logger.debug("Create Table [ {} ] From Dataframe".format(tableName))

    def table_lists(self):
        df = self.query_df("SELECT name FROM sqlite_master WHERE type='table';")
        return df

    def table_info(self, tableName):
        '''
        Gets the schema of a table
        :param tableName: name of the table
        :return:
        '''
        return self.query_df('PRAGMA TABLE_INFO({})'.format(tableName))

