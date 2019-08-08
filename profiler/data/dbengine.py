from sqlalchemy import create_engine
import pandas as pd
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DBEngine(object):

    def __init__(self, db='profiler', host='localhost', user="profileruser", password="abcd1234", port=5432):
        self.connect_db(db, host, user, password, port)

    def connect_db(self, db, host, user, password, port):
        """
        connect to database
        :param db:
        :param host:
        :param user:
        :param password:
        :param port:
        :return:
        """
        # postgres
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
