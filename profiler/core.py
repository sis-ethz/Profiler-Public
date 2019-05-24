from profiler.utility import GlobalTimer
from profiler.dataset import Dataset
from profiler.transformer import TransformEngine
from profiler.embedding import EmbeddingEngine
from profiler.learner import StructureLearner
from profiler.globalvar import *
import numpy as np
import logging, os, random


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


logging.basicConfig(format="%(asctime)s - [%(levelname)5s] - %(message)s", datefmt='%H:%M:%S')
root_logger = logging.getLogger()
gensim_logger = logging.getLogger('gensim')
root_logger.setLevel(logging.INFO)
gensim_logger.setLevel(logging.WARNING)


# Arguments for Profiler
arguments = [
    (('-u', '--db_user'),
        {'metavar': 'DB_USER',
         'dest': 'db_user',
         'default': 'profileruser',
         'type': str,
         'help': 'User for DB used to persist state.'}),
    (('-p', '--db-pwd', '--pass'),
        {'metavar': 'DB_PWD',
         'dest': 'db_pwd',
         'default': 'abcd1234',
         'type': str,
         'help': 'Password for DB used to persist state.'}),
    (('-h', '--db-host'),
        {'metavar': 'DB_HOST',
         'dest': 'db_host',
         'default': 'localhost',
         'type': str,
         'help': 'Host for DB used to persist state.'}),
    (('-d', '--db_name'),
        {'metavar': 'DB_NAME',
         'dest': 'db_name',
         'default': 'profiler',
         'type': str,
         'help': 'Name of DB used to persist state.'}),
    (('-w', '--workers'),
     {'metavar': 'WORKERS',
      'dest': 'workers',
      'default': 1,
      'type': int,
      'help': 'How many workers to use for parallel execution. If <= 1, then no pool workers are used.'}),
    (('-n', '--null_policy'),
        {'metavar': 'DB_NAME',
         'dest': 'null_policy',
         'default': NULL_NEQ,
         'type': str,
         'help': 'Policy to handle null. [neq, eq, skip]'}),
    (('-s', '--seed'),
        {'metavar': 'SEED',
         'dest': 'seed',
         'default': 10,
         'type': int,
         'help': 'random seed'}),
    (('-t', '--tol'),
        {'metavar': 'TOL',
         'dest': 'tol',
         'default': 0.01,
         'type': float,
         'help': "tolerance for being 'same'"}),
]

# Flags for Profiler
flags = [
    (tuple(['--verbose']),
        {'default': False,
         'dest': 'verbose',
         'action': 'store_true',
         'help': 'use DEBUG logging level if verbose enabled'}),
    (tuple(['--usedb']),
        {'default': False,
         'dest': 'usedb',
         'action': 'store_true',
         'help': 'use database engine'}),
    (tuple(['--embedtxt']),
        {'default': False,
         'dest': 'embedtxt',
         'action': 'store_true',
         'help': 'use language embedding for textual data'}),
]


class Profiler(object):
    """
    Main Entry Point for Profiler
    """

    def __init__(self, **kwargs):
        """
        Constructor for Holoclean
        :param kwargs: arguments for HoloClean
        """

        # Initialize default execution arguments
        arg_defaults = {}
        for arg, opts in arguments:
            if 'directory' in arg[0]:
                arg_defaults['directory'] = opts['default']
            else:
                arg_defaults[opts['dest']] = opts['default']

        # Initialize default execution flags
        for arg, opts in flags:
            arg_defaults[opts['dest']] = opts['default']

        # check env vars
        for arg, opts in arguments:
            # if env var is set use that
            if opts["metavar"] and opts["metavar"] in os.environ.keys():
                logging.debug(
                    "Overriding {} with env varible {} set to {}".format(
                        opts['dest'],
                        opts["metavar"],
                        os.environ[opts["metavar"]])
                )
                arg_defaults[opts['dest']] = os.environ[opts["metavar"]]

        # Override defaults with manual flags
        for key in kwargs:
            arg_defaults[key] = kwargs[key]

        # Initialize additional arguments
        for (arg, default) in arg_defaults.items():
            setattr(self, arg, kwargs.get(arg, default))

        # Init empty session collection
        self.session = Session(arg_defaults)


class Session(object):

    def __init__(self, env, name="session"):
        """
        Constructor for Profiler session
        :param env: Profiler environment
        :param name: Name for the Profiler session
        """
        # use DEBUG logging level if verbose enabled
        if env['verbose']:
            root_logger.setLevel(logging.DEBUG)
            gensim_logger.setLevel(logging.DEBUG)

        logging.debug('initiating session with parameters: %s', env)

        # Initialize random seeds.
        random.seed(env['seed'])
        #torch.manual_seed(env['seed'])
        np.random.seed(seed=env['seed'])

        # Initialize members
        self.embed = None
        self.training_data = None
        self.name = name
        self.env = env
        self.timer = GlobalTimer()
        self.ds = Dataset(name, env)
        self.trans_engine = TransformEngine(env, self.ds)
        self.struct_engine = StructureLearner(env, self.ds)
        #self.eval_engine = EvalEngine(env, self.ds)

    def load_data(self, name="", src=FILE, fpath='', df=None, **kwargs):
        """
        load_data takes the filepath to a CSV file to load as the initial dataset.
        :param name: (str) name to initialize dataset with.
        :param src: (str) input source ["file", "df", "db"]
        :param fpath: (str) filepath to CSV file.
        :param kwargs: 'na_values', 'header', 'dropcol', 'encoding', 'nan' (representation for null values)
        """
        self.timer.time_start('Load Data')
        self.ds.load_data(name, src, fpath, df, **kwargs)
        self.timer.time_end('Load Data')

    def change_operators(self, **kwargs):
        self.ds.change_operators(**kwargs)

    def change_dtypes(self, **kwargs):
        self.ds.change_dtypes(**kwargs)

    def load_embedding(self, embedding_size=128, embedding_type=ATTRIBUTE_EMBEDDING):
        self.timer.time_start('Load Embedding')
        if not self.embed:
            self.embed = EmbeddingEngine(self.env, self.ds)
        self.embed.train(embedding_size, embedding_type)
        self.timer.time_end('Load Embedding')

    def load_training_data(self, multiplier=15):
        self.timer.time_start('Create Training Data')
        self.training_data = self.trans_engine.create_training_data(multiplier=multiplier, embed=self.embed)
        self.timer.time_end('Create Training Data')

    def learn_structure(self, **kwargs):
        self.timer.time_start('Recover Moral Graph')
        inv_cov = self.struct_engine.recover_moral(self.training_data, **kwargs)
        self.timer.time_end('Recover Moral Graph')
        # self.timer.time_start('Recover DAG')
        # self.struct_engine.recover_dag(inv_cov)
        # self.timer.time_end('Recover DAG')


    #
    #
    # def __init__(self, use_db=True, db='profiler', ID="", host='localhost',
    #              user="profileruser", password="abcd1234", port=5432):
    #
    #
    #     '''
    #     constructor for profiler, create a database or connect to database if existed
    #     create a data engine object
    #     :param db: database name, default None
    #     '''
    #     self.ID = ID
    #     self.use_db = use_db
    #     if use_db:
    #         self.connect_db(db, host, user, password, port)
    #     self.dataEngine = DataEngine(self, use_db)
    #
    #     # init params
    #     self.key = None
    #     self.heatmap = None
    #     self.heatmap_name = None
    #     self.heatmap_csv_name = None
    #     self.detector = None
    #
    #     # start timer
    #     self.timer = GlobalTimer()
    #     self.timer.time_point("START")
    #
    # def load_data(self,
    #               key='id',
    #               # load data
    #               input_type='file',
    #               input_df=None,
    #               name=None,
    #               **kwargs
    #               ):
    #     # load data
    #     self.key = key
    #     self.timer.time_start("Preprocess Data")
    #     self.dataEngine.load_data(input_type=input_type, name=name, input_df=input_df, **kwargs)
    #     self.timer.time_end("Preprocess Data")
    #
    # def load_embedding(self):
    #     self.timer.time_start("Load Embedding")
    #     self.dataEngine.load_embedding()
    #     self.timer.time_end("Load Embedding")
    #
    # def change_dtypes(self, names, types):
    #     self.dataEngine.change_dtypes(names, types)
    #
    # def reset_dtypes(self):
    #     self.dataEngine.dtypes = self.dataEngine.original_dtypes
    #
    # def run_graphical_lasso(self, save_heatmap="csv", hm_path="./", **kwargs):
    #
    #     # train LR model
    #     self.gl = GLassoDetector(self, **kwargs)
    #     time = self.gl.run()
    #
    #     # save heatmap
    #     self.heatmap = {}
    #     try:
    #         self.heatmap['corr'] = self.gl.corr_heatmap
    #     except:
    #         logger.info('did not produce correlation')
    #     try:
    #         self.heatmap['cov'] = self.gl.cov_heatmap
    #     except:
    #         logger.info('did not produce covariance')
    #     self.heatmap_name = {}
    #     self.heatmap_csv_name = {}
    #     for name in ['cov', 'corr']:
    #         self.heatmap_name[name] = "{}_{}".format(self.ID, self.gl.heatmap_name[name])
    #         self.heatmap_csv_name[name] = "{}.csv".format(hm_path+self.heatmap_name[name])
    #     self.save_heatmap(save_heatmap=save_heatmap)
    #
    #     self.timer.time_point("DONE")
    #     return time
    #
    # def data_loaded(self):
    #     return self.dataEngine.loaded
    #
    # def load_heatmap(self, heatmap):
    #     if isinstance(heatmap, dict):
    #         self.heatmap = {}
    #         self.heatmap_csv_name = {}
    #         for name in ['corr', 'cov']:
    #             try:
    #                 self.heatmap[name] = pd.read_csv(heatmap[name], index_col=0)
    #                 self.heatmap_csv_name[name] = heatmap[name]
    #             except Exception as e:
    #                 if name not in heatmap:
    #                     pass
    #                 elif heatmap[name] != '' and heatmap[name]:
    #                     logger.warn('Fail to load heatmap %s: %s'%(name, e))
    #     else:
    #         self.heatmap = pd.read_csv(heatmap, index_col=0)
    #         self.heatmap_csv_name = heatmap
    #
    # def visualize_heatmap(self, heatmap, title=None, save=False, filename="heatmap.png"):
    #     import seaborn as sns
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     snsplt = sns.heatmap(heatmap, cmap=sns.color_palette("RdBu_r", 1000), center=0)
    #     if title:
    #         snsplt.set_title(title)
    #     if save:
    #         snsplt.get_figure().savefig(filename, bbox_inches='tight')
    #
    # def save_heatmap(self, save_heatmap='csv'):
    #     def save_heatmap_helper(heatmap, heatmap_name, heatmap_csv_name):
    #         if save_heatmap == "db":
    #             self.create_table_df(heatmap, heatmap_name)
    #             logger.info("heatmap stored in db: {}".format(heatmap_name))
    #         elif save_heatmap == "both":
    #             heatmap.to_csv(heatmap_csv_name)
    #             logger.info("heatmap stored in csv: {}".format(heatmap_csv_name))
    #             self.create_table_df(heatmap, heatmap_name)
    #             logger.info("heatmap stored in db: {}".format(heatmap_name))
    #         elif save_heatmap == "csv":
    #             heatmap.to_csv(heatmap_csv_name)
    #             logger.info("heatmap stored in csv: {}".format(heatmap_csv_name))
    #         else:
    #             logger.warn("did not save heatmap")
    #     if isinstance(self.heatmap, dict):
    #         for name in self.heatmap.keys():
    #             save_heatmap_helper(self.heatmap[name], self.heatmap_name[name], self.heatmap_csv_name[name])
    #     else:
    #         save_heatmap_helper(self.heatmap, self.heatmap_name, self.heatmap_csv_name)
