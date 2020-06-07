# Add global arguments
import random
import os
import logging
import numpy as np
from profiler.globalvar import *
from profiler.learner import StructureLearner
from profiler.data.embedding import EmbeddingEngine
from profiler.data.transformer import TransformEngine
from profiler.data.dataset import Dataset
from profiler.utility import GlobalTimer
import matplotlib
# matplotlib.use("Agg")


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


logging.basicConfig(
    format="%(asctime)s - [%(levelname)5s] - %(message)s", datefmt='%H:%M:%S')
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
    (('-p', '--null_policy'),
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
         'default': 1e-6,
         'type': float,
         'help': 'tolerance for differences'}),
    (('-e', '--eps'),
        {'metavar': 'EPS',
         'dest': 'eps',
         'default': 0.01,
         'type': float,
         'help': "error bound for inverse_covariance estimation"}),
    (('-n', '--null'),
        {'metavar': 'NULL',
         'dest': 'null',
         'default': "_empty_",
         'type': str,
         'help': 'null values'}),
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
    (tuple(['--inequality']),
        {'default': False,
         'dest': 'inequality',
         'action': 'store_true',
         'help': 'enable inequality operators'}),
    (tuple(['--continuous']),
     {'default': False,
      'dest': 'continuous',
      'action': 'store_true',
      'help': "use [0,1] instead of {0,1} for operator [EQ, NEQ, GT, LT] evaluation"
      }
     )
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
        np.random.seed(seed=env['seed'])

        # Initialize members
        self.embed = None
        self.training_data = None
        self.name = name
        self.env = env
        self.timer = GlobalTimer()
        self.ds = Dataset(name, env)
        self.trans_engine = TransformEngine(env, self.ds)
        self.struct_engine = StructureLearner(self, env, self.ds)
        #self.eval_engine = EvalEngine(env, self.ds)

    def load_data(self, name="", src=FILE, fpath='', df=None, check_param=False, **kwargs):
        """
        load_data takes the filepath to a CSV file to load as the initial dataset.
        :param name: (str) name to initialize dataset with.
        :param src: (str) input source ["file", "df", "db"]
        :param fpath: (str) filepath to CSV file.
        :param kwargs: 'na_values', 'header', 'dropcol', 'encoding', 'nan' (representation for null values)
        """
        self.timer.time_start('Load Data')
        self.ds.load_data(name, src, fpath, df, check_param, **kwargs)
        self.timer.time_end('Load Data')

    def change_operators(self, names, ops):
        self.ds.change_operators(names, ops)

    def change_dtypes(self, names, types, regexs=None):
        if regexs is None:
            regexs = [None] * len(names)
        self.ds.change_dtypes(names, types, regexs)

    def load_embedding(self, **kwargs):
        self.timer.time_start('Load Embedding')
        if not self.embed:
            self.embed = EmbeddingEngine(self.env, self.ds)
        self.embed.train(**kwargs)
        self.timer.time_end('Load Embedding')

    def load_training_data(self, multiplier=None, sample_frac=1, difference=True):
        self.timer.time_start('Create Training Data')
        self.trans_engine.create_training_data(multiplier=multiplier, sample_frac=sample_frac,
                                               embed=self.embed, difference=difference)
        self.timer.time_end('Create Training Data')

    def learn_structure(self, **kwargs):
        self.timer.time_start('Learn Structure')
        results = self.struct_engine.learn(**kwargs)
        self.timer.time_end('Learn Structure')
        return results

    def get_dependencies(self, heatmap=None, score="training_data_vio_ratio", write_to='FDs'):
        self.timer.time_start('Get Dependencies')
        results = self.struct_engine.get_dependencies(
            heatmap=heatmap, score=score, write_to=write_to)
        self.timer.time_end('Get Dependencies')
        print(results)
        return results

    def get_corelations(self, heatmap=None, write_to=None):
        self.timer.time_start('Get Corelations')
        results = self.struct_engine.get_corelations(
            heatmap=heatmap, write_to=write_to)
        self.timer.time_end('Get Corelations')
        print(results)
        return results

    def visualize_covariance(self, filename='Covariance Matrix', write_pairs_file=None):
        self.struct_engine.visualize_covariance(
            filename=filename, write_pairs_file=write_pairs_file)

    def visualize_inverse_covariance(self, filename='Inverse Covariance Matrix'):
        self.struct_engine.visualize_inverse_covariance(filename=filename)

    def visualize_autoregression(self, filename='Autoregression Matrix'):
        self.struct_engine.visualize_autoregression(filename=filename)
