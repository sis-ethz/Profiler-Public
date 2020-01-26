import pandas as pd
import numpy as np
import os
from multiprocessing.dummy import Pool as ThreadPool
import sys
from lib.query_util import *
from lib.table_util import *
import json

dataset_names = ['asia', 'cancer', 'alarm', 'australian', 'child',
                 'earthquake', 'hospital', 'mam', 'nypd', 'thoraric', 'ttt']
CMD = 'python3 run_tane_exp.py {}'

cmds = [CMD.format(cmd) for cmd in dataset_names]

workers = 11
pool = ThreadPool(workers)
print("Created pool with %d workers" % workers)
print("Start to run %d workers" % workers)
results = pool.map_async(os.system, cmds)
pool.close()
pool.join()
print("Finished !")
