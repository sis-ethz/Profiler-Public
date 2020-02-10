import pandas as pd
import numpy as np
from profiler.core import *
import sys
import getopt
from tqdm import tqdm

# join_table_list = ['customer_orders', 'lineitem_orders', 'lineitem_part',
#                    'lineitem_partsupp', 'lineitem_supplier', 'nation_customer', 'nation_region',
#                    'nation_supplier', 'part_partsupp', 'supplier_partsupp']
dataset_names = ['ttt', 'thoraric', 'nypdf', 'hospital', 'mam', 'australian']


def run_profiler(dataset, base_dir, result_dir='result', sparsity=0, difference=True, ordering_method='heu'):
    if '.csv' not in dataset:
        dataset += '.csv'
    name = dataset.split('.')[0]
    if 'high' in base_dir:
        name += '_nhigh'
    elif 'low' in base_dir:
        name += '_nlow'
    print("[Info]: Running dataset: ", name)
    if not os.path.exists('./%s/' % result_dir + name + '/'):
        os.system('mkdir ./%s/' % result_dir + name + '/')
    if not os.path.exists('./%s/%s/heatmap/' % (result_dir, name)):
        os.system('mkdir ./%s/%s/heatmap/' % (result_dir, name))
    pf = Profiler(workers=50, tol=0, eps=0.05, embedtxt=False)
    pf.session.load_data(name='%s' % name, src=FILE,
                         fpath='%s' % os.path.join(base_dir, dataset), check_param=True)
    pf.session.load_training_data(multiplier=50, difference=difference)
    autoregress_matrix = pf.session.learn_structure(
        sparsity=sparsity, infer_order=True, ordering_method=ordering_method, shrinkage=0)
    parent_sets = pf.session.get_dependencies(
        score="fit_error", write_to=('./%s/%s/FD_' % (result_dir, name)) + name + '_sparse_%.2f' % sparsity)
    pf.session.visualize_covariance(
        filename='./%s/%s/heatmap/%s_cov.png' % (result_dir, name, name))
    pf.session.visualize_inverse_covariance(
        filename='./%s/%s/heatmap/%s_inv_cov.png' % (result_dir, name, name))
    pf.session.visualize_autoregression(
        filename='./%s/%s/heatmap/%s_auto_reg.png' % (result_dir, name, name))
    pf.session.timer.to_csv(
        filename='./%s/%s/%s_time_point.csv' % (result_dir, name, name))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and 'gl' in sys.argv[1].lower():
        diff = False
    else:
        diff = True
    for ordering_method in ['heu', 'natural', 'amd', 'metis', 'nesdis', 'colamd']:
        os.system('mkdir ./results/%s/' % ordering_method)
        for f in tqdm(dataset_names):
            run_profiler(f, base_dir='./hce_data/%s/' % f, result_dir='results/%s' % ordering_method, sparsity=0.01,
                         difference=diff, ordering_method=ordering_method)
