import pandas as pd
import numpy as np
from profiler.core import *
import sys
import getopt
from tqdm import tqdm

# join_table_list = ['customer_orders', 'lineitem_orders', 'lineitem_part',
#                    'lineitem_partsupp', 'lineitem_supplier', 'nation_customer', 'nation_region',
#                    'nation_supplier', 'part_partsupp', 'supplier_partsupp']


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "d:", ["dataset="])
    except getopt.GetoptError:
        print('profiler.py -d <dataset>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            dataset = arg
        else:
            print('profiler.py -d <dataset>')
            sys.exit(2)

    sparsity = 0.1
    print("[Info]: Running profiler on %s" % dataset)

    if dataset == "ALL":
        for tbl in [t + '.csv' for t in join_table_list]:
            run_profiler(tbl, sparsity)
    else:
        run_profiler(dataset, sparsity)


def run_profiler(dataset, base_dir, sparsity=0, difference=True, ordering_method='heu'):

    name = dataset.split('.')[0]
    if 'high' in base_dir:
        name += '_nhigh'
    elif 'low' in base_dir:
        name += '_nlow'
    print("[Info]: Running dataset: ", name)
    if not os.path.exists('./result/' + name + '/'):
        os.system('mkdir ./result/' + name + '/')
    if not os.path.exists('./result/%s/heatmap/' % name):
        os.system('mkdir ./result/%s/heatmap/' % name)
    pf = Profiler(workers=50, tol=1e-3, eps=0.05, embedtxt=False)

    pf.session.load_data(name='%s' % name, src=FILE,
                         fpath='%s' % os.path.join(base_dir, dataset), check_param=True)
    # pf.session.change_dtypes(['Column_%d' % i for i in range(58)],
    #                          [CATEGORICAL for _ in range(58)],
    #                          [None for _ in range(58)])
    pf.session.load_training_data(multiplier=10, difference=difference)
    autoregress_matrix = pf.session.learn_structure(
        sparsity=sparsity, infer_order=True, ordering_method=ordering_method)
    parent_sets = pf.session.get_dependencies(
        score="fit_error", write_to=('./result/%s/FD_' % name) + name + '_%s_sparse_%.2f' % ('AutoFD' if difference else 'gl', sparsity))
    pf.session.visualize_covariance(
        filename='./result/%s/heatmap/%s_cov.png' % (name, name))
    pf.session.visualize_inverse_covariance(
        filename='./result/%s/heatmap/%s_inv_cov.png' % (name, name))
    pf.session.visualize_autoregression(
        filename='./result/%s/heatmap/%s_auto_reg.png' % (name, name))
    pf.session.timer.to_csv(
        filename='./result/%s/%s_time_point.csv' % (name, name))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and 'gl' in sys.argv[1].lower():
        diff = False
    else:
        diff = True
    # high_files = os.listdir('./nhigh/')
    # low_files = os.listdir('./nlow/')

    high_files = ['base_tlarge_rsmall_dlarge_n0_0.csv']
    # low_files = ['base_tsmall_rsmall_dsmall_n0_0.csv']
    low_files = []

    for f in tqdm(high_files):
        run_profiler(f, base_dir='./nhigh/', sparsity=0.02,
                     difference=diff, ordering_method='natural')
    # for f in tqdm(low_files):
    #     run_profiler(f, base_dir='./nlow/', sparsity=0, difference=diff)
