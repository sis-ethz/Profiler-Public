import pandas as pd
import numpy as np
from profiler.core import *
import sys
import getopt
from tqdm import tqdm
import time

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


def run_cords(dataset, base_dir):

    start = time.time()
    difference = False
    sparsity = 0
    if '.csv' not in dataset:
        dataset += '.csv'

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
    pf = Profiler(workers=50, tol=1e-6, eps=0.05, embedtxt=False)

    pf.session.load_data(name='%s' % name, src=FILE,
                         fpath='%s' % os.path.join(base_dir, dataset), check_param=True)
    pf.session.load_training_data(multiplier=None, difference=difference)

    autoregress_matrix = pf.session.learn_structure(
        sparsity=sparsity, infer_order=False)

    corelation_pairs = pf.session.get_corelations(write_to=(
        './result/%s/FD_' % name) + name + '_cords')
    end = time.time()

    with open(os.path.join('./result/%s/%s_time_point.txt' % (name, name)), 'w') as f:
        print("Runtime = %.3f" % (end - start))
        f.write("CORDS run time: %.3f" % (end - start))

    # pf.session.timer.to_csv(
    #     filename='./result/%s/%s_time_point.csv' % (name, name))


if __name__ == "__main__":

    # high_files = os.listdir('./nhigh/')
    # low_files = os.listdir('./nlow/')

    BN_files = os.listdir('./BN/')
    # hce_files = os.listdir('./hce_dataset/')

    for f in tqdm(BN_files):
        run_cords(f, base_dir='./BN/')

    # for f in tqdm(low_files):
    #     run_cords(f, base_dir='./nlow/')

    # for f in tqdm(high_files):
    #     run_cords(f, base_dir='./nhigh/')

    # for f in tqdm(hce_files):
    #     run_cords(f, base_dir='./hce_dataset/')
