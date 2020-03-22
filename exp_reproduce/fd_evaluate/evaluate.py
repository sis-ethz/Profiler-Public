import pandas as pd
import numpy as np
import math
import os
import json


# for threshold in range(1, df.shape[0]+1):


def evaluate(file_name, ground_truth_col, inferred_pairs_col, df=None, base_dir='./ordering/'):
    # file_name = './child.csv'

    if df is None:
        print("Read from CSV file:", os.path.join(base_dir, file_name))
        df = pd.read_csv(os.path.join(base_dir, file_name))
        df = df.fillna(-1)
        assert df is not None
        return evaluate(file_name, ground_truth_col, inferred_pairs_col, df=df, base_dir=base_dir)
    # ground_truth_col = 'Ground truth pairs extended'
    # inferred_pairs_col = 'RFI-top1-a-1'
    #  PYRO  GL  RFI-top1-a-0.3
    # inferred_pairs_col = 'TANE'

    result_summary = {}
    threshold = np.inf
    list_ground_truth = []
    list_inferred = []
    attr_dict = {}
    for i in range(df.shape[0]):
        # print(df[ground_truth_col][i])
        if df[ground_truth_col][i] != -1:
            attr_list = df[ground_truth_col][i].replace(' ', '').split('->')
            for attr_det in attr_list[0].split(','):
                list_ground_truth.append([attr_det, attr_list[1]])
            # list_ground_truth.append([attr_list[1], attr_list[0]])

        if df[inferred_pairs_col][i] != -1 and i < threshold:
            attr_list = df[inferred_pairs_col][i].replace(' ', '').split('->')
            for attr_det in attr_list[0].split(','):
                if [attr_det, attr_list[1]] not in list_inferred:  # add this line may change the results
                    list_inferred.append([attr_det, attr_list[1]])
                    list_inferred.append([attr_list[1], attr_det])

    # print("ground truth: \n", list_ground_truth)
    # print("inferred: \n", list_inferred)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(list_ground_truth)):
        if list_ground_truth[i] in list_inferred:
            # print("TP: ", list_ground_truth[i])
            TP += 1
    if len(list_inferred) == 0:
        return (np.nan, np.nan, np.nan), df
    else:
        precision = TP / (len(list_inferred)/2)
    recall = TP / (len(list_ground_truth))

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2*precision*recall / (precision + recall)

    # print("Precison Recall f1\t\n%f\n%f\n%f"%(precision, recall, f1_score))
    result_summary[threshold] = [precision, recall, f1_score]
    print("Results from %s, on %s" % (file_name, inferred_pairs_col))
    print("t = [precision: %.3f , recall: %.3f , f1: %.3f , total-inf: %d , total-groundtruth: %d]" %
          (precision, recall, f1_score, len(list_inferred)/2, len(list_ground_truth)))
    return (precision, recall, f1_score), df
# print("Recall\t%f"%recall)
# print("f1\t%f"%f1_score)

# print(result_summary)


def copy_result_files(base_file, to_file):
    if base_file == to_file:
        return
    path = './syn/'
    with open(os.path.join(path, base_file), 'r') as f, open(os.path.join(path, to_file), 'w') as g:
        m_file = f.read()
        g.write(m_file)
        f.close()
        g.close()


if __name__ == '__main__':
    # base_file = 'base_tlarge_rsmall_dsmall_n0_0_nhigh.csv'
    # mods = ['small', 'large']
    # for r in mods:
    #     for t in mods:
    #         for d in mods:
    #             for n in ['nlow', 'nhigh']:
    #                 to_file = '_'.join(
    #                     ['base', 't' + t, 'r' + r, 'd' + d, 'n0_0', n + '.csv'])
    #                 copy_result_files(base_file, to_file)
    base_dir = './synthetic/'
    files = []

    for f in os.listdir(base_dir):
        if 'csv' in f:
            files.append(f)

    # col_candidates = ['AutoFD', 'AutoFD_sparse', 'GL', 'PYRO', 'TANE']
    # col_candidates = ['AutoFD', 'AutoFD_sparse', 'GL', 'PYRO', 'TANE',
    #                   'RFI-top1-a-0.3', 'RFI-top1-a-0.5', 'RFI-top1-a-1']

    # col_candidates = ['AutoFD', 'AutoFD_sparse_0.002', 'AutoFD_sparse_0.004', 'AutoFD_sparse_0.006', 'AutoFD_sparse_0.008',
    #                   'AutoFD_sparse_0.010']
    # col_candidates = ['AutoFD', 'AutoFD gd', 'AutoFD_amd',
    #                   'AutoFD_colamd', 'AutoFD_metis',	'AutoFD_natural', 'AutoFD_nesdis']

    col_candidates = ['CORDS']

    ground_truth_col = 'Ground truth pairs extended'
    results = {}
    for f in files:
        res = {}
        df = None
        for col in col_candidates:
            evaluate_res, df = evaluate(
                f, ground_truth_col, col, df=df, base_dir=base_dir)
            res[col] = evaluate_res
        results[f] = res
    print("Results: ")
    print(results)
    with open('result_syn_CORDS.json', 'w') as fp:
        json.dump(results, fp)
