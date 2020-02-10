import pandas as pd
import numpy as np
import math
import os

# for threshold in range(1, df.shape[0]+1):


def evaluate():
    file_name = './child.csv'
    df = pd.read_csv(file_name)
    df = df.fillna(-1)
    print("Read from CSV file:")

    ground_truth_col = 'Ground truth pairs extended'
    # inferred_pairs_col = 'RFI-top1-a-1'
    #  PYRO  GL  RFI-top1-a-0.3
    inferred_pairs_col = 'TANE'

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
    return precision, recall, f1_score
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
    base_file = 'base_tlarge_rsmall_dsmall_n0_0_nhigh.csv'
    mods = ['small', 'large']
    for r in mods:
        for t in mods:
            for d in mods:
                for n in ['nlow', 'nhigh']:
                    to_file = '_'.join(
                        ['base', 't' + t, 'r' + r, 'd' + d, 'n0_0', n + '.csv'])
                    copy_result_files(base_file, to_file)

    # evaluate()
