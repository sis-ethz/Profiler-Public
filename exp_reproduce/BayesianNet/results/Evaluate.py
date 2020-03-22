import pandas as pd
import numpy as np
import math



#for threshold in range(1, df.shape[0]+1):
def evaluate():
    file_name = './earthquake.csv'
    df = pd.read_csv(file_name)
    df = df.fillna(-1)
    print("Read from CSV file:")

    ground_truth_col = 'Ground truth pairs extended'
    inferred_pairs_col = 'RFI-top1-a-1'
    #  PYRO  GL  RFI-top1-a-0.3

    result_summary = {}   
    threshold =  np.inf
    list_ground_truth = []
    list_inferred = []
    attr_dict = {}
    for i in range(df.shape[0]):
        # print(df[ground_truth_col][i])
        if df[ground_truth_col][i] != -1:
            attr_list = df[ground_truth_col][i].replace(' ','').split('->')
            list_ground_truth.append([attr_list[0], attr_list[1]])
            # list_ground_truth.append([attr_list[1], attr_list[0]])

        if df[inferred_pairs_col][i] != -1 and i < threshold:
            attr_list = df[inferred_pairs_col][i].replace(' ','').split('->')
            list_inferred.append([attr_list[0], attr_list[1]])
            list_inferred.append([attr_list[1], attr_list[0]])

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
    print("Results from %s, on %s"%(file_name, inferred_pairs_col))
    print("t = [precision: %.3f , recall: %.3f , f1: %.3f , total-inf: %d , total-groundtruth: %d]"%(precision, recall, f1_score, len(list_inferred)/2, len(list_ground_truth)))
    return precision, recall, f1_score
# print("Recall\t%f"%recall)
# print("f1\t%f"%f1_score)

# print(result_summary)



evaluate()