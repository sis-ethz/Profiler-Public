import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def read_add_col(name, val):
    df = pd.read_csv('exp1/experiment1_results'+name+'.csv', index_col=False)
    df['high_dim'] = [val]*df.shape[0]
    return df


def vis_recall_improv(df):
    fig, ax = plt.subplots()
    data = []
    classes = []
    err = []
    for cls, group in df.groupby('method'):
        d = group['c_r']-group['o_r']
        data.append(d.mean())
        classes.append(cls)
        err.append(d.std()/ np.sqrt(d.shape[0]))
    print("max improvements: %.4f"%max(data))
    ax.bar(classes,data)
    ax.errorbar(classes, data, yerr=err, fmt='o',ecolor='r',color='b')
    ax.set_xlabel('outlier detection algorithm')
    ax.set_ylabel('improvements in recall')
    

def vis_diff_neighbors(df):
    knn = df[df['knn'] & ~df['high_dim']]
    knn_high = df[df['knn'] & df['high_dim']]
    threshold = df[df['knn']==False]
    visualize_best_f1(knn, title='knn')
    visualize_best_f1(knn_high, title='knn_high_dim')
    visualize_best_f1(threshold, title='threshold')


# def visualize_best_f1(df, title=None):
#     c_f1 = []
#     s_f1 = []
#     o_f1 = []
#     classes = []
#     for cls, group in df.groupby('method'):
#         c_f1.append(group['c_f1'].max())
#         s_f1.append(group['s_f1'].max())
#         o_f1.append(group['o_f1'].max())
#         classes.append(cls)
#     fig, ax = plt.subplots()
#     width = 0.2
#     ax.bar(np.arange(len(o_f1)), o_f1, width=width)
#     ax.bar(np.arange(len(s_f1))+width, s_f1, width=width)
#     ax.bar(np.arange(len(c_f1))+width*2, c_f1, width=width)
#     height = [.015, .06, .105]
#     heights = [0,0,0]
    
#     for i, v in zip(np.arange(len(o_f1)), zip(o_f1, s_f1, c_f1)):
#         idx = np.argsort(list(v))
#         for j, ind in enumerate(idx):
#             heights[ind] = height[j]
#         for c, vc in enumerate(v): 
#             ax.text(i-0.1+width*c, vc + heights[c], "%.4f"%vc)
#     ax.set_ylim(0,1)
#     ax.set_xticks(np.arange(len(classes))+width)
#     ax.set_xticklabels(classes)
#     ax.set_xlabel('outlier detection algorithm')
#     ax.set_ylabel('f1')
#     ax.legend(['overall', 'structured', 'combined'])
#     if title is not None:
#         ax.set_title(title)
        
def visualize_best_f1(df, groupby='method', title=None):
    c_f1 = []
    s_f1 = []
    o_f1 = []
    classes = []
    for cls, group in df.groupby(groupby):
        c_f1.append(group['c_f1'].max())
        s_f1.append(group['s_f1'].max())
        o_f1.append(group['o_f1'].max())
        classes.append(cls)
    fig, ax = plt.subplots()
    width = 0.2
    ax.bar(np.arange(len(o_f1)), o_f1, width=width)
    ax.bar(np.arange(len(s_f1))+width, s_f1, width=width)
    ax.bar(np.arange(len(c_f1))+width*2, c_f1, width=width)
    height = [.015, .06, .105]
    heights = [0,0,0]
    
    for i, v in zip(np.arange(len(o_f1)), zip(o_f1, s_f1, c_f1)):
        idx = np.argsort(list(v))
        for j, ind in enumerate(idx):
            heights[ind] = height[j]
        for c, vc in enumerate(v): 
            ax.text(i-0.1+width*c, vc + heights[c], "%.4f"%vc)
    ax.set_ylim(0,1)
    ax.set_xticks(np.arange(len(classes))+width)
    ax.set_xticklabels(classes)
    ax.set_xlabel('outlier detection algorithm')
    ax.set_ylabel('f1')
    ax.legend(['overall', 'structured', 'combined'])
    if title is not None:
        ax.set_title(title)