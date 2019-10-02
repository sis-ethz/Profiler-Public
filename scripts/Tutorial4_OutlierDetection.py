
# coding: utf-8

# ## 0. Load Data

# In[1]:

from scipy.io import loadmat
data = loadmat('data/cardio.mat')


# In[2]:

gt = data['y']


# In[3]:

import pandas as pd
df = pd.DataFrame(data['X'], columns=['Columns_%d'%i for i in range(data['X'].shape[1])])


# In[4]:

from profiler.core import *


# ## 1. Instantiate Engine
# * workers : number of processes
# * tol     : tolerance for differences when creating training data (set to 0 if data is completely clean)
# * eps     : error bound for inverse covariance estimation (since we use conservative calculation when determining minimum sample size, we recommend to set eps <= 0.01)
# * embedtxt: if set to true, differentiate b/w textual data and categorical data, and use word embedding for the former

# In[5]:

pf = Profiler(workers=2, tol=0.01, eps=0.05, embedtxt=False)


# ## 2. Load Data
# * name: any name you like
# * src: \[FILE; DF; DB (not implemented)\]
# * fpath: required if src == FILE
# * df: required if src == DF
# * check_param: print parameters used for data loading

# In[6]:

pf.session.load_data(src=DF, df=df, check_param=True)


# In[7]:

pf.session.ds.df.head()


# ### 2.1 Change Data Types of Attributes
# * required input:
#     * a list of attributes
#     * a list of data types (must match the order of the attributes; can be CATEGORICAL, NUMERIC, TEXT, DATE)
# * optional input:
#     * a list of regular expression extractor

# In[8]:

# pf.session.change_dtypes(['ProviderNumber', 'ZipCode', 'PhoneNumber', 'State', 'EmergencyService','Score', 'Sample'], 
#                             [CATEGORICAL, NUMERIC, CATEGORICAL, TEXT, TEXT, NUMERIC, NUMERIC],
#                             [None, None, None, None, None, r'(\d+)%', r'(\d+)\spatients'])


# ### 2.2. Load/Train Embeddings for TEXT
# * path: path to saved/to-save embedding folder
# * load: set to true -- load saved vec from 'path'; set to false -- train locally
# * save: (only for load = False) save trained vectors to 'path'

# In[9]:

#pf.session.load_embedding(save=True, path='data/hospital/', load=True)


# ## 3. Load Training Data
# * multiplier: if set to None, will infer the minimal sample size; otherwise, it will create (# samples) * (# attributes) * (multiplier) training samples

# In[10]:

pf.session.load_training_data(multiplier = None)


# ## 4. Learn Structure
# * sparsity: intensity of L1-regularizer in inverse covariance estimation (glasso)
# * take_neg: if set to true, consider equal -> equal only

# In[186]:

autoregress_matrix = pf.session.learn_structure(sparsity=0.01,
                                                infer_order=True)


# * score: 
#     * "fit_error": mse for fitting y = B'X + c for each atttribute y 
#     * "training_data_fd_vio_ratio": the higher the score, the more violations of FDs in the training data. (bounded: \[0,1\])

# In[187]:

parent_sets = pf.session.get_dependencies(score="fit_error")


# ## 5. Visualization

# In[188]:

pf.session.visualize_covariance()


# In[189]:

pf.session.visualize_inverse_covariance()


# In[190]:

pf.session.visualize_autoregression()


# In[191]:

pf.session.timer.get_stat()


# In[192]:

gt = gt.reshape(-1,)
gt_idx = np.array(range(gt.shape[0]))[gt == 1]


# In[193]:

def outlier(data, m=4):
    return abs(data - np.mean(data)) > m * np.std(data)

def prec_recall(outliers, gt_idx):
    outliers = set(outliers)
    tp = 0.0
    # precision
    if len(outliers) == 0:
        print("no outlier is found")
        recall(tp, outliers, gt_idx)
        print("f1: 0")
        return 0
    for i in outliers:
        if i in gt_idx:
            tp += 1
    prec = tp / len(outliers)
    print("with %d detected outliers, precision is: %.4f"%(len(outliers), prec))
    rec = recall(tp, outliers, gt_idx)
    print("f1: %.4f"%(2 * (prec * rec) / (prec + rec)))
    
def recall(tp, outliers, gt_idx):
    if tp == 0:
        print("with %d outliers in gt, recall is: 0"%(len(gt_idx)))
        return 0 
    print("with %d detected outliers, recall is: %.4f"%(len(outliers), tp / len(gt_idx)))
    return tp / len(gt_idx)




# In[199]:

import sklearn
def cmpr_detection(df, left, right, m1=3, m2=3):
    overall=df.index.values[outlier(df[right],m=m1)]
    X = df[left].values.reshape(-1,len(left))
    outliers = list(overall)
    i = 0
    distances = sklearn.metrics.pairwise_distances(X)
    distances = distances / np.nanmax(distances, axis=1)
    indices = np.array(range(distances.shape[0]))
    for dis in distances:
        # normalize distance
        nbr = indices[dis < pf.session.env['tol']]
        outliers.extend(nbr[outlier(df[right].values[nbr], m=m2)])
        i += 1
    return overall, outliers


# In[202]:

from tqdm import tqdm
base = []
improv = []
for child in tqdm(parent_sets):
    if len(parent_sets[child]) == 0:
        continue
    overall, structured = cmpr_detection(df, parent_sets[child], child, m2=5)
    base.extend(list(overall))
    improv.extend(structured)
print("naive approach: ")
prec_recall(base, gt_idx)
print("with structural constraints: ")
prec_recall(improv, gt_idx)


# In[ ]:



