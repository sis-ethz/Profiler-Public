
# coding: utf-8

# ## 0. Load Data

# In[1]:

from scipy.io import loadmat
import numpy as np
data = loadmat('data/cardio.mat')


# In[2]:

gt = data['y']
gt = gt.reshape(-1,)
gt_idx = np.array(range(gt.shape[0]))[gt == 1]


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

pf = Profiler(workers=2, tol=1e-6, eps=0.05, embedtxt=False)


# ## 2. Load Data
# * name: any name you like
# * src: \[FILE; DF; DB (not implemented)\]
# * fpath: required if src == FILE
# * df: required if src == DF
# * check_param: print parameters used for data loading

# In[6]:

pf.session.load_data(src=DF, df=df, check_param=True)


# ### 2.1 Change Data Types of Attributes
# * required input:
#     * a list of attributes
#     * a list of data types (must match the order of the attributes; can be CATEGORICAL, NUMERIC, TEXT, DATE)
# * optional input:
#     * a list of regular expression extractor

# In[7]:

# pf.session.change_dtypes(['ProviderNumber', 'ZipCode', 'PhoneNumber', 'State', 'EmergencyService','Score', 'Sample'], 
#                             [CATEGORICAL, NUMERIC, CATEGORICAL, TEXT, TEXT, NUMERIC, NUMERIC],
#                             [None, None, None, None, None, r'(\d+)%', r'(\d+)\spatients'])


# ### 2.2. Load/Train Embeddings for TEXT
# * path: path to saved/to-save embedding folder
# * load: set to true -- load saved vec from 'path'; set to false -- train locally
# * save: (only for load = False) save trained vectors to 'path'

# In[8]:

#pf.session.load_embedding(save=True, path='data/hospital/', load=True)


# ## 3. Load Training Data
# * multiplier: if set to None, will infer the minimal sample size; otherwise, it will create (# samples) * (# attributes) * (multiplier) training samples

# In[9]:

pf.session.load_training_data(multiplier = None)


# ## 4. Learn Structure
# * sparsity: intensity of L1-regularizer in inverse covariance estimation (glasso)
# * take_neg: if set to true, consider equal -> equal only

# In[10]:

autoregress_matrix = pf.session.learn_structure(sparsity=0.01,
                                                infer_order=True)


# * score: 
#     * "fit_error": mse for fitting y = B'X + c for each atttribute y 
#     * "training_data_fd_vio_ratio": the higher the score, the more violations of FDs in the training data. (bounded: \[0,1\])

# In[11]:

parent_sets = pf.session.get_dependencies(score="fit_error")


# ## 5. Visualization

# In[12]:

pf.session.visualize_covariance()


# In[13]:

pf.session.visualize_inverse_covariance()


# In[14]:

pf.session.visualize_autoregression()


# In[15]:

pf.session.timer.get_stat()


# In[16]:

from profiler.app.od import *


# In[1]:

contamination = 0.1


# In[2]:

detector = ScikitDetector(pf.session.ds.df, attr=pf.session.ds.dtypes, method="ocsvm", gt_idx=gt_idx,
                           nu=contamination, gamma='auto', tol=1e-6, t=0.01)
detector.run_all(parent_sets)
detector.evaluate()


# In[17]:

detector = STDDetector(pf.session.ds.df, gt_idx)
detector.run_all(parent_sets)
detector.evaluate()


# In[18]:

detector2 = ISFDetector(pf.session.ds.df, gt_idx)
detector2.run_overall(separate=False)
detector2.compute_pr(detector2.overall)


# In[19]:

detector3 = OCSVMDetector(pf.session.ds.df, gt_idx)
detector3.run_all(parent_sets, separate=False)
detector3.evaluate()


# In[20]:

detector4 = LOFDetector(pf.session.ds.df, gt_idx)
detector4.run_all(parent_sets, separate=False)
detector4.evaluate()


# In[ ]:



