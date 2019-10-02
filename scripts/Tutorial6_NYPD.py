
# coding: utf-8

# ## 0. Load Data

# ## 1. Instantiate Engine
# * workers : number of processes
# * tol     : tolerance for differences when creating training data (set to 0 if data is completely clean)
# * eps     : error bound for inverse covariance estimation (since we use conservative calculation when determining minimum sample size, we recommend to set eps <= 0.01)
# * embedtxt: if set to true, differentiate b/w textual data and categorical data, and use word embedding for the former

# In[1]:

from profiler.core import *


# In[2]:

pf = Profiler(workers=2, tol=1e-6, eps=0.01, embedtxt=True)


# ## 2. Load Data
# * name: any name you like
# * src: \[FILE; DF; DB (not implemented)\]
# * fpath: required if src == FILE
# * df: required if src == DF
# * check_param: print parameters used for data loading

# In[3]:

pf.session.load_data(name='nypd', src=FILE, fpath='data/large_data/nypd.csv', check_param=True)


# In[4]:

pf.session.ds.df.head()


# ### 2.1 Change Data Types of Attributes
# * required input:
#     * a list of attributes
#     * a list of data types (must match the order of the attributes; can be CATEGORICAL, NUMERIC, TEXT, DATE)
# * optional input:
#     * a list of regular expression extractor

# In[5]:

pf.session.change_dtypes(["CMPLNT_FR_DT","CMPLNT_FR_TM", "CMPLNT_TO_DT", "CMPLNT_TO_TM"],
                          [DATE,DATE,DATE,DATE])


# ### 2.2. Load/Train Embeddings for TEXT
# * path: path to saved/to-save embedding folder
# * load: set to true -- load saved vec from 'path'; set to false -- train locally
# * save: (only for load = False) save trained vectors to 'path'

# In[6]:

pf.session.load_embedding(save=True, path='data/nypd/', load=True)


# In[7]:

pf.session.embed.models


# ## 3. Load Training Data
# * multiplier: if set to None, will infer the minimal sample size; otherwise, it will create (# samples) * (# attributes) * (multiplier) training samples

# In[8]:

pf.session.load_training_data(multiplier = None, sample_frac=0.2)


# ## 4. Learn Structure
# * sparsity: intensity of L1-regularizer in inverse covariance estimation (glasso)
# * take_neg: if set to true, consider equal -> equal only

# In[9]:

autoregress_matrix = pf.session.learn_structure(sparsity=0.03, infer_order=True)


# * score: 
#     * "fit_error": mse for fitting y = B'X + c for each atttribute y 
#     * "training_data_fd_vio_ratio": the higher the score, the more violations of FDs in the training data. (bounded: \[0,1\])

# In[17]:

parent_sets = pf.session.get_dependencies(score="fit_error")


# ## 5. Visualization

# In[18]:

pf.session.visualize_covariance()


# In[12]:

pf.session.visualize_inverse_covariance()


# In[13]:

pf.session.visualize_autoregression()


# In[14]:

pf.session.timer.get_stat()


# In[ ]:



