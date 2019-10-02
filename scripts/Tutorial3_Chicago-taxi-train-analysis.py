
# coding: utf-8

# In[ ]:

## 0. Load Data


# In[1]:

from __future__ import print_function
import sys, os
import tempfile, urllib, zipfile

# Set up some globals for our file paths
BASE_DIR = tempfile.mkdtemp()
DATA_DIR = os.path.join(BASE_DIR, 'data/taxi/')
OUTPUT_DIR = os.path.join(BASE_DIR, 'chicago_taxi_output')
TRAIN_DATA = os.path.join(DATA_DIR, 'train', 'data.csv')
EVAL_DATA = os.path.join(DATA_DIR, 'eval', 'data.csv')
SERVING_DATA = os.path.join(DATA_DIR, 'serving', 'data.csv')

# Download the zip file from GCP and unzip it
zip, headers = urllib.request.urlretrieve('https://storage.googleapis.com/tfx-colab-datasets/chicago_data.zip')
zipfile.ZipFile(zip).extractall(BASE_DIR)
zipfile.ZipFile(zip).close()

print("Here's what we downloaded:")
get_ipython().system("ls -R {os.path.join(BASE_DIR, 'data')}")


# In[2]:

from profiler.core import *


# ## 1. Instantiate Engine
# * workers : number of processes
# * tol     : tolerance for differences when creating training data (set to 0 if data is completely clean)
# * eps     : error bound for inverse covariance estimation (since we use conservative calculation when determining minimum sample size, we recommend to set eps <= 0.01)
# * embedtxt: if set to true, differentiate b/w textual data and categorical data, and use word embedding for the former

# In[3]:

pf = Profiler(workers=2, tol=0, eps=0.05, embedtxt=False)


# ## 2. Load Data
# * name: any name you like
# * src: \[FILE; DF; DB (not implemented)\]
# * fpath: required if src == FILE
# * df: required if src == DF
# * check_param: print parameters used for data loading

# In[4]:

pf.session.load_data(name='hospital', src=FILE, fpath='/var/folders/gm/yftb1thj7rl4hwccxwffhpmc0000gn/T/tmpql7omx4z/data/eval/data.csv', check_param=True)


# In[5]:

pf.session.ds.df.head()


# ### 2.1 Change Data Types of Attributes
# * required input:
#     * a list of attributes
#     * a list of data types (must match the order of the attributes; can be CATEGORICAL, NUMERIC, TEXT, DATE)
# * optional input:
#     * a list of regular expression extractor

# In[6]:

# pf.session.change_dtypes(['ProviderNumber', 'ZipCode', 'PhoneNumber', 'State', 'EmergencyService','Score', 'Sample'], 
#                             [CATEGORICAL, NUMERIC, CATEGORICAL, TEXT, TEXT, NUMERIC, NUMERIC],
#                             [None, None, None, None, None, r'(\d+)%', r'(\d+)\spatients'])


# ### 2.2. Load/Train Embeddings for TEXT
# * path: path to saved/to-save embedding folder
# * load: set to true -- load saved vec from 'path'; set to false -- train locally
# * save: (only for load = False) save trained vectors to 'path'

# In[7]:

#pf.session.load_embedding(save=True, path='data/hospital/', load=True)


# ## 3. Load Training Data
# * multiplier: if set to None, will infer the minimal sample size; otherwise, it will create (# samples) * (# attributes) * (multiplier) training samples

# In[8]:

pf.session.load_training_data(multiplier = None)


# ## 4. Learn Structure
# * sparsity: intensity of L1-regularizer in inverse covariance estimation (glasso)
# * take_neg: if set to true, consider equal -> equal only

# In[9]:

autoregress_matrix = pf.session.learn_structure(sparsity=0.05, infer_order=True)


# * score: 
#     * "fit_error": mse for fitting y = B'X + c for each atttribute y 
#     * "training_data_fd_vio_ratio": the higher the score, the more violations of FDs in the training data. (bounded: \[0,1\])

# In[10]:

parent_sets = pf.session.get_dependencies(score="fit_error")


# ## 5. Visualization

# In[15]:

pf.session.visualize_covariance()


# In[12]:

pf.session.visualize_inverse_covariance()


# In[13]:

pf.session.visualize_autoregression()


# In[14]:

pf.session.timer.get_stat()


# In[16]:

import matplotlib.pyplot as plt


# In[31]:

fig, axs = plt.subplots(2, figsize=(8,5), sharex=True)
i = 0
fig.suptitle('eval')
for cls, group in pf.session.ds.df.groupby(['payment_type']):
    if cls == "prcard":
        continue
    if cls == "dispute":
        break
    group['tips'].hist(bins=10, ax=axs[i])
    axs[i].set_title(cls)
    axs[i].set_xlim(left=0,right=20)
    i += 1


# In[38]:

for cls, group in pf.session.ds.df.groupby(['payment_type']):
#     if cls == "cash":
#         break
    if cls == "credit card":
        break


# In[41]:

count, bins = np.histogram(group.tips, bins=[ 0.   ,  2.655,  5.31 ,  7.965, 10.62 , 13.275, 15.93 , 18.585,
       21.24 , 23.895, 26.55 ])


# In[42]:

bins


# In[44]:

count


# In[ ]:



