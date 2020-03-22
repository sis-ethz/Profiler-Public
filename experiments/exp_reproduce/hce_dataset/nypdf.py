
# coding: utf-8

# In[1]:

from profiler.core import *


# In[2]:

pf = Profiler(workers = 2, tol=1e-6, eps=0.05, embedtxt=True)


# In[3]:

pf.session.load_data(name='nypdf', src=FILE, fpath='./hce_data/nypdf/nypdf.csv', check_param=True, na_values='empty')


# In[4]:

pf.session.load_embedding(save=True, path='./hce_data/nypdf/', load=True)


# In[5]:

pf.session.load_training_data(multiplier = None)


# In[6]:

autoregress_matrix = pf.session.learn_structure(sparsity=0.01, infer_order=True)


# In[7]:

parent_sets = pf.session.get_dependencies(score="fit_error")


# In[8]:

pf.session.visualize_covariance()


# In[9]:

pf.session.visualize_inverse_covariance()


# In[10]:

pf.session.visualize_autoregression()


# In[11]:

pf.session.timer.get_stat()


# In[12]:

pf.session.timer.to_csv()


# In[ ]:



