
# coding: utf-8

# In[1]:

from profiler.core import *
pf = Profiler(workers = 2, tol=1e-6, eps=0.05, embedtxt=True)


# In[2]:

pf.session.load_data(name='eeg', src=FILE, fpath='./hce_data/eeg2/eeg2.csv', check_param=True, na_values='empty')


# In[3]:

pf.session.load_training_data(multiplier = None)


# In[4]:

autoregress_matrix = pf.session.learn_structure(sparsity=0.05, infer_order=True)


# In[5]:

parent_sets = pf.session.get_dependencies(score="fit_error")


# In[6]:

pf.session.visualize_covariance()


# In[7]:

pf.session.visualize_inverse_covariance()


# In[8]:

pf.session.visualize_autoregression()


# In[ ]:



