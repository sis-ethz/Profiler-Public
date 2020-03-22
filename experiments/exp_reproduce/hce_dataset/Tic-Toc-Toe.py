
# coding: utf-8

# In[1]:

from profiler.core import *


# In[2]:

pf = Profiler(workers = 2, tol=1e-6, eps=0.05, embedtxt=True)


# In[3]:

pf.session.load_data(name='ttt', src=FILE, fpath='./hce_data/ttt/ttt.csv', check_param=True, na_values='empty')


# In[4]:

# pf.session.change_dtypes(['tls', 'ZipCode', 'PhoneNumber', 'State', 'EmergencyService','Score', 'Sample'], 
#                             [CATEGORICAL, NUMERIC, CATEGORICAL, TEXT, TEXT, NUMERIC, NUMERIC],
#                             [None, None, None, None, None, r'(\d+)%', r'(\d+)\spatients'])


# In[5]:

pf.session.load_training_data(multiplier = 500)


# In[6]:

autoregress_matrix = pf.session.learn_structure(sparsity=0.007, infer_order=True)


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




# In[ ]:




# In[ ]:



