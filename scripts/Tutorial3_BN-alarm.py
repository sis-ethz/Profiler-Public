
# coding: utf-8

# In[1]:

from profiler.core import *


# ## 0. Load Model

# In[2]:

from pgmpy.readwrite import BIFReader
reader = BIFReader("data/alarm.bif")
model = reader.get_model()
from pgmpy.sampling import BayesianModelSampling
inference = BayesianModelSampling(model)
data = inference.forward_sample(size=100000, return_type='dataframe')


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

pf.session.load_data(name='hospital', src=DF, df=data, check_param=True)


# ## 3. Load Training Data
# * multiplier: if set to None, will infer the minimal sample size; otherwise, it will create (# samples) * (# attributes) * (multiplier) training samples

# In[5]:

pf.session.load_training_data(difference=False)


# ## 4. Learn Structure
# * sparsity: intensity of L1-regularizer in inverse covariance estimation (glasso)
# * take_neg: if set to true, consider equal -> equal only

# In[53]:

autoregress_matrix = pf.session.learn_structure(sparsity=0.01, infer_order=False)


# * score: 
#     * "fit_error": mse for fitting y = B'X + c for each atttribute y 
#     * "training_data_fd_vio_ratio": the higher the score, the more violations of FDs in the training data. (bounded: \[0,1\])

# In[54]:

parent_sets = pf.session.get_dependencies(score="fit_error")


# In[55]:

# create ancester set
model_parents = {}
for node, children in model.edge.items():
    for child in list(children.keys()):
        if child not in model_parents:
            model_parents[child] = []
        model_parents[child].append(node)


# In[56]:

# create ancesters
def find_ancesters(node):
    a = []
    if node not in model_parents:
        return a
    for p in model_parents[node]:
        a.append(p)
        a.extend(find_ancesters(p))
    return a


# In[57]:

count = 0
tp = 0
for right in parent_sets:
    for parent in parent_sets[right]:
        count += 1
        if parent in find_ancesters(right):
            tp += 1
print("Precision: %.4f"%(tp / float(count)))


# ## 5. Visualization

# In[11]:

pf.session.visualize_covariance()


# In[12]:

pf.session.visualize_inverse_covariance()


# In[13]:

pf.session.visualize_autoregression()


# In[14]:

pf.session.timer.get_stat()


# In[ ]:




# In[ ]:



