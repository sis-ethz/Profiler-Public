
# coding: utf-8

# In[1]:

from profiler.core import *


# ## 1. Instantiate Engine
# * workers : number of processes
# * tol     : tolerance for differences when creating training data (set to 0 if data is completely clean)
# * eps     : error bound for inverse covariance estimation (since we use conservative calculation when determining minimum sample size, we recommend to set eps <= 0.01)
# * embedtxt: if set to true, differentiate b/w textual data and categorical data, and use word embedding for the former

# In[57]:

pf = Profiler(workers=2, tol=0, eps=0.05, embedtxt=False)


# ## 2. Load Data
# * name: any name you like
# * src: \[FILE; DF; DB (not implemented)\]
# * fpath: required if src == FILE
# * df: required if src == DF
# * check_param: print parameters used for data loading

# In[58]:

pf.session.load_data(name='hospital', src=FILE, fpath='data/TECHospital.csv', encoding="latin-1",
                     check_param=True, na_values='not available')


# ### 2.1 Change Data Types of Attributes
# * required input:
#     * a list of attributes
#     * a list of data types (must match the order of the attributes; can be CATEGORICAL, NUMERIC, TEXT, DATE)
# * optional input:
#     * a list of regular expression extractor

# In[59]:

pf.session.change_dtypes(['Provider ID', 'Phone Number', 'Score', 'Sample', 'Measure Start Date', 'Measure End Date'], 
                            [CATEGORICAL, CATEGORICAL, NUMERIC, NUMERIC, DATE, DATE])


# ### 2.2. Load/Train Embeddings for TEXT
# * path: path to saved/to-save embedding folder
# * load: set to true -- load saved vec from 'path'; set to false -- train locally
# * save: (only for load = False) save trained vectors to 'path'

# ## 3. Load Training Data
# * multiplier: if set to None, will infer the minimal sample size; otherwise, it will create (# samples) * (# attributes) * (multiplier) training samples

# In[60]:

pf.session.load_training_data(multiplier = None)


# ## 4. Learn Structure
# * sparsity: intensity of L1-regularizer in inverse covariance estimation (glasso)
# * take_neg: if set to true, consider equal -> equal only

# In[137]:

autoregress_matrix = pf.session.learn_structure(sparsity=0, infer_order=False, threshold=0)


# * score: 
#     * "fit_error": mse for fitting y = B'X + c for each atttribute y 
#     * "training_data_fd_vio_ratio": the higher the score, the more violations of FDs in the training data. (bounded: \[0,1\])

# In[138]:

parent_sets = pf.session.get_dependencies(score="fit_error")


# ### Evaluate FDs -- Precision

# In[139]:

import os, json
def read_fds(path='data/fds', f='TECHospital-hyfd'):
    all_fds = {}
    for line in open(os.path.join(path,f)):
        fd = json.loads(line)
        right = fd[u'dependant']['columnIdentifier']
        left = [l[u'columnIdentifier'] for l in fd[ u'determinant'][u'columnIdentifiers']]
        if right not in all_fds:
            all_fds[right] = set()
        all_fds[right].add(frozenset(left))
    return all_fds


# In[140]:

gt = read_fds()


# In[141]:

tp = 0
count = 0
for child in parent_sets:
    found = parent_sets[child]
    if len(found) == 0:
        continue
    count += 1
    match = False
    for parent in gt[child]:
        if set(parent).issubset(found):
            tp += 1
            match = True
            break
    if not match:
        print("{} -> {} is not valid".format(found, child))
    
print("Precision: %.4f"%(float(tp) / count))


# ### Evaluate FDs -- Recall

# In[142]:

def find_ancesters(node, dic):
    a = []
    if node not in dic:
        return a
    for p in dic[node]:
        a.append(p)
        a.extend(find_ancesters(p, dic))
    return a
def ancester_sets(dic):
    ancesters = {}
    for child in dic:
        ancesters[child] = find_ancesters(child, dic)
    return ancesters


# In[143]:

ancesters['Hospital Name']


# In[144]:

def get_neighbors(hm):
    neighbor = {}
    for i in hm:
        neighbor[i] = set(hm.columns.values[hm.loc[i, :] != 0]) - (set([i]))
    return neighbor


# In[145]:

neighbor_sets = get_neighbors(pf.session.struct_engine.inv_cov)


# In[146]:

count = 0
miss = 0
for child in neighbor_sets:
    found = neighbor_sets[child]
    for parent in gt[child]:
        count += 1
        if not set(parent).issubset(found):
            miss += 1
            print("{} -> {} is not found".format(parent, child))
print("Recall: %.4f"%(1 - float(miss) / count))


# In[147]:

pf.session.trans_engine.training_data.reset_index().groupby(['Measure Name', 'Hospital Name', 'Address', 'Footnote'])['index'].count()


# In[148]:

gt['Footnote']


# In[149]:

neighbor_sets['Footnote']


# In[150]:

ancesters = ancester_sets(parent_sets)
count = 0
miss = 0
for child in ancesters:
    found = parent_sets[child]
    for parent in gt[child]:
        count += 1
        if not set(parent).issubset(found):
            miss += 1
            print("{} -> {} is not found".format(parent, child))
print("Recall: %.4f"%(1 - float(miss) / count))


# ## 5. Visualization

# In[151]:

pf.session.visualize_covariance()


# In[152]:

pf.session.visualize_inverse_covariance()


# In[153]:

pf.session.visualize_autoregression()


# In[11]:

pf.session.timer.get_stat()


# In[ ]:




# In[ ]:



