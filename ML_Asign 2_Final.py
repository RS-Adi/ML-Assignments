#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib as mpl


# In[2]:


cho = datasets.fetch_openml(data_id=688)


# In[3]:


cho.data.info()


# In[4]:


cho.data


# In[5]:


cho.target


# In[6]:


from sklearn.compose import ColumnTransformer 


# In[7]:


ohe = OneHotEncoder(sparse=False)


# In[8]:


CT = ColumnTransformer([("encoder",
OneHotEncoder(sparse=False), [3])],
remainder="passthrough")


# In[9]:


new_data = CT.fit_transform(cho.data)


# In[10]:


CT.get_feature_names_out()


# In[11]:


type(new_data)


# In[12]:


cho_new_data = pd.DataFrame(new_data, columns =CT.get_feature_names_out(), index = cho.data.index)


# In[13]:


cho_new_data.info()


# In[14]:


#Linear Regressor
from sklearn.linear_model import LinearRegression


# In[15]:


LR = LinearRegression()


# In[16]:


from sklearn.model_selection import cross_validate


# In[17]:


scores = cross_validate(LR, cho_new_data, cho.target,cv=10, scoring="neg_root_mean_squared_error")


# In[18]:


scores["test_score"]


# In[19]:


rmse = 0-scores["test_score"]


# In[20]:


rmse


# In[21]:


rmse.mean()


# In[22]:


from sklearn.ensemble import BaggingRegressor


# In[23]:


bagged_LR = BaggingRegressor(LinearRegression())


# In[24]:


bg_scores = cross_validate(bagged_LR, cho_new_data, cho.target, cv=10, scoring="neg_root_mean_squared_error")


# In[25]:


bg_scores["test_score"]


# In[26]:


rmse_bg = 0-bg_scores["test_score"]


# In[27]:


rmse_bg


# In[28]:


rmse_bg.mean()


# In[29]:


from sklearn.ensemble import AdaBoostRegressor


# In[30]:


boosted_LR = AdaBoostRegressor(LinearRegression())


# In[31]:


bo_scores = cross_validate(boosted_LR, cho_new_data, cho.target, cv=10, scoring="neg_root_mean_squared_error")


# In[32]:


bo_scores["test_score"]


# In[33]:


bo_rmse = 0-bo_scores["test_score"]


# In[34]:


bo_rmse


# In[35]:


bo_rmse.mean()


# In[36]:


#Decision Tree (Regular)

DT = DecisionTreeRegressor()


# In[37]:


tuned_DT = model_selection.GridSearchCV(DT,[{'min_samples_leaf':[8,9,10,11]}], scoring="neg_root_mean_squared_error", cv = 10)


# In[38]:


DT_scores = cross_validate(tuned_DT, cho_new_data, cho.target, cv = 10, scoring="neg_root_mean_squared_error")


# In[39]:


DT_scores["test_score"]
DT_rmse = 0-DT_scores["test_score"]
DT_rmse


# In[40]:


bagged_DT = BaggingRegressor(DecisionTreeRegressor())


# In[41]:


bg_scores = cross_validate(bagged_DT, cho_new_data, cho.target, cv = 10, scoring="neg_root_mean_squared_error")


# In[42]:


bg_dt_rmse = 0-bg_scores["test_score"]
bg_dt_rmse


# In[43]:


bg_dt_rmse.mean()


# In[44]:


boosted_DT = AdaBoostRegressor(DecisionTreeRegressor())


# In[45]:


bo_scores = cross_validate(boosted_DT, cho_new_data, cho.target, cv = 10, scoring="neg_root_mean_squared_error")


# In[46]:


bo_dt_rmse = 0-bo_scores["test_score"]
bo_dt_rmse


# In[47]:


bo_dt_rmse.mean()


# In[48]:


#K-Nearest Neighbours Method
from sklearn.neighbors import KNeighborsRegressor


# In[49]:


knn = KNeighborsRegressor()


# In[50]:


parameters = [{"n_neighbors":[8,9,10,11,13]}]


# In[51]:


tuned_knn = GridSearchCV(KNeighborsRegressor(), parameters,scoring = "neg_root_mean_squared_error", cv = 5)


# In[52]:


knn_scores = cross_validate(tuned_knn,cho_new_data, cho.target, cv = 10, scoring="neg_root_mean_squared_error")


# In[53]:


knn_scores["test_score"]
knn_rmse = 0-knn_scores["test_score"]
knn_rmse


# In[54]:


knn_rmse.mean()


# In[55]:


bagged_knn = BaggingRegressor(KNeighborsRegressor())


# In[56]:


bg_knn_scores = cross_validate(bagged_knn, cho_new_data, cho.target, cv = 10, scoring = "neg_root_mean_squared_error")


# In[57]:


bg_knn_rmse = 0-bg_knn_scores["test_score"]
bg_knn_rmse


# In[58]:


bg_knn_rmse.mean()


# In[59]:


boosted_knn = AdaBoostRegressor(KNeighborsRegressor())


# In[60]:


bo_knn_scores = cross_validate(boosted_knn, cho_new_data, cho.target, cv = 10, scoring="neg_root_mean_squared_error")


# In[61]:


bo_knn_rmse = 0-bo_knn_scores["test_score"]
bo_knn_rmse


# In[62]:


bo_knn_rmse.mean()


# In[63]:


#Suppport_Vector Machines
from sklearn.svm import SVR


# In[64]:


svr = SVR()
svr_scores = cross_validate(svr, cho_new_data, cho.target, cv = 10, scoring = "neg_root_mean_squared_error")


# In[65]:


svr_rmse = 0-svr_scores["test_score"]
svr_rmse


# In[66]:


svr_rmse.mean()


# In[67]:


bagged_svr = BaggingRegressor(SVR())


# In[68]:


bg_svr_scores = cross_validate(bagged_svr, cho_new_data, cho.target, cv = 10, scoring="neg_root_mean_squared_error")


# In[69]:


bg_svr_rmse = 0-bg_svr_scores["test_score"]
bg_svr_rmse


# In[70]:


bg_svr_rmse.mean()


# In[71]:


boosted_svr = AdaBoostRegressor(SVR())


# In[72]:


bo_svr_scores = cross_validate(boosted_svr, cho_new_data, cho.target, cv = 10, scoring = "neg_root_mean_squared_error")


# In[73]:


bo_svr_rmse = 0-bo_svr_scores["test_score"]
bo_svr_rmse


# In[74]:


bo_svr_rmse.mean()


# In[75]:


from sklearn.ensemble import VotingRegressor


# In[76]:


vr = VotingRegressor([("LR", LinearRegression()), ("svr", SVR()), ("DT", DecisionTreeRegressor()), ("knn", KNeighborsRegressor())])


# In[88]:


vr_scores = cross_validate(vr, cho_new_data, cho.target, cv = 10, scoring = "neg_root_mean_squared_error")


# In[89]:


vr_rmse = 0-vr_scores["test_score"]
vr_rmse


# In[90]:


vr_rmse.mean()


# In[78]:


#P values
from scipy.stats import ttest_rel


# In[79]:


#LR
ttest_rel(rmse, rmse_bg)


# In[80]:


#DT
ttest_rel(DT_rmse,bg_dt_rmse)


# In[81]:


#KNN
ttest_rel(knn_rmse,bg_knn_rmse)


# In[82]:


#SVR
ttest_rel(svr_rmse,bg_svr_rmse)


# In[83]:


#LRbo
ttest_rel(rmse, bo_rmse)


# In[84]:


#DTbo
ttest_rel(DT_rmse,bo_dt_rmse)


# In[85]:


#KNNbo
ttest_rel(knn_rmse,bo_knn_rmse)


# In[86]:


#SVRbo
ttest_rel(svr_rmse,bo_svr_rmse)


# In[91]:


#Voting Decision tree
ttest_rel(vr_rmse, DT_rmse)


# In[92]:


#VT DT
ttest_rel(DT_rmse, DT_rmse)


# In[95]:


#vt Lr
ttest_rel(DT_rmse, rmse)


# In[96]:


#vr knn
ttest_rel(DT_rmse, knn_rmse)


# In[97]:


#vr svr
ttest_rel(DT_rmse, svr_rmse)


# In[ ]:




