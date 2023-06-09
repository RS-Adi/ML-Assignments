#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn import datasets
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as MpL

cps = datasets.fetch_openml(data_id=971)


mc1 = datasets.fetch_openml(data_id=1056)


cps.feature_names
cps.data
cps.target


mc1.feature_names
mc1.data
mc1.target


mytree = tree.DecisionTreeClassifier(criterion="entropy")

new_mytree = tree.DecisionTreeClassifier(criterion="entropy")



mytree.fit(cps.data, cps.target)


new_mytree.fit(mc1.data, mc1.target)




print(tree.export_text(mytree))

print(tree.export_text(new_mytree))



my_dt1 = tree.DecisionTreeClassifier(min_samples_leaf=410) 

cv = model_selection.cross_validate(my_dt1, cps.data, cps.target, scoring="roc_auc", cv=10,return_train_score=True)



my_dt2 = tree.DecisionTreeClassifier(min_samples_leaf=554) 

cv = model_selection.cross_validate(my_dt2, mc1.data, mc1.target, scoring="roc_auc", cv=10,return_train_score=True)



cv["test_score"].mean()

cv["train_score"].mean()





#Sub_task_1

from sklearn import model_selection
def Sub_task_1(cps):
    arr=[]
    arr1=[]
    cv=[]
    Ind=[10,20,30,40,50]
    for i in Ind:
        my_dt1 = tree.DecisionTreeClassifier(min_samples_leaf=i) 
        my_dt2 = tree.DecisionTreeClassifier(min_samples_leaf=i)
        cv = model_selection.cross_validate(my_dt1, cps.data, cps.target, scoring="roc_auc", cv=10,return_train_score=True)
        cv = model_selection.cross_validate(my_dt2, mc1.data, mc1.target, scoring="roc_auc", cv=10,return_train_score=True)
        arr.append(cv["test_score"].mean())
        arr1.append(cv["train_score"].mean())
    ind_max = 0
    ind_min = 0
    
    for i in range(len(arr)):
        AV = arr1[i] - arr[i]
        if(AV>arr1[ind_max] - arr[ind_max]):ind_max = i
        if(arr1[ind_min]> arr1[i]):ind_min = i
    MpL.annotate(text="Over_Fit", xy= (Ind[ind_max], arr1[ind_max]))
    MpL.annotate(text="Under_Fit", xy= (Ind[ind_min], arr1[ind_min]))
    MpL.axvline(x = Ind[ind_max], linestyle='dashed')
    MpL.axvline(x = Ind[ind_min], linestyle='dashed')

    
    MpL.plot(Ind, arr)
    MpL.plot(Ind, arr1)
    MpL.xlabel('Min Leaf Sample')
    MpL.ylabel('ROC - AUC Score')
    MpL.show()
    return cv
    





MpL.title("Data_Set_1")
Sub_task_1(cps) #Calling the function for Data_Set_1

MpL.title("Data_set_2")
Sub_task_1(mc1) #Recursively Calling the function again for Data_Set_2



parameters = [{"min_samples_leaf":[10,20,30,40,50]}]


mytree = tree.DecisionTreeClassifier(criterion="entropy")
new_mytree = tree.DecisionTreeClassifier(criterion="entropy")



#Sub_Task_2

my_dt1 = model_selection.GridSearchCV(mytree, parameters, scoring="roc_auc", cv=10,return_train_score=True)
my_dt2 = model_selection.GridSearchCV(new_mytree, parameters, scoring="roc_auc", cv=10,return_train_score=True)

my_dt1.fit(cps.data, cps.target)
my_dt2.fit(mc1.data, mc1.target)


my_dt1.cv_results_["mean_test_score"]
my_dt2.cv_results_["mean_test_score"]



#Data_Set_1
Sample_old = [10,20,30,40,50]
 
MpL.plot([10,20,30,40,50],my_dt1.cv_results_["mean_test_score"])
MpL.plot([10,20,30,40,50],my_dt1.cv_results_["mean_train_score"])
MpL.xlabel("Parameters")
MpL.ylabel("Mean_test_score")

#Heading for lines
MpL.annotate(text="Best_Params",xy=[my_dt1.best_params_["min_samples_leaf"],my_dt1.cv_results_["mean_train_score"][Sample_old.index(my_dt1.best_params_["min_samples_leaf"])]])
MpL.annotate(text="Best_Params",xy=[my_dt1.best_params_["min_samples_leaf"],my_dt1.cv_results_["mean_test_score"][Sample_old.index(my_dt1.best_params_["min_samples_leaf"])]])
MpL.axvline(x=my_dt1.best_params_["min_samples_leaf"])
MpL.show()


my_dt1.best_params_



#Data_Set_2
New_sample = [10,20,30,40,50]
 
MpL.plot([10,20,30,40,50],my_dt2.cv_results_["mean_test_score"])
MpL.plot([10,20,30,40,50],my_dt2.cv_results_["mean_train_score"])
MpL.xlabel("Parameters")
MpL.ylabel("Mean_test_score")

#Heading for lines
MpL.annotate(text="Best_Params",xy=[my_dt2.best_params_["min_samples_leaf"],my_dt2.cv_results_["mean_train_score"][New_sample.index(my_dt2.best_params_["min_samples_leaf"])]])
MpL.annotate(text="Best_Params",xy=[my_dt2.best_params_["min_samples_leaf"],my_dt2.cv_results_["mean_test_score"][New_sample.index(my_dt2.best_params_["min_samples_leaf"])]])
MpL.axvline(x=my_dt2.best_params_["min_samples_leaf"])
MpL.show()



my_dt2.best_params_


# In[ ]:




