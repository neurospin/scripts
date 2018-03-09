
import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from nibabel import gifti
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import recall_score
from sklearn import svm, metrics, linear_model
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import  make_scorer,accuracy_score,recall_score,precision_score
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn import grid_search
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif


##############################################################################

age = pop_volume["age"].values
sex = pop_volume["sex"].values
df = pd.DataFrame()
df["site"] = pop_volume["site"].values
df["labels"] = np.nan
df["labels"][y_all==1] = labels_all_scz
df["labels"][y_all==0] = "controls"
LABELS_DICT = {"controls":"controls",0: "cluster 1", 1: "cluster 2",2: "cluster 3"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)

labels = df["labels"].values
labels_name = df["labels_name"].values


#1) CONTROLS vs CLUSTER 1
X = features[np.logical_or(labels_name =="controls",labels_name =="cluster 1") ,:]
y = y_all[np.logical_or(labels_name =="controls",labels_name =="cluster 1")]
assert sum(y==0) == sum(labels_name=="controls") == 314
assert sum(y==1) == sum(labels_name=="cluster 1") == 77
assert X.shape[0] == y.shape[0] == 391
bacc,recall,auc,beta1 = svm_score(X,y)
print(" AUC: %s, Balanced acc: %s, Sen: %s, Spe : %s"%(auc, bacc,recall[0],recall[1]))
#Balanced acc: 0.78306725122, Sen: 0.605095541401, Spe : 0.961038961039


#2) CONTROLS vs CLUSTER 2
X = features[np.logical_or(labels_name =="controls",labels_name =="cluster 2") ,:]
y = y_all[np.logical_or(labels_name =="controls",labels_name =="cluster 2")]
assert sum(y==0) == sum(labels_name=="controls") == 314
assert sum(y==1) == sum(labels_name=="cluster 2") == 99
assert X.shape[0] == y.shape[0] == 413
bacc,recall,auc,beta2 = svm_score(X,y)
print(" AUC: %s, Balanced acc: %s, Sen: %s, Spe : %s"%(auc, bacc,recall[0],recall[1]))

#Balanced acc: 0.572138583285, Sen: 0.53821656051, Spe : 0.606060606061

#3) CONTROLS vs CLUSTER 3
X = features[np.logical_or(labels_name =="controls",labels_name =="cluster 3") ,:]
y = y_all[np.logical_or(labels_name =="controls",labels_name =="cluster 3")]
assert sum(y==0) == sum(labels_name=="controls") == 314
assert sum(y==1) == sum(labels_name=="cluster 3") == 77
assert X.shape[0] == y.shape[0] == 391
bacc,recall,auc,beta3 = svm_score(X,y)
print(" AUC: %s, Balanced acc: %s, Sen: %s, Spe : %s"%(auc, bacc,recall[0],recall[1]))
#Balanced acc: 0.835987261146, Sen: 0.671974522293, Spe : 1.0


#5) CONTROLS vs ALL SCZ
X = features
y = y_all
assert sum(y==0) == sum(labels_name=="controls") == 314
assert sum(y==1)  == 253
assert X.shape[0] == y.shape[0] == 567
bacc,recall,auc,beta4 = svm_score(X,y)
print(" AUC: %s, Balanced acc: %s, Sen: %s, Spe : %s"%(auc, bacc,recall[0],recall[1]))
#Balanced acc: 0.677884494348, Sen: 0.671974522293, Spe : 0.683794466403

##############################################################################
#Print barplot of SVM weight
df = pd.DataFrame()
df["Classifier"] = ["all vs HC","all vs SCZ 1","all vs SCZ 2","all vs SCZ 3"]
df["temporal_thickness"] = [beta1[0,0],beta2[0,0],beta3[0,0],beta4[0,0]]
df["frontal_thickness"] = [beta1[0,1],beta2[0,1],beta3[0,1],beta4[0,1]]
df["hippocampus volume"] = [beta1[0,2],beta2[0,2],beta3[0,2],beta4[0,2]]
df["amygdala volume"] = [beta1[0,3],beta2[0,3],beta3[0,3],beta4[0,3]]
df["thalamus volume"] = [beta1[0,4],beta2[0,4],beta3[0,4],beta4[0,4]]

#PLOT WEIGHTS OF PC FOR EACH CLUSTER
features_of_interest_name =  ['temporal_thickness',"frontal_thickness",\
               'hippocampus volume','amygdala volume','thalamus volume']
df = pd.DataFrame()
df["Feature"] = np.hstack((features_of_interest_name,features_of_interest_name,features_of_interest_name,\
features_of_interest_name))
df["score"] = np.hstack((beta1,beta2,beta3,beta4))[0,:]
df["Classifier"] = np.hstack((np.repeat("all SCZ vs HC",5),np.repeat("all vs SCZ 1",5),\
np.repeat("all vs SCZ 2",5),np.repeat("all vs SCZ 3",5)))

sns.set_style("whitegrid")
ax = sns.factorplot(x="Feature", y="score",data=df, kind="bar",\
                    col="Classifier",\
                    col_order=["all SCZ vs HC","all vs SCZ 1","all vs SCZ 2","all vs SCZ 3"],\
                   palette="binary")
ax.set_titles("{col_name}",size=15)
ax.set_xlabels("Features")

##############################################################################

def svm_score(X,y):

    list_predict=list()
    list_true=list()
    list_prob_pred=list()

    clf= svm.LinearSVC(fit_intercept=False,class_weight='balanced')
    parameters={'C':[1,1e-1,1e1]}

    def balanced_acc(t, p):
        recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
        ba = recall_scores.mean()
        return ba


    score=make_scorer(balanced_acc,greater_is_better=True)
    clf = grid_search.GridSearchCV(clf,parameters,cv=3,scoring=score)
    skf = StratifiedKFold(y,5)

    for train, test in skf:

        X_train=X[train,:]
        X_test=X[test,:]
        y_train=y[train]
        y_test=y[test]
        list_true.append(y_test.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_prob_pred = clf.decision_function(X_test)
        list_predict.append(y_pred)
        list_prob_pred.append(y_prob_pred)

    t=np.concatenate(list_true)
    prob_pred=np.concatenate(list_prob_pred)
    p=np.concatenate(list_predict)
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    balanced_acc= recall_scores.mean()
    auc = roc_auc_score(t,prob_pred)
    clf= svm.LinearSVC(fit_intercept=False,class_weight='balanced')
    clf.fit(X,y)
    beta = clf.coef_
    return balanced_acc,recall_scores,auc,beta