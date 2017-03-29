# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:45:20 2016

@author: ad247405
"""

import numpy as np
import os
import re
import glob
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn import grid_search
import brainomics.image_atlas
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from mulm import MUOLS
import random
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import recall_score
from sklearn import svm, metrics, linear_model
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import  make_scorer,accuracy_score,recall_score,precision_score


##############################################################################
##############################################################################
#Import csv file
BASE_PATH = "/neurospin/brainomics/2015_bd_sulci"
INPUT_CSV = os.path.join(BASE_PATH,"SVM_base_export.csv")
pop = pd.read_csv(INPUT_CSV,sep = ';',decimal=',')


#replace missing value by NaN
features = (list(pop.columns.values))
for i in range(8,len(features)):
    f = (features[i])
    pop[f].replace(0,NaN, inplace=True)


#drop rows and colums when too many missing values (if more than 75% missing, then drop)
pop.dropna(axis=0,thresh=732*0.75,inplace=True)
pop.dropna(axis=1,thresh=545*0.75,inplace=True)

#impute missing values with mean 
features = (list(pop.columns.values))
for i in range(8,len(features)):
    f = (features[i])
    pop[f].fillna(pop[f].mean(),inplace=True)
    
    
##############################################################################     
     
#Correct for age, gender, center and handedness (Residualization) 

import patsy
cov=patsy.dmatrix('AGEATMRI + C(SEX) + C(SCANNER)+ C(HANDEDNESS)', data=pop)
cov=np.asarray(cov)
data=np.asarray(pop)
sulci=data[:,8:]

pred=linear_model.LinearRegression().fit(cov,sulci).predict(cov)
res=sulci-pred

##############################################################################   
#SVM classification with Stratified KFold BD vs HC 
    

y=data[:,1].astype(int)    
X=res


acc=list()
list_predict=list()
list_true=list()


def balanced_acc(t, p):
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    ba = recall_scores.mean()
    return ba


#L1 SFVM clf
#clf= svm.LinearSVC(fit_intercept=False,class_weight='auto',dual=False,penalty='l1') 
#L2 SVM clf:
clf= svm.LinearSVC(fit_intercept=False,class_weight='auto') 
parameters={'C':[10e-6,10e-4,10e-3,10e-2,1,10e2,10e4]}

score=make_scorer(balanced_acc,greater_is_better=True)
clf = grid_search.GridSearchCV(clf,parameters,cv=3,scoring=score)
skf = StratifiedKFold(y,10)

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
    list_predict.append(y_pred)

    
t=np.concatenate(list_true)
p=np.concatenate(list_predict)
recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
acc=metrics.accuracy_score(t,p)
balanced_acc= recall_scores.mean()
metrics.roc_auc_score(t,p)


   
 ##############################################################################
#SVM classification (L2 penalty) with Stratified KFold with Univariate Features selection
  
y=data[:,1].astype(int)    
X=res


def balanced_acc(t, p):
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    ba = recall_scores.mean()
    return ba


scores=np.zeros((3,100))
acc=list()
for f in range(1,100):
    n=0
    list_predict=list()
    list_true=list()
    
    #L1 SFVM clf
    #clf= svm.LinearSVC(fit_intercept=False,class_weight='auto',dual=False,penalty='l1') 
    #L2 SVM clf:
    clf= svm.LinearSVC(fit_intercept=False,class_weight='auto') 
    parameters={'C':[10e-6,10e-4,10e-3,10e-2,1,10e2,10e4]}
    score=make_scorer(balanced_acc,greater_is_better=True)
    clf = grid_search.GridSearchCV(clf,parameters,cv=3,scoring=score)
    skf = StratifiedKFold(y,10)

    for train, test in skf:
        
        X_train=X[train,:]
        X_test=X[test,:]
        y_train=y[train]
        y_test=y[test] 
        list_true.append(y_test.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
        selector = SelectPercentile(f_classif, percentile=f)
        clf.fit(selector.fit_transform(X_train,y_train.ravel()), y_train.ravel())
        X_test=selector.transform(X_test)
        y_pred = clf.predict(X_test)
        list_predict.append(y_pred)
        
    t=np.concatenate(list_true)
    p=np.concatenate(list_predict)
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    scores[0,f]=recall_scores.mean()
    scores[1,f]=recall_scores[0]
    scores[2,f]=recall_scores[1]   
    metrics.roc_auc_score(t,p)

     
    print f
                                     
import matplotlib.pyplot as plt
plt.plot(scores[0,:],label="Accuracy")
plt.plot(scores[1,:],label="Specificity")
plt.plot(scores[2,:],label="Sensitivity")
plt.xlabel('Percentiles')
plt.ylabel('Scores')
plt.legend(loc='lower right')




#############################################################################
#############################################################################
#Test with permutations the significance of a classification score
# The p-value is then given by the percentage of runs for which the score
#obtained is greater than the classification score obtained in the first place

nperms=1000
scores_perm = list()
recall_perm = list()

#true_acc= acc

for n in xrange(nperms):

#BS with PF vs w/o PF    
    data=np.asarray(pop)
    covariate=data[:,3:7]
    existingData=data[:,2]!=' '
    existingData.astype(int)
    y=data[existingData,2].astype(float)
    y = np.random.permutation(y)
    X=data[existingData,3:].astype(float)

# BD vs HC
#    data=np.asarray(pop)
#    covariate=data[:,2:7]
#    y=data[:,1].astype(int)    
#    y = np.random.permutation(y)
#    X=data[:,2:].astype(float)


    acc=list()
    list_predict=list()
    list_true=list()
    clf= svm.LinearSVC(C=1,fit_intercept=False,class_weight='auto') 
    skf = StratifiedKFold(y, 10)

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
        list_predict.append(y_pred)
    
    t=np.concatenate(list_true)
    p=np.concatenate(list_predict)
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    acc=metrics.accuracy_score(t,p)
    scores_perm.append(acc)
    recall_perm.append(recall_score(t,p,pos_label=None, average=None,labels=[0,1]))
    print acc
   
      
scores_perm=np.array(scores_perm)
pval=np.sum(scores_perm >=0.63)/float(nperms)
print pval

recall_perm=np.array(recall_perm)
spe=recall_perm[:,0]
sen=recall_perm[:,1]
pval=np.sum(spe >=0.6)/float(nperms)
print pval
pval=np.sum(sen >=0.66)/float(nperms)
print pval


plt.hist(scores_perm, 10, label='Permutation scores')
plt.plot(2 * [0.63],plt.ylim(),'--g', linewidth=3)
plt.xlabel('Accuracy')

plt.hist(sen, 10, label='Permutation scores')
plt.plot(2 * [0.6],plt.ylim(),'--g', linewidth=3)
plt.xlabel('Specificity')

plt.hist(sen, 10, label='Permutation scores')
plt.plot(2 * [0.66],plt.ylim(),'--g', linewidth=3)
plt.xlabel('Sensitivity')
