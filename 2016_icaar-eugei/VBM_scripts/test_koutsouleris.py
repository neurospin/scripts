# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:46:29 2016

@author: ad247405
"""


import sklearn
import nilearn.signal
import re
import glob
import os
import nibabel as nibabel
import numpy as np
import os
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn import grid_search, metrics
import brainomics.image_atlas
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from mulm import MUOLS
import scipy
from scipy import stats 
from sklearn.preprocessing import StandardScaler  
from sklearn import svm, metrics, linear_model, decomposition    

BASE_PATH= '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR'
INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
mask = nibabel.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/mask.nii')
mask_bool = mask.get_data() !=0

INPUT_CSV_ICAAR = '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/population.csv'
pop = pd.read_csv(INPUT_CSV_ICAAR)
##################################################################################
pop = pd.read_csv(INPUT_CSV_ICAAR)

#############################################################################
 #Retreive variables
#############################################################################
X = np.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/X.npy')
y = np.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/y.npy')


number_subjects = X.shape[0]
number_features = X.shape[1]


#Correct for age, gender, center and handedness (Residualization) 


age=pop[~np.isnan(pop['group_outcom.num'])]["age"]
sex=pop[~np.isnan(pop['group_outcom.num'])]["sex.num"]
cov = np.vstack((np.asarray(age),np.asarray(sex))).T


pred=linear_model.LinearRegression().fit(cov,features).predict(cov)
res=features-pred

X=res




#SVM & Leave one subject-out - no feature selection
#############################################################################

C_range = [100,10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
N_FOLDS = 5

for c in C_range:   
    n=0
    list_predict=list()
    list_true=list()
    coef=np.zeros((N_FOLDS,number_features - 3))
    #clf = svm.LinearSVC(C=c,fit_intercept=False,class_weight='auto')
    clf= svm.SVC(C=c,class_weight = 'auto',gamma = 1e-1)    
    skf = StratifiedKFold(n_folds=N_FOLDS,y=y)
    pca = sklearn.decomposition.PCA(n_components=20)
    for train_index, test_index in skf:
        X_test = X[test_index,:]
        X_train = X[train_index,:]
        y_test = y[test_index]
        y_train = y[train_index]
        list_true.append(y_test.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test=scaler.transform(X_test)
        pca_train = pca.fit_transform(X_train)   
        pca_test = pca.transform(X_test)  
        clf.fit(pca_train, y_train.ravel())
        pred=(clf.predict(pca_test))
        list_predict.append(pred)
        print n 
        n=n+1 

    #############################################################################
       
     #Display scores of classification (accuracy, precision and recall)
    #############################################################################   
    
    t=np.concatenate(list_true)
    p=np.concatenate(list_predict)
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    acc=metrics.accuracy_score(t,p)
    pre=recall_scores[0]
    rec=recall_scores[1]
    print acc
    print pre
    print rec
#############################################################################
#############################################################################   
