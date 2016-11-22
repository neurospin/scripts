# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 17:39:44 2016

@author: ad247405
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:56:36 2016

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

BASE_PATH= '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR+EUGEI'
INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
mask = nibabel.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR+EUGEI/mask.nii')
mask_bool = mask.get_data() !=0


INPUT_CSV_ICAAR_EUGEI = '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR+EUGEI/population.csv'
pop = pd.read_csv(INPUT_CSV_ICAAR_EUGEI)
OUTPUT =  '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR+EUGEI/svm_residualized/no_model_selection'
out_scores_file = '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR+EUGEI/svm_residualized/no_model_selection/svm_scores.txt'
###################################################################################

#covariate_cohort = np.zeros((59,1))
#covariate_cohort[41:,0]=1
#covariate_cohort = covariate_cohort - covariate_cohort.mean()
#X = np.hstack([covariate_cohort,X])
#

#############################################################################
 #Retreive variables
#############################################################################
X = np.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR+EUGEI/X.npy')
y = np.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR+EUGEI/y.npy')

features = X[:,3:]

#Residualize to correct for site effect
#############################################################################
from sklearn import linear_model
site = pop[~np.isnan(pop['group_outcom.num'])]["cohort.num"]
age=pop[~np.isnan(pop['group_outcom.num'])]["age"]
sex=pop[~np.isnan(pop['group_outcom.num'])]["sex.num"]
cov = np.vstack((np.asarray(site)))
pred=linear_model.LinearRegression().fit(cov,features).predict(cov)
res=features-pred
X=res
#############################################################################





number_subjects = X.shape[0]
number_features = X.shape[1]
#SVM & Leave one subject-out - no feature selection
#############################################################################
outf=open(out_scores_file, "w")
outf.write("C"+" "+"accuracy"+" "+"sensitivity"+" "+"specificity"+" "+"weights_map_path\n")
outf.flush()


C_range = [100,10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
N_FOLDS = 5

for c in C_range:   
    n=0
    list_predict=list()
    list_true=list()
    coef=np.zeros((N_FOLDS,number_features))
    clf = svm.LinearSVC(C=c,fit_intercept=False,class_weight='auto')
    skf = StratifiedKFold(n_folds=N_FOLDS,y=y)
    for train_index, test_index in skf:
        X_test = X[test_index,:]
        X_train = X[train_index,:]
        y_test = y[test_index]
        y_train = y[train_index]
        list_true.append(y_test.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test=scaler.transform(X_test)
        clf.fit(X_train, y_train.ravel())
        coef[n,:] = clf.coef_
        pred=(clf.predict(X_test))
        list_predict.append(pred)
        print n 
        n=n+1 


    #############################################################################
    #Save weights and std of SVM coef
    std_coef=coef.std(axis=0)
    mean_coef=coef.mean(axis=0)
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = mean_coef
    out_im = nibabel.Nifti1Image(arr, affine=mask.get_affine())
    filename = os.path.join(OUTPUT,"coef_mean_c=%r.nii.gz")%c
    out_im.to_filename(filename)
        
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

    outf.write(str(c)+" "+ str(acc)+" "+str(pre)+" "+str(rec)+" "+ filename+"\n")

    outf.flush()

outf.close()

