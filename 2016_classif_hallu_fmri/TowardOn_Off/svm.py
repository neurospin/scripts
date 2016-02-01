# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:03:22 2015

@author: ad247405
"""


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


BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"
INPUT_CSV = os.path.join(BASE_PATH,"toward_on", "patients.csv")
pop = pd.read_csv(INPUT_CSV)
periods = pd.read_csv(os.path.join(BASE_PATH,"results", "nperiods.csv"))


 #Retreive variables
#############################################################################
T=np.load(os.path.join(BASE_PATH,'toward_on','svm','T.npy'))
betas=np.load(os.path.join(BASE_PATH,'toward_on','svm','betas.npy'))
y=np.load(os.path.join(BASE_PATH,'toward_on','svm','y.npy'))
subject=np.load(os.path.join(BASE_PATH,'toward_on','svm','subject.npy'))
#############################################################################
selector = SelectPercentile(f_classif, percentile=1)
selector.fit_transform(T,y.ravel())
plt.hist(selector.pvalues_,10,normed=True)


Tc=T-np.mean(T,axis=0)

#SVM & Leave one subject-out - no feature selection
#############################################################################

n=0
list_predict=list()
list_true=list()
coef=np.zeros((24,sum(mask_bool)))
clf = svm.LinearSVC(C=10e-7,fit_intercept=False,class_weight='auto')

for i in range(1,24):
    test_bool=(subject==i)
    train_bool=(subject!=i)
    Xtest=T[test_bool,:]
    ytest=y[test_bool]
    Xtrain=T[train_bool,:]
    ytrain=y[train_bool]
    list_true.append(ytest.ravel())
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest=scaler.transform(Xtest)
    clf.fit(Xtrain, ytrain.ravel())
    coef[n,:]=clf.coef_
    pred=(clf.predict(Xtest))
    list_predict.append(pred)
    print n 
    n=n+1 
    
#############################################################################
#Save weights and std of SVM coef
std_coef=coef.std(axis=0)
mean_coef=coef.mean(axis=0)
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = mean_coef
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"toward_on","svm","coef_mean.nii.gz"))
    
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

    
    
    
    
    
    