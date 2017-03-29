# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:29:08 2015

@author: ad247405
"""

import nilearn.signal
import re
import glob
import os
import nibabel as nibabel
import numpy as np
import nilearn.image
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
import brainomics.image_atlas
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from mulm import MUOLS    
import csv
from sklearn.metrics import roc_auc_score, recall_score
 

#############################################################################
#Mask on resampled Images (We use intecept between Harvard/Oxford cort/sub mask and MNI152linT1 mask)
BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"
ref=os.path.join(BASE_PATH,"atlases","MNI152lin_T1_3mm_brain_mask.nii.gz")
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(ref=ref
,output=(os.path.join(BASE_PATH,"results","mask.nii.gz")),smooth_size=None,dilation_size=None)
a=babel_mask_atlas.get_data()
babel_mask=nibabel.load(ref)
b=babel_mask.get_data()
b[a==0]=0
mask_bool=b!=0
#############################################################################
     
#Compute Tstats image for each block 
#############################################################################

T=np.zeros((128,63966))
betas=np.zeros((128,63966))

subject=np.zeros((128))  
subject_num=0
t=0

from nipy.modalities.fmri import hemodynamic_models 
hrf=hemodynamic_models.spm_hrf(tr=3.0,oversampling=1,onset=0.0,time_length=400)

DesignMat = np.zeros((132,8)) 
DesignMat[20:27:, 0] = np.arange(1, 8)
DesignMat[34:41:, 1] = np.arange(1, 8)
DesignMat[48:55, 2] = np.arange(1, 8)
DesignMat[62:69, 3] = np.arange(1, 8)
DesignMat[76:83, 4] = np.arange(1, 8)
DesignMat[90:97, 5] = np.arange(1, 8)
DesignMat[104:111, 6] = np.arange(1, 8)
DesignMat[118:125, 7] = np.arange(1, 8)
  

lab = np.zeros((132)) 
lab[20:27] = 1
lab[34:41] = 1
lab[48:55] = 1
lab[62:69] = 1
lab[76:83] = 1
lab[90:97] = 1
lab[104:111] = 1
lab[118:125] = 1  
#Convolve regressor with HRF          
for i in range(0,8):
    DesignMat[:,i]=np.convolve(DesignMat[:,i],hrf)[0:132]
                   
               
BASE_PATH ="/neurospin/brainomics/2016_classif_hallu_fmri/data/DATA_Localizer/Sujets_sains"
                                   
                
for i in range(1,20):
    imagefile_pattern = 'Sujet'+str(i)+'_'
    for file in os.listdir(BASE_PATH):
        if re.match(imagefile_pattern, file):
            name= file
            print(name)
            path=os.path.join(BASE_PATH,name,'SPM')
            subject_num=subject_num + 1
            images = list()
            pathlist=list()
            all_scans=sorted(glob.glob(os.path.join(path,'*swraf*')))
           
            for imagefile_name in all_scans:
               pathlist.append(imagefile_name)
               babel_image = nibabel.load(imagefile_name)
               #babel_image=nilearn.image.smooth_img(imgs=babel_image, fwhm=6)
               babel_image=nilearn.image.resample_img(babel_image, target_affine=babel_image.get_affine()*2, target_shape=[ x /2 for x in babel_image.shape], interpolation='continuous', copy=True, order='F')
               images.append(babel_image.get_data().ravel())
            
            X = np.vstack(images)
            X = X[:, mask_bool.ravel()] 
            X=nilearn.signal.clean(X,detrend=True,standardize=True,confounds=None,low_pass=None, high_pass=None, t_r=3)
            
          
#            X=X-X.mean(axis=0)             
#            
            muols = MUOLS(Y=X,X=DesignMat)
            muols.fit()
            contrasts=np.identity((8))
            tvals, pvals, dfs = muols.t_test(contrasts,pval=True, two_tailed=True)
            for m in range (0,8):
                 T[t,:]=tvals[m]
                 betas[t,:]=muols.coef[m,:]
                 subject[t]=subject_num
                 t=t+1
                 print t
                 
BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"                 
y=np.zeros((128))                
np.save(os.path.join(BASE_PATH,'toward_on','svm_with_HC','T_IMA.npy'),T)
np.save(os.path.join(BASE_PATH,'toward_on','svm_with_HC','betas_IMA.npy'),betas)
np.save(os.path.join(BASE_PATH,'toward_on','svm_with_HC','y_IMA.npy'),y)
np.save(os.path.join(BASE_PATH,'toward_on','svm_with_HC','subject_IMA.npy'),subject) 
                   

#Retrieve variables
#############################################################################
#############################################################################
T=np.load(os.path.join(BASE_PATH,'toward_on','svm','T.npy'))
b=np.load(os.path.join(BASE_PATH,'toward_on','svm','betas.npy'))
y=np.load(os.path.join(BASE_PATH,'toward_on','svm','y.npy'))
subject=np.load(os.path.join(BASE_PATH,'toward_on','svm','subject.npy'))

T_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','T_IMA.npy'))
b_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','betas_IMA.npy'))
y_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','y_IMA.npy'))
subject_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','subject_IMA.npy'))
#   
 
Tdiff=np.mean(T_IMA,axis=0)-np.mean(T[y==0],axis=0)
T_IMA_diff=T_IMA-Tdiff

 
#SVM & Leave one subject-out - no feature selection - WITH IMA samples###########################################################################

#Use betas in the classifier

#bdiff=np.mean(b_IMA,axis=0)-np.mean(b[y==0],axis=0)
#b_IMA_diff=b_IMA-bdiff
#T_IMA_diff=b_IMA_diff
#T=b


n=0
list_predict=list()
list_true=list()
coef=np.zeros((23,63966))
#coef=np.zeros((24,8028))
clf = svm.LinearSVC(C=10e-4,fit_intercept=True,class_weight='auto')

for i in range(1,24):
    test_bool=(subject==i)
    train_bool=(subject!=i)
    Xtest=T[test_bool,:]
    ytest=y[test_bool]
    Xtrain=np.vstack((T_IMA_diff,T[train_bool,:]))
    ytrain=np.hstack((y_IMA,y[train_bool]))
#    Xtrain=T[train_bool,:]
#    ytrain=y[train_bool]
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
    

t=np.concatenate(list_true)
p=np.concatenate(list_predict)
recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
acc=metrics.accuracy_score(t,p)
auc = roc_auc_score(t,p)
pre=recall_scores[0]
rec=recall_scores[1]
print acc
print auc
print pre
print rec

np.save(os.path.join(BASE_PATH,'toward_on','svm_with_HC','betas_24.npy'),coef)
#############################################################################
#Save weights and std of SVM coef
std_coef=coef.std(axis=0)
mean_coef=coef.mean(axis=0)
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = mean_coef
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"toward_on","svm_with_HC","beta_mean.nii"))


############################################################################
def dice_bar(thresh_comp):
    """Given an array of thresholded component of size n_voxels x n_folds,
    compute the average DICE coefficient.
    """
    n_voxels, n_folds = thresh_comp.shape
    # Paire-wise DICE coefficient (there is the same number than
    # pair-wise correlations)
    n_corr = n_folds * (n_folds - 1) / 2
    thresh_comp_n0 = thresh_comp != 0
    # Index of lines (folds) to use
    ij = [[i, j] for i in xrange(n_folds) for j in xrange(i + 1, n_folds)]
    num =([2 * (np.sum(thresh_comp_n0[:,idx[0]] & thresh_comp_n0[:,idx[1]]))
    for idx in ij])

    denom = [(np.sum(thresh_comp_n0[:,idx[0]]) + \
              np.sum(thresh_comp_n0[:,idx[1]]))
             for idx in ij]
    dices = np.array([float(num[i]) / denom[i] for i in range(n_corr)])
    print dices
    return dices.mean()
    
   
from brainomics import array_utils   
T=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','betas_24.npy'))
#T=np.load(os.path.join(BASE_PATH,'toward_on','Logistic_L1_L2_TV_with_HC','betas_mean.npy'))

T_thresh = np.zeros(T.shape)
for i in range(0,23):
    T_thresh[i,:], t = array_utils.arr_threshold_from_norm2_ratio(T[i,:] )
############################################################################
dice=dice_bar(T_thresh.T)



#Test with permutations the significance of a classification score
# The p-value is then given by the percentage of runs for which the score
#obtained is greater than the classification score obtained in the first place
T=np.load(os.path.join(BASE_PATH,'toward_on','svm','T.npy'))
betas=np.load(os.path.join(BASE_PATH,'toward_on','svm','betas.npy'))
y=np.load(os.path.join(BASE_PATH,'toward_on','svm','y.npy'))
subject=np.load(os.path.join(BASE_PATH,'toward_on','svm','subject.npy'))

T_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','T_IMA.npy'))
betas_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','betas_IMA.npy'))
y_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','y_IMA.npy'))
subject_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','subject_IMA.npy'))
#   
 
Tdiff=np.mean(T_IMA,axis=0)-np.mean(T[y==0],axis=0)
T_IMA_diff=T_IMA-Tdiff

nperms=1000
scores_perm = list()
recall_perm = list()

#true_acc= acc

for n in xrange(nperms):
   
    y=np.load(os.path.join(BASE_PATH,'toward_on','svm','y.npy'))
    y = np.random.permutation(y)
    list_predict=list()
    list_true=list()
    clf = svm.LinearSVC(C=10e-5,fit_intercept=True,class_weight='auto')

    for i in range(1,24):
        test_bool=(subject==i)
        train_bool=(subject!=i)
        Xtest=T[test_bool,:]
        ytest=y[test_bool]
        Xtrain=np.vstack((T_IMA_diff,T[train_bool,:]))
        ytrain=np.hstack((y_IMA,y[train_bool]))
        list_true.append(ytest.ravel())
        scaler = preprocessing.StandardScaler().fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xtest=scaler.transform(Xtest)
        clf.fit(Xtrain, ytrain.ravel())
        pred=(clf.predict(Xtest))
        list_predict.append(pred)

   
     
    t=np.concatenate(list_true)
    p=np.concatenate(list_predict)
    acc=metrics.accuracy_score(t,p)
    scores_perm.append(acc)
    recall_perm.append(recall_score(t,p,pos_label=None, average=None,labels=[0,1]))
    print n


scores_perm=np.array(scores_perm)
pval=np.sum(scores_perm >=0.61)/float(nperms)

recall_perm=np.array(recall_perm)
spe=recall_perm[:,0]
sen=recall_perm[:,1]
pval=np.sum(spe >=0.66)/float(nperms)
pval=np.sum(sen >=0.57)/float(nperms)


plt.hist(scores_perm, 10, label='Permutation scores')
plt.plot(2 * [0.61],plt.ylim(),'--g', linewidth=3)
plt.xlabel('Accuracy')

plt.hist(sen, 10, label='Permutation scores')
plt.plot(2 * [0.68],plt.ylim(),'--g', linewidth=3)
plt.xlabel('Specificity')

plt.hist(sen, 10, label='Permutation scores')
plt.plot(2 * [0.54],plt.ylim(),'--g', linewidth=3)
plt.xlabel('Sensitivity')





#Classification
###############################################################################

import csv


error = np.zeros((165))
for i in range(165):
    if t[i]==p[i]:
        error[i]=0
        
    else:
        error[i]=1  
        
    
a=np.array((subject,t,p,error))
a=a.T
   

df = pd.DataFrame(a,columns=['subject', 'true', 'prediction','error'])
df.to_csv('/neurospin/brainomics/2016_classif_hallu_fmri/toward_on/Logistic_L1_L2_TV_with_HC/0.1_0.1_0.1_classification_file.csv')