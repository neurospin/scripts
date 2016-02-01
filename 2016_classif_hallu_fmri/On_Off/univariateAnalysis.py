# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:20:08 2015

@author: ad247405
"""

import nibabel as nibabel
import numpy as np
import os
import nilearn.signal
import nilearn.image
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


# Read pop csv

BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"
INPUT_CSV = os.path.join(BASE_PATH,"results", "patients.csv")
pop = pd.read_csv(os.path.join(BASE_PATH,"results", "patients.csv"))
periods = pd.read_csv(os.path.join(BASE_PATH,"results", "nperiods.csv"))

#############################################################################
#Mask on resampled Images (We use intecept between Harvard/Oxford cort/sub mask and MNI152linT1 mask)
ref=os.path.join(BASE_PATH,"atlases","MNI152lin_T1_3mm_brain_mask.nii.gz")
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(ref=ref
,output=(os.path.join(BASE_PATH,"results","mask.nii.gz")),smooth_size=None,dilation_size=None)
a=babel_mask_atlas.get_data()
babel_mask=nibabel.load(ref)
b=babel_mask.get_data()
b[a==0]=0
mask_bool=b!=0



#Save mask
out_im =nibabel.Nifti1Image(mask_bool.astype("int16"),affine=babel_mask_atlas.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"results","mask.nii.gz"))

#############################################################################
    

#First Level analysis, compute one contrast T map per subject
T=np.zeros((23,sum(mask_bool)))
k=0
t=0
subject_num=0

for i in range(1,31):
    
    curr=pop[pop['Subject']=='Lil'+str(i)]
    
    if len(curr) != 0 and i!=3:
        subject_num=subject_num+1
        curr=curr.set_index(np.arange(len(curr)))
        images = list()
        pathlist=list()
        all_scans=sorted(glob.glob(os.path.join(BASE_PATH,"data/DATA_Localizer/Patients MMH/",'Lil'+str(i)+'/*swr*')))
        for imagefile_name in all_scans:
            pathlist.append(imagefile_name)
            babel_image = nibabel.load(imagefile_name)
            #babel_image=nilearn.image.smooth_img(imgs=babel_image, fwhm=6)
            babel_image=nilearn.image.resample_img(babel_image, target_affine=babel_image.get_affine()*2, target_shape=[ x / 2 for x in babel_image.shape], interpolation='continuous', copy=True, order='F')

            images.append(babel_image.get_data().ravel())
        
        
   
        X = np.vstack(images)
        X = X[:, mask_bool.ravel()] 
        X=nilearn.signal.clean(X,detrend=True,standardize=True,confounds=None,low_pass=None, high_pass=None, t_r=1)

    #keep only off and on scans for further analysis
        mask_slicing=np.zeros(len(pathlist),dtype=bool)   
        for n in np.array(curr['Time']):
            for name in sorted(pathlist):
                if str(n) in name:
                    mask_slicing[int(pathlist.index(name))]=True           
       
        X=X[mask_slicing,:] 
        
        state_X=np.zeros(X.shape[0])

        nperiods_off=periods[periods['Subject']=='Lil'+str(i)]['off']
        nperiods_on=periods[periods['Subject']=='Lil'+str(i)]['on']
        p=max(int(nperiods_off),int(nperiods_on))
        
        DesignMat = np.zeros((X.shape[0],nperiods_off + nperiods_on)) # 
        contrasts=np.zeros([nperiods_off+nperiods_on])
        k=0
        curr_pos=0
        for n  in range(1,p+1):        
            l=len(curr[(curr['periods']==n) & (curr['State']=='off')])
            if l!=0:
                DesignMat[curr_pos:curr_pos+l,k] = 1 #Off
                state_X[curr_pos:curr_pos+l]=0
                contrasts[k]=-1
                curr_pos=curr_pos+l
                k=k+1
               
            l=len(curr[(curr['periods']==n) & (curr['State']=='on')])
            if l!=0:
                DesignMat[curr_pos:curr_pos+l,k] = 1 #On
                state_X[curr_pos:curr_pos+l]=1
                contrasts[k]=1
                curr_pos=curr_pos+l
                k=k+1
                
        #Remove inter subject variability        
        mean_X=( X[state_X==0].mean(axis=0) +  X[state_X==1].mean(axis=0) ) / float(2)      
        X=X-mean_X       
               
        muols = MUOLS(Y=X,X=DesignMat)
        muols.fit()
        tvals, pvals, dfs = muols.t_test(contrasts,pval=True, two_tailed=True)
        T[t,:]=tvals[0]
        t=t+1
        print t


  
 #Store variables
#############################################################################
np.save(os.path.join(BASE_PATH,'results','univariate_analysis','resolution_1.5mm','T.npy'),T)
np.save(os.path.join(BASE_PATH,'results','univariate_analysis','resolution_3mm','T.npy'),T)
#############################################################################

#Retrieve variables
#############################################################################
T=np.load(os.path.join(BASE_PATH,'results','univariate_analysis','resolution_1.5mm','T.npy'))
T=np.load(os.path.join(BASE_PATH,'results','univariate_analysis','resolution_3mm','T.npy'))
#############################################################################
############################################################################


#Second Level analysis
############################################################################

DesignMat=np.ones((23,1))
muols = MUOLS(Y=T,X=DesignMat)
muols.fit()
contrast=np.zeros((1,1))
contrast[0]=1

#Uncorrected raw pvalue
tvals, pvals, dfs = muols.t_test(contrast,pval=True, two_tailed=True)

#Save tvals
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = tvals[0]
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"results","univariate_analysis",'resolution_3mm',"tval_on-off.nii.gz"))


#Save raw pvalue
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = pvals[0]
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"results","univariate_analysis",'resolution_3mm',"pval_on-off.nii.gz"))


#Save log raw pvalues
log10_ps = -np.log10(pvals[0])
arr = np.zeros(mask_bool.shape)
arr[mask_bool] = log10_ps
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"results","univariate_analysis",'resolution_3mm',"p-log10_on-off.nii.gz"))


#############################################################################
#############################################################################

#Corrected pvalue using Tmax with 1000 permutation
nsubjects=23
nperms=1000
two_tailed=True
DesignMat=np.ones((23,1))
muols = MUOLS(Y=T,X=DesignMat)
muols.fit()
contrast=np.zeros((1,1))
contrast[0]=1
tvals, pvals, dfs = muols.t_test(contrast,pval=True, two_tailed=True)
        
max_t = list()

for i in xrange(nperms): 
        r=np.zeros((nsubjects,1))
        r[:,0]=np.random.choice((-1,1),nsubjects)
        Tp=r*abs(T)

        muols = MUOLS(Y=Tp,X=DesignMat)
        muols.fit()
        tvals_perm,_,_ = muols.t_test(contrast,pval=False, two_tailed=True)
                               
        if two_tailed:
            tvals_perm = np.abs(tvals_perm)
        max_t.append(np.max(tvals_perm, axis=1))
        del muols
        print i
            
max_t = np.array(max_t)
tvals_ = np.abs(tvals) if two_tailed else tvals
pvalues = np.array([np.array([np.sum(max_t[:, con] >= t) for t in tvals_[con, :]])\
                / float(nperms) for con in xrange(contrast.shape[0])])


#############################################################################
#############################################################################

#Save corrected pvalues
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = pvalues[0]
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"results","univariate_analysis",'resolution_3mm',"corrected_pval_perm_on-off.nii.gz"))

#Save log10corrected pvalues
log10_ps = -np.log10(pvalues[0])
arr = np.zeros(mask_bool.shape)
arr[mask_bool] = log10_ps
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"results","univariate_analysis",'resolution_3mm',"corrected_p-log10_on-off.nii.gz"))



clust = pd.read_csv("/neurospin/brainomics/2016_classif_hallu_fmri/results/population.csv")

