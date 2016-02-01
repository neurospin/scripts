# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:26:17 2015

@author: ad247405
"""


import nilearn.signal
import re
import glob
import os
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
#############################################################################
      

#Compute Tstats image for each block 
#############################################################################

T=np.zeros((165,63966))
betas=np.zeros((165,63966))
y=np.zeros((165))
subject=np.zeros((165))  
s=0
subject_num=0
t=0

for i in range(1,31):
    
    curr=pop[pop['Subject']=='Lil'+str(i)]
    if len(curr) != 0 and i!=3:
        subject_num=subject_num+1
        a=np.arange(len(curr))
        curr=curr.set_index(a)
        images = list()
        pathlist=list()
        all_scans=sorted(glob.glob(os.path.join(BASE_PATH,"data/DATA_Localizer/Patients MMH/",'Lil'+str(i)+'/*swr*')))
      
        for imagefile_name in all_scans:
            pathlist.append(imagefile_name)
            babel_image = nibabel.load(imagefile_name)
            #babel_image=nilearn.image.smooth_img(imgs=babel_image, fwhm=6)
            babel_image=nilearn.image.resample_img(babel_image, target_affine=babel_image.get_affine()*2, target_shape=[ x /2 for x in babel_image.shape], interpolation='continuous', copy=True, order='F')
            images.append(babel_image.get_data().ravel())
            
        X = np.vstack(images)
        X = X[:, mask_bool.ravel()] 
        X=nilearn.signal.clean(X,detrend=True,standardize=True,confounds=None,low_pass=None, high_pass=None, t_r=1)
       
    #keep only off and on scan for further analysis
        mask_slicing=np.zeros(len(pathlist),dtype=bool)   
        for n in np.array(curr['Time']):
            if n<100:
                n='00'+str(n)
            for name in sorted(pathlist):
                if str(n) in name:
                    mask_slicing[int(pathlist.index(name))]=True           
        X=X[mask_slicing,:]
        
        state_X=np.zeros(X.shape[0])
    
    

#############################################################################
    
        nperiods_off=periods[periods['Subject']=='Lil'+str(i)]['off']
        nperiods_on=periods[periods['Subject']=='Lil'+str(i)]['on']
        p=max(int(nperiods_off),int(nperiods_on))
        
        DesignMat = np.zeros((X.shape[0],nperiods_off + nperiods_on)) # 
        nsamples=np.zeros((nperiods_off + nperiods_on,1))
        k=0
        curr_pos=0
        for n  in range(1,p+1):        
            l=len(curr[(curr['periods']==n) & (curr['State']=='off')])
            if l!=0:
                DesignMat[curr_pos:curr_pos+l,k] = 1 #Off
                state_X[curr_pos:curr_pos+l]=0
                nsamples[k,0]=l
                curr_pos=curr_pos+l
                k=k+1
                y[s]=0
                subject[s]=subject_num
                s=s+1
      
            l=len(curr[(curr['periods']==n) & (curr['State']=='on')])
            if l!=0:
                nsamples[k,0]=l
                DesignMat[curr_pos:curr_pos+l,k] = 1 #On
                state_X[curr_pos:curr_pos+l]=1
                curr_pos=curr_pos+l
                k=k+1
                y[s]=1
                subject[s]=subject_num
                s=s+1
        
        
#        mean_X=( X[state_X==0].mean(axis=0) +  X[state_X==1].mean(axis=0) ) / float(2)      
#        X=X-mean_X
      
        muols = MUOLS(Y=X,X=DesignMat)
        muols.fit()
       
        nt=float(nsamples.sum())
        contrasts=np.identity((len(nsamples)))
#        for i in range(0,len(nsamples)):
#            contrasts[:,i]=contrasts[:,i]-nsamples[i]/nt
           
        tvals, pvals, dfs = muols.t_test(contrasts,pval=True, two_tailed=True)
        for m in range (0,nperiods_off+nperiods_on):
            T[t,:]=tvals[m]
            betas[t,:]=muols.coef[m,:]
            t=t+1
            print t


 