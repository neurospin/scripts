# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:25:41 2016

@author: ad247405

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT_ICAARZ:
- mask.nii
- y.npy
- X.npy = intercept + Age + Gender + Voxel
"""

import os
import numpy as np
import pandas as pd
import nibabel
#import brainomics.image_atlas
import nilearn
from nilearn import plotting
from mulm import MUOLS
import matplotlib.pyplot as plt
#import array_utils
#import proj_classif_config
GENDER_MAP = {'F': 0, 'M': 1}
Lithresponse_MAP = {'Good': 1, 'Bad': 0}

BASE_PATH = "C:/Users/js247994/Documents/Bipli2/"
INPUT_CSV_ICAAR = os.path.join(BASE_PATH,"Processing","BipLipop.csv")
INPUT_FILES_DIR = os.path.join(BASE_PATH,"Processing/Processing2018_02/Lithiumfiles_02_mask_b/")

OUTPUT_DATA = os.path.join(BASE_PATH,"Processing/Analysisoutputs")

# Read pop csv
pop = pd.read_csv(INPUT_CSV_ICAAR)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)
pop['Lithresp.num']=pop["lithresponse"].map(Lithresponse_MAP)
#############################################################################
# Read images
n = len(pop)
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
Y = np.zeros((n, 1)) # DX
images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    #print(cur)
    imagefile_name = cur.path_VBM
    imagefile_path = os.path.join(INPUT_FILES_DIR,imagefile_name.as_matrix()[0])
    babel_image = nibabel.load(imagefile_path)
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["age", "sex.num"]]).ravel()
    Y[i, 0] = cur["Lithresp.num"]

shape = babel_image.get_data().shape

#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)

mask_ima = nibabel.load(os.path.join(BASE_PATH,"Processing", "ROIs", "Wholebrain.nii"))
mask_arr = mask_ima.get_data() != 0

#############################################################################

# Save data X and y
X = Xtot[:, mask_arr.ravel()]
#Use mean imputation, we could have used median for age
#Remove nan lines 
X = np.nan_to_num(X)

np.save(os.path.join(OUTPUT_DATA, "X.npy"), X)
np.save(os.path.join(OUTPUT_DATA, "Z.npy"), Z)
np.save(os.path.join(OUTPUT_DATA, "Y.npy"), Y)


###############################################################################
#############################################################################
import pandas as pd
#import seaborn as sns

X = np.load(os.path.join(OUTPUT_DATA, "X.npy"))
Z = np.load(os.path.join(OUTPUT_DATA, "Z.npy"))
#X=X[0:9,:]

X = X - X.mean(axis=1)[:, np.newaxis]

#Xn=np.copy(X)
#Xn1 -= X.mean(axis=0)
#Xn1 /= X.std(axis=0)

DesignMat=Z

muols = MUOLS(Y=X,X=DesignMat)
muols.fit()
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0], pval=True)

import statsmodels.sandbox.stats.multicomp as multicomp
_, pvals_fwer, _, _ = multicomp.multipletests(pvals, alpha=0.05,
method='bonferroni')
n_features=np.size(X,1)
n_info = int(n_features/10)
TP = np.sum(pvals_fwer[:n_info ] < 0.05) # True Positives
FP = np.sum(pvals_fwer[n_info: ] < 0.05) # False Positives
print("FWER correction, FP: %i, TP: %i" % (FP, TP))


pvallogged=-np.log10(pvals[0])
pd.Series(tvals.ravel()).describe()
pd.Series(pvals.ravel()).describe()
pd.Series(pvallogged.ravel()).describe()
#sns.


#check for multiple comparison, Bonferonni and/or False Discovery Rate

#arr = np.zeros(mask_arr.shape); arr[mask_arr] = (Xmeannorm)
#out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())

save=False
display=False
predict=False

varname='new'

if predict:
    
    Xtestf = np.load(os.path.join(OUTPUT_DATA, "X.npy"))
    Xtest=Xtestf[9,3:]
    Ztest=Xtestf[9,0:3]
    #Xtest -= Xtest.mean(axis=0)
    #Xtest /= Xtest.std(axis=0)
    yvals=muols.predict(Ztest)
    yvalsn= (yvals*Xtest.std(axis=0)+Xtest.mean(axis=0))
    yvalsarr = np.zeros(mask_arr.shape);
    yvalsarr[mask_arr] = yvalsn
    out_im = nibabel.Nifti1Image(yvalsarr, affine=mask_ima.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_DATA,"test_im10.nii.gz"))

if save:
    
    #arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals[0])
    pvallogged=-np.log10(pvals[0])
    arrlogp = np.zeros(mask_arr.shape); arrlogp[mask_arr] = pvallogged
    out_imlogp = nibabel.Nifti1Image(arrlogp, affine=mask_ima.get_affine())
    out_imlogp.to_filename(os.path.join(OUTPUT_DATA,varname+"p_vals_subj_base_log10.nii.gz"))
    
    arrpval = np.zeros(mask_arr.shape); arrpval[mask_arr] = (pvals[0])
    out_impval = nibabel.Nifti1Image(arrpval, affine=mask_ima.get_affine())
    out_impval.to_filename(os.path.join(OUTPUT_DATA,varname+"p_vals_subj_base.nii.gz"))
    
    arrtval = np.zeros(mask_arr.shape); arrtval[mask_arr] = tvals[0]
    out_imtval = nibabel.Nifti1Image(arrtval, affine=mask_ima.get_affine())
    out_imtval.to_filename(os.path.join(OUTPUT_DATA,varname+"t_vals_subj_base.nii.gz"))
    
if display:

    filename = os.path.join(OUTPUT_DATA,varname+"p_vals_subj_base_log10.nii.gz")
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic Pvals log10 map",cmap=plt.cm.bwr,vmax=3)
    
    filename = os.path.join(OUTPUT_DATA,varname+"p_vals_subj_base.nii.gz")
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic Pvals map")
    
    filename = os.path.join(OUTPUT_DATA,varname+"t_vals_subj_base.nii.gz")
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic Tvals map")
    ##################################################################################