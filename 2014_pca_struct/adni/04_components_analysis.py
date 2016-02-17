# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:12:36 2016

@author: ad247405
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from brainomics import plot_utilities
import parsimony.utils.check_arrays as check_arrays
import scipy.stats



################
# Input/Output #
################

BASE_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_5"
INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_5/adni_5folds/"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "results")

INPUT_DATASET = os.path.join(INPUT_BASE_DIR,
                             "X.npy")
INPUT_MASK = os.path.join(BASE_DIR,
                          "mask.npy")
                          
INPUT_RESULTS_FILE=os.path.join(INPUT_BASE_DIR,"results.csv")
                           
INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR,
                                 "config_5folds.json")

OUTPUT_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_5/adni_5folds"
OUTPUT_COMPONENTS = os.path.join(OUTPUT_DIR,
                                 "components.csv")

##############
# Parameters #
##############

N_COMP = 3

INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'components.npz')
INPUT_PROJECTIONS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'X_train_transform.npz')

OUTPUT_COMPONENTS_FILE_FORMAT = os.path.join(OUTPUT_DIR,
                                             '{name}.nii')

def transform(V, X, n_components, in_place=False):
    """ Project a (new) dataset onto the components.
    Return the projected data and the associated d.
    We have to recompute U and d because the argument may not have the same
    number of lines.
    The argument must have the same number of columns than the datset used
    to fit the estimator.
    """
    Xk = check_arrays(X)
    if not in_place:
        Xk = Xk.copy()
    n, p = Xk.shape
    if p != V.shape[0]:
        raise ValueError(
                    "The argument must have the same number of columns "
                    "than the datset used to fit the estimator.")
    U = np.zeros((n, n_components))
    d = np.zeros((n_components, ))
    for k in range(n_components):
        # Project on component j
        vk = V[:, k].reshape(-1, 1)
        uk = np.dot(X, vk)
        uk /= np.linalg.norm(uk)
        U[:, k] = uk[:, 0]
        dk = np.dot(uk.T, np.dot(Xk, vk))
        d[k] = dk
        # Residualize
        Xk -= dk * np.dot(uk, vk.T)
    return U, d


####################################################################

# Load data
####################################################################
data = pd.read_csv(INPUT_RESULTS_FILE)
X=np.load(os.path.join(BASE_DIR,'X.npy'))
y=np.load(os.path.join(INPUT_BASE_DIR,'y.npy'))
  
  
#Define the parameter to load  
params=np.array(('struct_pca', '0.1', '0.5', '0.5')) 
components  =np.zeros((X.shape[1], 3))
fold=0 # First Fold is whole dataset
key = '_'.join([str(param)for param in params])
print "process", key
name=params[0]

# Load components and projections
####################################################################
components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=fold,key=key)  
components = np.load(components_filename)['arr_0']
projections = np.load(projections_filename)['arr_0']
assert projections.shape[1] == components.shape[1]


# Save loadings to visualize
loadings_arr = np.zeros((IM_SHAPE[0], IM_SHAPE[1], IM_SHAPE[2], components.shape[1]))
for l in range(components.shape[1]):
    loadings_arr[mask, l] = components[:,l]

im = nib.Nifti1Image(loadings_arr,affine=mask_ima.get_affine())
figname = OUTPUT_COMPONENTS_FILE_FORMAT.format(name=key)
nib.save(im, figname)


#Test correlation of projection with clinical DX
#####################################################################
t, pval = scipy.stats.ttest_ind(projections[y==0],projections[y==1,0])
print "pvalue for first component : r%" %(pval)
t, pval = scipy.stats.ttest_ind(projections[y==0],projections[y==1,1])
print "pvalue for second component : r%" %(pval)
t, pval = scipy.stats.ttest_ind(projections[y==0],projections[y==1,2])
print "pvalue for third component : r%" %(pval)





  


