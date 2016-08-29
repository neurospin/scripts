# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:46:54 2016

@author: ad247405
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from brainomics import plot_utilities
import parsimony.utils.check_arrays as check_arrays

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"results")
INPUT_DATASET = os.path.join(INPUT_BASE_DIR,"T_hallu_only.npy")
INPUT_MASK = os.path.join(INPUT_BASE_DIR, "mask.nii.gz")                        
INPUT_RESULTS_FILE=os.path.join(INPUT_BASE_DIR,"results.csv")                          
INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR,"config_5folds.json")
OUTPUT_DIR = "/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/components_selected"
OUTPUT_COMPONENTS = os.path.join(OUTPUT_DIR,"components.csv")

##############
# Parameters #
##############

N_COMP = 3
EXAMPLE_FOLD = 0

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






# Open mask and data
#####################################################################
INPUT_MASK = os.path.join(INPUT_BASE_DIR,"mask.nii.gz")
mask=nib.load(INPUT_MASK)
mask=mask.get_data()
mask = mask !=0
mask_ima = nib.load(INPUT_MASK)
IM_SHAPE=mask.shape
X = np.load(INPUT_DATASET)

#####################################################################


# Load components and store them as nifti images
####################################################################
#data = pd.read_csv(INPUT_RESULTS_FILE)
#
#params=np.array(('pca', '0.0', '0.0', '0.0')) 
#params=np.array(('sparse_pca', '0.1', '0.0', '10.0')) 
#params=np.array(('struct_pca', '0.1', '1e-06', '0.5')) 
#params=np.array(('struct_pca', '0.1', '0.5', '0.5')) 


params=np.array(('sparse_pca', '0.0', '0.0', '5.0')) 
params=np.array(('struct_pca', '0.1', '0.5', '0.8'))
params=np.array(('struct_pca', '0.1', '1e-06', '0.1')) 

components  =np.zeros((63966, 10))
fold=0
key = '_'.join([str(param)for param in params])
print "process", key
name=params[0]

components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=fold,key=key) 

      
components = np.load(components_filename)['arr_0']
projections = np.load(projections_filename)['arr_0']
assert projections.shape[1] == components.shape[1]

# Loading as images
loadings_arr = np.zeros((IM_SHAPE[0], IM_SHAPE[1], IM_SHAPE[2], components.shape[1]))
for l in range(components.shape[1]):
    loadings_arr[mask, l] = components[:,l]

im = nib.Nifti1Image(loadings_arr,affine=mask_ima.get_affine())
figname = OUTPUT_COMPONENTS_FILE_FORMAT.format(name=key)
nib.save(im, figname)
#####################################################################





