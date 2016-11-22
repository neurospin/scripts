# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:21:01 2016

@author: ad247405
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import nibabel as nib
import json
################
# Input/Output #
################

INPUT_BASE_DIR = '/neurospin/brainomics/2016_AUSZ/results/VBM/pca'
INPUT_DIR = '/neurospin/brainomics/2016_AUSZ/results/VBM/pca/5_folds/results'
INPUT_MASK = '/neurospin/brainomics/2016_AUSZ/results/VBM/mask.nii'             
INPUT_RESULTS_FILE=os.path.join(INPUT_BASE_DIR,"results.csv")                          
INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR,"5_folds","config.json")
OUTPUT_DIR = os.path.join(INPUT_BASE_DIR,"components_extracted")



babel_mask  = nib.load(INPUT_MASK)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()



##############
# Parameters #
##############

N_COMP = 5
EXAMPLE_FOLD = 0
INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,'{fold}','{key}','components.npz')
INPUT_PROJECTIONS_FILE_FORMAT = os.path.join(INPUT_DIR,'{fold}','{key}','X_train_transform.npz')
OUTPUT_COMPONENTS_FILE_FORMAT = os.path.join(OUTPUT_DIR,'{name}.nii')



config = json.load(open(INPUT_CONFIG_FILE))


# Load components and store them as nifti images
####################################################################
#data = pd.read_csv(INPUT_RESULTS_FILE)

for param in config["params"]:
    components = np.zeros((number_features, N_COMP))
    fold=0
    key = '_'.join([str(p)for p in param])
    print "process", key

    components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
    projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=fold,key=key) 

    components = np.load(components_filename)['arr_0']
    projections = np.load(projections_filename)['arr_0']
    assert projections.shape[1] == components.shape[1]

    # Loading as images
    loadings_arr = np.zeros((mask_bool.shape[0], mask_bool.shape[1], mask_bool.shape[2], N_COMP))
    for l in range(components.shape[1]):
        loadings_arr[mask_bool, l] = components[:,l]
 
            
    im = nib.Nifti1Image(loadings_arr,affine = babel_mask.get_affine())
    figname = OUTPUT_COMPONENTS_FILE_FORMAT.format(name=key)
    nib.save(im, figname)
#####################################################################

#
#
#import nilearn  
#from nilearn import plotting
#from nilearn import image
#filename = '/neurospin/brainomics/2016_AUSZ/results/VBM/pca/components_extracted/struct_pca_0.1_0.5_0.8/vol0004.nii.gz'
#nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)
