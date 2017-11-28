# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:12:45 2016

@author: ad247405
"""

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
import nibabel as nib
import json
################
# Input/Output #
################

INPUT_BASE_DIR = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis_wto_s20'
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"5_folds","results")
INPUT_MASK = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results/multivariate_analysis/data/MNI152_T1_3mm_brain_mask.nii.gz'
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
    print("process", key)

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
#With all subjects
comp1= '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis/components_extracted/struct_pca_0.1_0.1_0.1/vol0000.nii.gz'
comp2= '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis/components_extracted/struct_pca_0.1_0.1_0.1/vol0001.nii.gz'
comp3= '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis/components_extracted/struct_pca_0.1_0.1_0.1/vol0002.nii.gz'
comp4= '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis/components_extracted/struct_pca_0.1_0.1_0.1/vol0003.nii.gz'
comp5= '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis/components_extracted/struct_pca_0.1_0.1_0.1/vol0004.nii.gz'


#Without subject 19
comp1= '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis_wto_s20/components_extracted/0.1_0.1_0.1/vol0000.nii.gz'
comp2= '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis_wto_s20/components_extracted/0.1_0.1_0.1/vol0001.nii.gz'
comp3= '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis_wto_s20/components_extracted/0.1_0.1_0.1/vol0002.nii.gz'
comp4= '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis_wto_s20/components_extracted/0.1_0.1_0.1/vol0003.nii.gz'
comp5= '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis_wto_s20/components_extracted/0.1_0.1_0.1/vol0004.nii.gz'




import array_utils
comp_t1,t1 = array_utils.arr_threshold_from_norm2_ratio(nib.load(comp1).get_data(), .99)
comp_t2,t2 = array_utils.arr_threshold_from_norm2_ratio(nib.load(comp2).get_data(), .99)
comp_t3,t3 = array_utils.arr_threshold_from_norm2_ratio(nib.load(comp3).get_data(), .99)
comp_t4,t4 = array_utils.arr_threshold_from_norm2_ratio(nib.load(comp4).get_data(), .99)
comp_t5,t5 = array_utils.arr_threshold_from_norm2_ratio(nib.load(comp5).get_data(), .99)

import nilearn
from nilearn import plotting
from nilearn import image
nilearn.plotting.plot_glass_brain(comp1,colorbar=True,plot_abs=False,threshold=t1)
nilearn.plotting.plot_glass_brain(comp2,colorbar=True,plot_abs=False,threshold=t2)
nilearn.plotting.plot_glass_brain(comp3,colorbar=True,plot_abs=False,threshold=t3)
nilearn.plotting.plot_glass_brain(comp4,colorbar=True,plot_abs=False,threshold=t4)
nilearn.plotting.plot_glass_brain(comp5,colorbar=True,plot_abs=False,threshold=t5)


nilearn.plotting.plot_glass_brain(comp1,colorbar=True,plot_abs=False,threshold=t1,vmax=0.25)
plt.savefig("/neurospin/brainomics/2016_classif_hallu_fmri_bis/submission/revised_HBM/comp1.png")

nilearn.plotting.plot_glass_brain(comp2,colorbar=True,plot_abs=False,threshold=t2,vmax=0.20)
plt.savefig("/neurospin/brainomics/2016_classif_hallu_fmri_bis/submission/revised_HBM/comp2.png")

nilearn.plotting.plot_glass_brain(comp3,colorbar=True,plot_abs=False,threshold=t3,vmax=0.20)
plt.savefig("/neurospin/brainomics/2016_classif_hallu_fmri_bis/submission/revised_HBM/comp3.png")

nilearn.plotting.plot_glass_brain(comp4,colorbar=True,plot_abs=False,threshold=t4,vmax=0.15)
plt.savefig("/neurospin/brainomics/2016_classif_hallu_fmri_bis/submission/revised_HBM/comp4.png")
