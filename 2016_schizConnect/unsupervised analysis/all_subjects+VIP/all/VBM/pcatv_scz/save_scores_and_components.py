#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:30:56 2017

@author: ad247405
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import nibabel as nib
import json
from nilearn import plotting
from nilearn import image
from scipy.stats.stats import pearsonr
import pandas as pd
import nibabel as nb
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
import nibabel
import pandas as pd
import nibabel as nib
import json
import nilearn
from nilearn import plotting
from nilearn import image
import array_utils

INPUT_MASK = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/mask.nii'

components = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/vbm_pcatv_all+VIP_scz/results/0/struct_pca_0.1_0.1_0.1/components.npz")["arr_0"]

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data"
#Save only positive components to facilitate interpretation
assert components.shape == (125959, 10)
components = np.abs(components)
np.save(os.path.join(output,"components"),np.abs(components))



babel_mask  = nib.load(INPUT_MASK)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()

#############################################################################
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/components_pictures"
N_COMP =10

for i in range(components.shape[1]):
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = components[:,i]
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(WD,"comp_%s.nii.gz") % (i)
    out_im.to_filename(filename)
    comp_data = nibabel.load(filename).get_data()
    comp_t,t = array_utils.arr_threshold_from_norm2_ratio(comp_data, .99)
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)
    plt.savefig(os.path.join(WD,"comp_%s.png") % (i))
    print (i)
    print (t)





#Save all subjects
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
X_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/X.npy")
assert X_all.shape == (606, 125961)
X_all = X_all[:,2:]
U_all, d = transform(V=components , X = X_all , n_components=components.shape[1], in_place=False)
assert U_all.shape == (606, 10)
U_all_scz = U_all[y_all==1,:]
U_all_con = U_all[y_all==0,:]
np.save(os.path.join(output,"U_all"),U_all)
np.save(os.path.join(output,"U_all_scz"),U_all_scz)
np.save(os.path.join(output,"U_all_con"),U_all_con)

#Save VIP subjects
y_vip = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/VIP/y.npy")
X_vip = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/VIP/X.npy")
assert X_vip.shape ==  (92, 125961)
X_vip = X_vip[:,2:]
U_vip, d = transform(V=components , X = X_vip , n_components=components.shape[1], in_place=False)
assert U_vip.shape == (92, 10)
U_vip_scz = U_vip[y_vip==1,:]
U_vip_con = U_vip[y_vip==0,:]
np.save(os.path.join(output,"U_vip"),U_vip)
np.save(os.path.join(output,"U_vip_scz"),U_vip_scz)
np.save(os.path.join(output,"U_vip_con"),U_vip_con)


#Save NUDAST subjects
y_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/y.npy")
X_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/X.npy")
assert X_nudast.shape ==  (270, 125961)
X_nudast = X_nudast[:,2:]
U_nudast, d = transform(V=components , X = X_nudast , n_components=components.shape[1], in_place=False)
assert U_nudast.shape == (270, 10)
U_nudast_scz = U_nudast[y_nudast==1,:]
U_nudast_con = U_nudast[y_nudast==0,:]
np.save(os.path.join(output,"U_nudast"),U_nudast)
np.save(os.path.join(output,"U_nudast_scz"),U_nudast_scz)
np.save(os.path.join(output,"U_nudast_con"),U_nudast_con)

#Save COBRE subjects
y_cobre = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/COBRE/y.npy")
X_cobre = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/COBRE/X.npy")
assert X_cobre.shape ==  (164, 125961)
X_cobre = X_cobre[:,2:]
U_cobre, d = transform(V=components , X = X_cobre , n_components=components.shape[1], in_place=False)
assert U_cobre.shape == (164, 10)
U_cobre_scz = U_cobre[y_cobre==1,:]
U_cobre_con = U_cobre[y_cobre==0,:]
np.save(os.path.join(output,"U_cobre"),U_cobre)
np.save(os.path.join(output,"U_cobre_scz"),U_cobre_scz)
np.save(os.path.join(output,"U_cobre_con"),U_cobre_con)

#Save Nmoprh subjects
y_nmorph= np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NMORPH/y.npy")
X_nmorph = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NMORPH/X.npy")
assert X_nmorph.shape ==  (80, 125961)
X_nmorph = X_nmorph[:,2:]
U_nmorph, d = transform(V=components , X = X_nmorph , n_components=components.shape[1], in_place=False)
assert U_nmorph.shape == (80, 10)
U_nmorph_scz = U_nmorph[y_nmorph==1,:]
U_nmorph_con = U_nmorph[y_nmorph==0,:]
np.save(os.path.join(output,"U_nmorph"),U_nmorph)
np.save(os.path.join(output,"U_nmorph_scz"),U_nmorph_scz)
np.save(os.path.join(output,"U_nmorph_con"),U_nmorph_con)

#Save PRAGUE subjects
y_prague= np.load("/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/VBM/data/y.npy")
X_prague = np.load("/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/VBM/data/X.npy")
assert X_prague.shape ==  (133, 125961)
X_prague = X_prague[:,2:]
U_prague, d = transform(V=components , X = X_prague , n_components=components.shape[1], in_place=False)
assert U_prague.shape == (133, 10)
U_prague_scz = U_prague[y_prague==1,:]
U_prague_con = U_prague[y_prague==0,:]
np.save(os.path.join(output,"U_prague"),U_prague)
np.save(os.path.join(output,"U_prague_scz"),U_prague_scz)
np.save(os.path.join(output,"U_prague_con"),U_prague_con)


def transform(V, X, n_components, in_place=False):
    """ Project a (new) dataset onto the components.
    Return the projected data and the associated d.
    We have to recompute U and d because the argument may not have the same
    number of lines.
    The argument must have the same number of columns than the datset used
    to fit the estimator.
    """
    Xk = X
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
