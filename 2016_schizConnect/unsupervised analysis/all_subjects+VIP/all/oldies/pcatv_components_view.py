
"""
Created on Fri Mar 25 12:46:54 2016

@author: ad247405
"""


import os
import subprocess
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
import brainomics.image_atlas
import brainomics.array_utils
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, recall_score
import pandas as pd
from collections import OrderedDict
import nilearn
from nilearn import plotting
from nilearn import image
import seaborn as sns
import matplotlib.pylab as plt
import shutil
import sys
sys.path.insert(0,'/home/ed203246/git/scripts/brainomics')
import array_utils, mesh_processing
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from nibabel import gifti
from brainomics import array_utils

BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/pcatv_scz/FS_pcatv_all+VIP_scz"
MASK_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/mask.npy"


comp = "struct_pca_0.1_0.5_0.1"
COMP_PATH = os.path.join(BASE_PATH,"results","0",comp,"components.npz")
components = np.load(COMP_PATH)["arr_0"]



WD = "/neurospin/brainomics/neuroimaging_ressources/freesurfer_utils/fsaverage"
mesh_l = os.path.join(WD, "lh.inflated.gii")
mesh_r = os.path.join(WD,"rh.inflated.gii")

sulc_l = os.path.join(WD,"lh.sulc")
sulc_r = os.path.join(WD, "rh.sulc")


mask_mesh = np.load(MASK_PATH)


#Plot weight map of all 10 components (both lateral and ventral view)
for i in range(11):
    beta = components[:,i]
    beta,_ = array_utils.arr_threshold_from_norm2_ratio(beta,0.99)


    [coords_l, faces_l], beta_mesh_l, [coords_r, faces_r], beta_mesh_r, stat = \
        beta_to_mesh_lr(beta, mask_mesh, mesh_l, mesh_r, threshold=1.)

    output_filename = os.path.join("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/pcatv_scz/vizu",comp,"comp%r_right.png"%(i))
    hemi, view = 'right', 'medial'
    coords_x, faces_x, beta_mesh_x, sulc_x = coords_r, faces_r, beta_mesh_r, sulc_r
    vmax_beta = np.max(np.abs(beta)) / 10
    vmax_beta = np.max(np.abs(beta_mesh_x) * 1000) / 10

    plotting.plot_surf_stat_map([coords_x, faces_x], stat_map=1000 * beta_mesh_x,
                                hemi=hemi, view=view,
                                bg_map=sulc_x,
                                darkness=.5,
                                cmap="cold_hot",
                                output_file=output_filename
                                )


    output_filename = os.path.join("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/pcatv_scz/vizu",comp,"comp%r_right_lateral.png"%(i))
    hemi, view = 'right', 'medial'
    hemi, view = 'right', 'lateral'
    coords_x, faces_x, beta_mesh_x, sulc_x = coords_r, faces_r, beta_mesh_r, sulc_r
    vmax_beta = np.max(np.abs(beta)) / 10
    vmax_beta = np.max(np.abs(beta_mesh_x) * 1000) / 10
    plotting.plot_surf_stat_map([coords_x, faces_x], stat_map=1000 * beta_mesh_x,
                                hemi=hemi, view=view,
                                bg_map=sulc_x, #bg_on_data=True,
                                #vmax = vmax_beta,#stat[2] / 10,#vmax=vmax_beta,
                                darkness=.5,
                                cmap="cold_hot",
                                #symmetric_cbar=True,
                                output_file=output_filename
                                )


    output_filename = os.path.join("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/pcatv_scz/vizu",comp,"comp%r_left.png"%(i))
    hemi, view = 'left', 'medial'
    coords_x, faces_x, beta_mesh_x, sulc_x = coords_l, faces_l, beta_mesh_l, sulc_l
    vmax_beta = np.max(np.abs(beta)) / 10
    vmax_beta = np.max(np.abs(beta_mesh_x) * 1000) / 10
    plotting.plot_surf_stat_map([coords_x, faces_x], stat_map=1000 * beta_mesh_x,
                                hemi=hemi, view=view,
                                bg_map=sulc_x, #bg_on_data=True,
                                #vmax = vmax_beta,#stat[2] / 10,#vmax=vmax_beta,
                                darkness=.5,
                                cmap="cold_hot",
                                #symmetric_cbar=True,
                                output_file=output_filename
                                )


    output_filename = os.path.join("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/pcatv_scz/vizu",comp,"comp%r_left_lateral.png"%(i))
    hemi, view = 'left', 'lateral'
    coords_x, faces_x, beta_mesh_x, sulc_x = coords_l, faces_l, beta_mesh_l, sulc_l
    vmax_beta = np.max(np.abs(beta)) / 10
    vmax_beta = np.max(np.abs(beta_mesh_x) * 1000) / 10
    plotting.plot_surf_stat_map([coords_x, faces_x], stat_map=1000 * beta_mesh_x,
                                hemi=hemi, view=view,
                                bg_map=sulc_x, #bg_on_data=True,
                                #vmax = vmax_beta,#stat[2] / 10,#vmax=vmax_beta,
                                darkness=.5,
                                cmap="cold_hot",
                                #symmetric_cbar=True,
                                output_file=output_filename
                                )





#Utilities FS
##############################################################################

def beta_to_mesh_lr(beta, mask_mesh, mesh_l, mesh_r, threshold=.99):
    # beta to array of mesh size
    #ouput_filename = os.path.splitext(beta_filename)[0] + ".nii.gz"
    assert beta.shape[0] == mask_mesh.sum()
    beta_t, t = array_utils.arr_threshold_from_norm2_ratio(beta, threshold)
    #print(np.sum(beta != 0), np.sum(beta_t != 0), np.max(np.abs(beta_t)))
    beta_mesh = np.zeros(mask_mesh.shape)
    beta_mesh[mask_mesh] = beta_t.ravel()

     # mesh, l+r
    mesh_l = nilearn.plotting.surf_plotting.load_surf_mesh(mesh_l)
    coords_l, faces_l = mesh_l[0], mesh_l[1]
    mesh_r = nilearn.plotting.surf_plotting.load_surf_mesh(mesh_r)
    coords_r, faces_r = mesh_r[0], mesh_r[1]
    assert coords_l.shape[0] == coords_r.shape[0] == beta_mesh.shape[0] / 2

    beta_mesh_l = np.zeros(coords_l.shape)
    beta_mesh_l = beta_mesh[:coords_l.shape[0]]
    beta_mesh_r = np.zeros(coords_r.shape)
    beta_mesh_r = beta_mesh[coords_l.shape[0]:]

    return [coords_l, faces_l], beta_mesh_l, [coords_r, faces_r], beta_mesh_r, [np.sum(beta != 0), np.sum(beta_t != 0), np.max(np.abs(beta_t))]
