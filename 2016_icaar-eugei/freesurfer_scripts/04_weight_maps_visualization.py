# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:11:01 2016

@author: ad247405
"""


import os
import numpy as np
import json
import glob
from brainomics import array_utils
import brainomics.mesh_processing as mesh_utils
import shutil

BASE_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer"    
TEMPLATE_PATH = "/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_template"      

#ICAAR
###################################################################################
#svm
#penalty_start = 2
#MASK_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/data/mask.npy"
#OUTPUT = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/svm/model_selection_5folds/1" 
#beta = np.load(os.path.join(OUTPUT,"beta.npz"))['arr_0'][0,2:]
#beta,_ = array_utils.arr_threshold_from_norm2_ratio(beta,0.99)

#enttv
#penalty_start = 2
#MASK_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/data/mask.npy"
#OUTPUT = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/enettv/model_selection_5folds/0.5_0.72_0.08_0.2" 
#beta = np.load(os.path.join(OUTPUT,"beta.npz"))['arr_0']
#beta,_ = array_utils.arr_threshold_from_norm2_ratio(beta,0.99)
###################################################################################


#ICAAR+EUGEI
###################################################################################
#svm
#penalty_start = 2
#MASK_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR+EUGEI/data/mask.npy"
#OUTPUT = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR+EUGEI/svm/model_selection_5folds/1" 
#beta = np.load(os.path.join(OUTPUT,"beta.npz"))['arr_0'][0,2:]
#beta,_ = array_utils.arr_threshold_from_norm2_ratio(beta,0.99)

#enttv
penalty_start = 2
MASK_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR+EUGEI/data/mask.npy"
OUTPUT = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR+EUGEI/enettv/model_selection_5folds/0.5_0.56_0.24_0.2" 
beta = np.load(os.path.join(OUTPUT,"beta.npz"))['arr_0']
beta,_ = array_utils.arr_threshold_from_norm2_ratio(beta,0.99)
####################################################################################







shutil.copyfile(os.path.join(TEMPLATE_PATH, "lh.pial.gii"), os.path.join(OUTPUT, "lh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "rh.pial.gii"), os.path.join(OUTPUT, "rh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "lrh.pial.gii"), os.path.join(OUTPUT, "lrh.pial.gii"))


cor_l, tri_l = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "lh.pial.gii"))
cor_r, tri_r = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "rh.pial.gii"))
assert cor_l.shape[0] == cor_r.shape[0] 


cor_both, tri_both = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "lrh.pial.gii"))
mask__mesh = np.load(MASK_PATH)
assert mask__mesh.shape[0] == cor_both.shape[0] == cor_l.shape[0] * 2 ==  cor_l.shape[0] + cor_r.shape[0]
assert mask__mesh.shape[0], mask__mesh.sum() 

# Find the mapping from components in masked mesh to left_mesh and right_mesh
# concat was initialy: cor = np.vstack([cor_l, cor_r])
mask_left__mesh = np.arange(mask__mesh.shape[0])  < mask__mesh.shape[0] / 2
mask_left__mesh[np.logical_not(mask__mesh)] = False
mask_right__mesh = np.arange(mask__mesh.shape[0]) >= mask__mesh.shape[0] / 2
mask_right__mesh[np.logical_not(mask__mesh)] = False
assert mask__mesh.sum() ==  (mask_left__mesh.sum() + mask_right__mesh.sum())

# the mask of the left/right emisphere within the left/right mesh
mask_left__left_mesh = mask_left__mesh[:cor_l.shape[0]]
mask_right__right_mesh = mask_right__mesh[cor_l.shape[0]:]

# compute mask from components (in masked mesh) to left/right
a = np.zeros(mask__mesh.shape, int)
a[mask_left__mesh] = 1
a[mask_right__mesh] = 2
mask_left__beta = a[mask__mesh] == 1  # project mesh to mesh masked
mask_right__beta = a[mask__mesh] == 2
assert (mask_left__beta.sum() + mask_right__beta.sum()) == mask_left__beta.shape[0] == mask_right__beta.shape[0] == mask__mesh.sum() 
assert mask_left__mesh.sum() == mask_left__beta.sum()
assert mask_right__mesh.sum() == mask_right__beta.sum()

# Check mapping from beta left part to left_mesh
assert mask_left__beta.sum() == mask_left__left_mesh.sum()
assert mask_right__beta.sum() == mask_right__right_mesh.sum()

#############################################################################

#Save loading vectors
#############################################################################
 # left
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = beta[mask_left__beta]
print("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT,"tex_beta_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = beta[mask_right__beta]
print("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_beta_right.gii"), data=tex)
