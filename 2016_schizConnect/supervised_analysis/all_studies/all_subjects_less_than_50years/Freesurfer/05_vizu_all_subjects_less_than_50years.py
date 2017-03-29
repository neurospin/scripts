# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:48:02 2017

@author: ad247405
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:41:20 2017

@author: ad247405
"""



import os
import numpy as np
import json
from brainomics import array_utils
import brainomics.mesh_processing as mesh_utils
import shutil

BASE_PATH= "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/Freesurfer/all_subjects_less_than_50years/results"    
MASK_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/Freesurfer/all_subjects_less_than_50years/data/mask.npy"
TEMPLATE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/Freesurfer/all_subjects_less_than_50years/freesurfer_template"               
OUTPUT = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/Freesurfer/all_subjects_less_than_50years/vizu"


shutil.copyfile(os.path.join(TEMPLATE_PATH, "lh.pial.gii"), os.path.join(OUTPUT, "lh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "rh.pial.gii"), os.path.join(OUTPUT, "rh.pial.gii"))



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
#SVM weight map
svm_weight_map = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/Freesurfer/all_subjects_less_than_50years/results/svm/svm_model_selection_5folds_all_subjects_less_than_50years/model_selectionCV/all/all/1e-05/beta.npz"
svm = np.load(svm_weight_map)['arr_0'].T


svm_t,t = array_utils.arr_threshold_from_norm2_ratio(svm,0.99)
 # left
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = svm_t[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "svm_weight_map_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = svm_t[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "svm_weight_map_right.gii"), data=tex)
   
#############################################################################
#Enet weight map
enet_weight_map = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/Freesurfer/all_subjects_less_than_50years/results/enettv/model_selection_5folds_all_subjects_less_than_50years/model_selectionCV/refit/refit/1.0_0.03_0.27_0.7/beta.npz"
enet = np.load(enet_weight_map)['arr_0']


enet_t,t = array_utils.arr_threshold_from_norm2_ratio(enet,0.99)
 # left
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = enet_t[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "enet_1.0_0.03_0.27_0.7_weight_map_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = enet_t[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "enet_1.0_0.03_0.27_0.7_weight_map_right.gii"), data=tex)
   

