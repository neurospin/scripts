#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:24:17 2017

@author: ad247405
"""


import os
import numpy as np
import json
from brainomics import array_utils
import brainomics.mesh_processing as mesh_utils
import shutil

BASE_PATH= "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results"    
MASK_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/data/mask.npy"
TEMPLATE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/freesurfer_template"               
OUTPUT = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results/enettv/weight_map"


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
#Enet weight map
enet_weight_map = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/\
results/enettv/enettv_VIP/model_selectionCV/refit/refit/0.1_0.15_0.15_0.7"
param = os.path.basename(enet_weight_map)
beta_path = os.path.join(enet_weight_map,"beta.npz")
enet = np.load(beta_path)['arr_0']


enet_t,t = array_utils.arr_threshold_from_norm2_ratio(enet,0.99)
 # left
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = enet_t[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, param+"_weight_map_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = enet_t[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, param+"_weight_map_right.gii"), data=tex)
   

#######################################################################################vBASE_PATH= "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/Freesurfer/results"    
BASE_PATH= "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/resultso"    
MASK_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/data/mask.npy"
TEMPLATE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/freesurfer_template"               
OUTPUT = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results/svm/weight_map"



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
svm_weight_map = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results/svm/svm_VIP/model_selectionCV/all/all/1e-05"
param = os.path.basename(svm_weight_map)
beta_path = os.path.join(svm_weight_map,"beta.npz")
svm = np.load(beta_path)['arr_0'].T
OUTPUT_SVM = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results/svm/weight_map"


svm_t,t = array_utils.arr_threshold_from_norm2_ratio(svm,0.99)
 # left
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = svm_t[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT_SVM, param+"_weight_map_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = svm_t[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT_SVM, param+"_weight_map_right.gii"), data=tex)
   


