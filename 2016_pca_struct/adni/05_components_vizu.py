#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:08:03 2017

@author: ad247405
"""



import os
import numpy as np
import json
from brainomics import array_utils
import brainomics.mesh_processing as mesh_utils
import shutil

BASE_PATH= "/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds"    
TEMPLATE_PATH = "/neurospin/brainomics/2016_pca_struct/adni/data/freesurfer_template"               
BASE_OUTPUT = "/neurospin/brainomics/2016_pca_struct/adni/components_extracted"


shutil.copyfile(os.path.join(TEMPLATE_PATH, "lh.pial.gii"), os.path.join(OUTPUT, "lh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "rh.pial.gii"), os.path.join(OUTPUT, "rh.pial.gii"))

config  = json.load(open(os.path.join(BASE_PATH,"config_dCV.json")))



cor_l, tri_l = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "lh.pial.gii"))
cor_r, tri_r = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "rh.pial.gii"))
assert cor_l.shape[0] == cor_r.shape[0] 


cor_both, tri_both = mesh_utils.mesh_arrays(os.path.join(BASE_PATH, "lrh.pial.gii"))
mask__mesh = np.load(os.path.join(BASE_PATH, "mask.npy"))
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
#Load coomponen

params = "struct_pca_0.1_0.5_0.1"
OUTPUT = os.path.join(BASE_OUTPUT,params)
params = "struct_pca_0.01_0.0001_0.8"
OUTPUT = os.path.join(BASE_OUTPUT,params)
params = "sparse_pca_0.0_0.0_1.0"
OUTPUT = os.path.join(BASE_OUTPUT,params)
params = "struct_pca_0.1_0.5_0.5"
OUTPUT = os.path.join(BASE_OUTPUT,params)


INPUT_COMPONENTS_FILE_FORMAT = os.path.join(BASE_PATH,"model_selectionCV",'all','all',params,'components.npz')
components = np.zeros((79440, 3))
components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(params=params)
components = np.load(components_filename)['arr_0']

#
comp0,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,0],0.99)
comp1,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,1],0.99)
comp2,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,2],0.99)

#comp0 = components[:,0]
#comp1 = components[:,1]
#comp2 = components[:,2]

comp0_all=comp0
comp1_all=comp1
comp2_all=comp2

#Save loading vectors
#############################################################################
 # left
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp0[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_0_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp0[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_0_right.gii"), data=tex)
   
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp1[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_1_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp1[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_1_right.gii"), data=tex)
  
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp2[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_2_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp2[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_2_right.gii"), data=tex)
