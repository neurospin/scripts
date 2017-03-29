#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:56:26 2017

@author: ad247405
"""



import os
import numpy as np
import json
from brainomics import array_utils
import brainomics.mesh_processing as mesh_utils
import shutil

BASE_PATH= "/neurospin/brainomics/2016_AUSZ/results/Freesurfer/FS_pca_tv_patients_only"    
TEMPLATE_PATH = "/neurospin/brainomics/2016_AUSZ/preproc_FS/freesurfer_template"               
OUTPUT = "/neurospin/brainomics/2016_AUSZ/results/Freesurfer/FS_pca_tv_patients_only/components_extracted"


shutil.copyfile(os.path.join(TEMPLATE_PATH, "lh.pial.gii"), os.path.join(OUTPUT, "lh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "rh.pial.gii"), os.path.join(OUTPUT, "rh.pial.gii"))



cor_l, tri_l = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "lh.pial.gii"))
cor_r, tri_r = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "rh.pial.gii"))
assert cor_l.shape[0] == cor_r.shape[0] 


cor_both, tri_both = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "lrh.pial.gii"))
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

params = "struct_pca_0.1_0.8_0.1"
OUTPUT = os.path.join(OUTPUT,params)

INPUT_COMPONENTS_FILE_FORMAT = os.path.join(BASE_PATH,"model_selectionCV",'all','all',params,'components.npz')
components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(params=params)
components = np.load(components_filename)['arr_0']

#
comp0,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,0],0.99)
comp1,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,1],0.99)
comp2,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,2],0.99)
comp3,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,3],0.99)
comp4,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,4],0.99)



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

tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp3[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_3_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp3[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_3_right.gii"), data=tex)

tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp4[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_4_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp4[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_4_right.gii"), data=tex)