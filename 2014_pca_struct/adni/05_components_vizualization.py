# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:37:38 2016

@author: ad247405
"""



import os
import numpy as np
import json
import glob
from brainomics import array_utils
import brainomics.mesh_processing as mesh_utils
import shutil

BASE_PATH= "/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only"    
TEMPLATE_PATH = os.path.join(BASE_PATH, "freesurfer_template")                      
OUTPUT = "/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only/components_extracted"


shutil.copyfile(os.path.join(TEMPLATE_PATH, "lh.pial.gii"), os.path.join(OUTPUT, "lh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "rh.pial.gii"), os.path.join(OUTPUT, "rh.pial.gii"))

config  = json.load(open(os.path.join(BASE_PATH,'adni_5folds',"config_5folds.json")))



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
#Load coomponent
INPUT_COMPONENTS_FILE_FORMAT = os.path.join(BASE_PATH,"adni_5folds/results",'{fold}','{key}','components.npz')

params=np.array(('struct_pca', '0.1', '0.5', '0.1')) 

params=np.array(('struct_pca', '0.1', '1e-06', '0.1')) 

params=np.array(('sparse_pca', '0.0', '0.0', '5.0')) 

params=np.array(('pca', '0.0', '0.0', '0.0')) 

params=np.array(('struct_pca', '0.1', '1e-06', '0.5')) 

components = np.zeros((79440, 3))
fold=0 # First Fold is whole dataset
key = '_'.join([str(param)for param in params])
print "process", key
name=params[0]
components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
components = np.load(components_filename)['arr_0']


comp0,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,0],0.99)
comp1,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,1],0.99)
comp2,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,2],0.99)
comp3,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,3],0.99)
comp4,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,4],0.99)
comp5,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,5],0.99)
comp6,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,6],0.99)
comp7,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,7],0.99)
comp8,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,8],0.99)
comp9,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,9],0.99)

comp0_all=comp0
comp1_all=comp1
comp2_all=comp2
comp3_all=comp3
comp4_all=comp4
comp5_all=comp5
comp6_all=comp6
comp7_all=comp7
comp8_all=comp8
comp9_all=comp9


#Save loading vectors
#############################################################################
 # left
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp0[mask_left__beta]
print "left", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_0_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp0[mask_right__beta]
print "right", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_0_right.gii"), data=tex)
   
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp1[mask_left__beta]
print "left", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_1_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp1[mask_right__beta]
print "right", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_1_right.gii"), data=tex)
  
tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp2[mask_left__beta]
print "left", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_2_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp2[mask_right__beta]
print "right", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_2_right.gii"), data=tex)

tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp3[mask_left__beta]
print "left", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_3_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp3[mask_right__beta]
print "right", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_3_right.gii"), data=tex)

tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp4[mask_left__beta]
print "left", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_4_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp4[mask_right__beta]
print "right", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_4_right.gii"), data=tex)

tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp5[mask_left__beta]
print "left", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_5_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp5[mask_right__beta]
print "right", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_5_right.gii"), data=tex)


tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp6[mask_left__beta]
print "left", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_6_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp6[mask_right__beta]
print "right", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_6_right.gii"), data=tex)


tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp7[mask_left__beta]
print "left", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_7_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp7[mask_right__beta]
print "right", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_7_right.gii"), data=tex)


tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp8[mask_left__beta]
print "left", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_8_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp8[mask_right__beta]
print "right", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_8_right.gii"), data=tex)


tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = comp9[mask_left__beta]
print "left", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_9_left.gii"), data=tex)

# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = comp9[mask_right__beta]
print "right", np.sum(tex != 0), tex.max(), tex.min()
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_9_right.gii"), data=tex)
#############################################################################


























#
##Obtain proportion of non null voxel in loading vectors across 5 fold CV
##############################################################################
##Load coomponent
#INPUT_COMPONENTS_FILE_FORMAT = os.path.join(BASE_PATH,"adni_5folds/results",'{fold}','{key}','components.npz')
#
#params=np.array(('struct_pca', '0.1', '1e-06', '0.1')) 
##params=np.array(('struct_pca', '0.1', '0.5', '0.1')) 
##params=np.array(('sparse_pca', '0.1', '1e-06', '0.1')) 
##params=np.array(('pca', '0.0', '0.0', '0.0')) 
#
#components = np.zeros((317379, 3,5))
#for fold in range(1,6):
#    key = '_'.join([str(param)for param in params])
#    name=params[0]
#    components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
#    components[:,:,fold-1] = np.load(components_filename)['arr_0']
#
#
#comp0,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,0,:],0.99)
#comp1,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,1,:],0.99)
#comp2,_ = array_utils.arr_threshold_from_norm2_ratio(components[:,2,:],0.99)
#
##Solve the problem of non-iden(tifiaibility of components, with respect to the first fold
#for i in range(1,5):
#        if np.abs(np.corrcoef(components[:,1,0],components[:,1,i])[0,1]) <  np.abs(np.corrcoef(components[:,1,0],components[:,2,i])[0,1]):
#            print "components inverted" 
#            print i
#            temp_comp2 = np.copy(components[:,2,i])
#            components[:,2,i] = components[:,1,i]
#            components[:,1,i] = temp_comp2  
#
#
################ comp 0
## left
#tex = np.zeros(mask_left__left_mesh.shape)
#tex[mask_left__left_mesh] = np.sum(comp0[mask_left__beta,:]!= 0, axis=1)
#mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_0_left_cvcountnonnull.gii"), data=tex)
## right
#tex = np.zeros(mask_right__right_mesh.shape)
#tex[mask_right__right_mesh] = np.sum(comp0[mask_right__beta,:]!= 0, axis=1)
#mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_0_right_cvcountnonnull.gii"), data=tex)
#
################ comp 1
## left
#tex = np.zeros(mask_left__left_mesh.shape)
#tex[mask_left__left_mesh] = np.sum(comp1[mask_left__beta,:]!= 0, axis=1)
#mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_1_left_cvcountnonnull.gii"), data=tex)
## right
#tex = np.zeros(mask_right__right_mesh.shape)
#tex[mask_right__right_mesh] = np.sum(comp1[mask_right__beta,:]!= 0, axis=1)
#mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_1_right_cvcountnonnull.gii"), data=tex)
#
#
################ comp 2
## left
#tex = np.zeros(mask_left__left_mesh.shape)
#tex[mask_left__left_mesh] = np.sum(comp2[mask_left__beta,:]!= 0, axis=1)
#mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_2_left_cvcountnonnull.gii"), data=tex)
## right
#tex = np.zeros(mask_right__right_mesh.shape)
#tex[mask_right__right_mesh] = np.sum(comp2[mask_right__beta,:]!= 0, axis=1)
#mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_2_right_cvcountnonnull.gii"), data=tex)


#First before launcing spyder :
#do: . /i2bm/local/Ubuntu-14.04-x86_64/brainvisa/bin/bv_env.sh
#then spyder

#Go in /i2bm/local/Ubuntu-14.04-x86_64/brainvisa/bin/anatomist and compile bv_env.py
#to set the environement variable

"""


/neurospin/brainvisa/build/Ubuntu-14.04-x86_64/trunk/bin/bv_env /neurospin/brainvisa/build/Ubuntu-14.04-x86_64/trunk/bin/anatomist tex_

/neurospin/brainvisa/build/Ubuntu-14.04-x86_64/trunk/bin/bv_env /neurospin/brainvisa/build/Ubuntu-14.04-x86_64/trunk/bin/anatomist *.gii

Pour lh/lg.pial charger les référentiels, les afficher dans lh.pial/rh.pial
Color / Rendering / Polygines face is clockwize

"""
