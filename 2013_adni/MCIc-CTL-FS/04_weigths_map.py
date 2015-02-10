# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 19:24:37 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import os
import numpy as np
import json
import glob
from brainomics import array_utils
import brainomics.mesh_processing as mesh_utils
import shutil

BASE_PATH = "/neurospin/brainomics/2013_adni"
WD = os.path.join(BASE_PATH, "MCIc-CTL-FS")
OUTPUT = "weight_map"

os.chdir(WD)
if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)

"""
TEMPLATE_PATH = os.path.join(BASE_PATH, "freesurfer_template")
shutil.copyfile(os.path.join(TEMPLATE_PATH, "lh.pial.gii"), os.path.join(OUPTUT, "lh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "rh.pial.gii"), os.path.join(OUPTUT, "rh.pial.gii"))
"""

config  = json.load(open("config_5cv.json"))

cor_l, tri_l = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "lh.pial.gii"))
cor_r, tri_r = mesh_utils.mesh_arrays(os.path.join(OUTPUT, "rh.pial.gii"))

cor_both, tri_both = mesh_utils.mesh_arrays(config["structure"]["mesh"])
mask_all_mesh = np.load(config["structure"]["mask"])
assert mask_all_mesh.shape[0] == cor_both.shape[0] == cor_l.shape[0] * 2 == cor_r.shape[0] * 2

# concat was initialy: cor = np.vstack([cor_l, cor_r])
mask_left_mesh = np.arange(mask_all_mesh.shape[0])  < mask_all_mesh.shape[0] / 2
mask_left_mesh[np.logical_not(mask_all_mesh)] = False
mask_right_mesh = np.arange(mask_all_mesh.shape[0]) >= mask_all_mesh.shape[0] / 2
mask_right_mesh[np.logical_not(mask_all_mesh)] = False
assert mask_all_mesh.sum() ==  (mask_left_mesh.sum() + mask_right_mesh.sum())
# compute mask from beta to left/right
a = np.zeros(mask_all_mesh.shape, int)
a[mask_left_mesh] = 1
a[mask_right_mesh] = 2
mask_left_beta = a[mask_all_mesh] == 1
mask_right_beta = a[mask_all_mesh] == 2
assert (mask_left_beta.sum() + mask_right_beta.sum()) == mask_right_beta.shape[0] == mask_all_mesh.sum() 


todo = dict(
l2	  = dict(param=(0.01, 0.0, 1.0, 0.0, -1.0)),
l2tv	  = dict(param=(0.01, 0.0, 0.5, 0.5, -1.0)),
l1	  = dict(param=(0.01, 1.0, 0.0, 0.0, -1.0)),
l1tv	  = dict(param=(0.01, 0.5, 0.0, 0.5, -1.0)),
tv	  = dict(param=(0.01, 0.0, 0.0, 1.0, -1.0)),
l1l2	  = dict(param=(0.01, 0.5, 0.5, 0.0, -1.0)),
l1l2tv  = dict(param=(0.01, 0.35, 0.35, 0.3, -1.0)))

for k in todo:
    todo[k]["path"] = os.path.join("5cv", "*", "_".join([str(p) for p in todo[k]["param"]]))

#############################################################################
## CV 0
for k in todo:
    Betas = np.vstack([array_utils.arr_threshold_from_norm2_ratio(
    np.load(os.path.join(filename, "beta.npz"))['arr_0'][config["penalty_start"]:, :].ravel(), .99)[0]
        for filename in glob.glob(todo[k]["path"])])
    # left
    arr = np.zeros(mask_left_mesh.shape)
    arr[mask_left_mesh] = Betas[0, mask_left_beta]
    mesh_utils.save_texture(path=os.path.join(OUTPUT, "tex_%s_left.gii" % k), data=arr, intent='NIFTI_INTENT_TTEST')
    arr[mask_left_mesh] = np.sum(Betas[1:, mask_left_beta] != 0, axis=0)
    mesh_utils.save_texture(path=os.path.join(OUTPUT, "tex_%s_left_cvcountnonnull.gii" % k), data=arr, intent='NIFTI_INTENT_TTEST')
    # right
    arr = np.zeros(mask_right_mesh.shape)
    arr[mask_right_mesh] = Betas[0, mask_right_beta]
    mesh_utils.save_texture(path=os.path.join(OUTPUT, "tex_%s_right.gii" % k), data=arr, intent='NIFTI_INTENT_TTEST')
    arr[mask_right_mesh] = np.sum(Betas[1:, mask_right_beta] != 0, axis=0)
    mesh_utils.save_texture(path=os.path.join(OUTPUT, "tex_%s_right_cvcountnonnull.gii" % k), data=arr, intent='NIFTI_INTENT_TTEST')
