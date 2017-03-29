# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:20:13 2016

@author: ad247405
"""


import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil


BASE_PATH = '/neurospin/brainomics/2016_deptms'

INPUT_FS = "/neurospin/brainomics/2016_deptms/results/Freesurfer/freesurfer_assembled_data_fsaverage"

TEMPLATE_PATH = "/neurospin/brainomics/2016_deptms/results/Freesurfer/freesurfer_template"

INPUT_CSV = "/neurospin/brainomics/2016_deptms/results/Freesurfer/population.csv"
OUTPUT = "/neurospin/brainomics/2016_deptms/results/Freesurfer/data"

# Read pop csv
pop = pd.read_csv(INPUT_CSV)

#############################################################################
## Build mesh template
import brainomics.mesh_processing as mesh_utils
cor_l, tri_l = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "lh.pial.gii"))
cor_r, tri_r = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "rh.pial.gii"))
cor = np.vstack([cor_l, cor_r])
tri_r += cor_l.shape[0]
tri = np.vstack([tri_l, tri_r])
mesh_utils.mesh_from_arrays(cor, tri, path=os.path.join(TEMPLATE_PATH, "lrh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "lrh.pial.gii"), os.path.join(OUTPUT , "lrh.pial.gii"))

#############################################################################
# Read images
n = len(pop)
assert n == 34
Z = np.zeros((n, 2)) #Age,sex 
y = np.zeros((n, 1)) # DX
surfaces = list()

for i, ID in enumerate(pop['subject']):
    cur = pop[pop["subject"] == ID]
ics
    mri_path_lh = cur["mri_path_lh"].values[0]
    mri_path_rh = cur["mri_path_rh"].values[0]
    left = nb.load(mri_path_lh).get_data().ravel()
    right = nb.load(mri_path_rh).get_data().ravel()
    surf = np.hstack([left, right])
    surfaces.append(surf)
    y[i, 0] = cur["Response.num"]
    Z[i, :] = np.asarray(cur[["Age", "Sex"]]).ravel()
    print i


Xtot = np.vstack(surfaces)
assert Xtot.shape == (34, 327684)

mask = ((np.max(Xtot, axis=0) > 0) & (np.std(Xtot, axis=0) > 1e-2))
assert mask.sum() ==  314783

np.save(os.path.join(OUTPUT, "mask.npy"), mask)

X = Xtot[:, mask]
assert X.shape == (34, 314783)




#############################################################################
# Some basic stat before centering/scaling
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
print "Means:", means.min(), means.max(), means.mean()
print "Std:",  stds.min(), stds.max(), stds.mean()
print "Mins:", mins.min(), mins.max(), mins.mean()
print "Maxs:", maxs.min(), maxs.max(), maxs.mean(), (maxs == 0).sum()



arr = np.zeros(mask.shape); arr[mask] = means
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "mean.gii"), data=arr)#, intent='NIFTI_INTENT_NONE')
arr = np.zeros(mask.shape); arr[mask] = stds
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "std.gii"), data=arr)#, intent='NIFTI_INTENT_NONE')
arr = np.zeros(mask.shape); arr[mask] = maxs
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "max.gii"), data=arr)#, intent='NIFTI_INTENT_NONE')


#############################################################################

X = np.hstack([Z, X])
assert X.shape == (34, 314785)


# Center/scale 
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT, "X.npy"), X)
fh = open(os.path.join(OUTPUT, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i surface points' % \
    (n, p, mask.sum()))
fh.close()

np.save(os.path.join(OUTPUT, "y.npy"), y)



##############################################################################



