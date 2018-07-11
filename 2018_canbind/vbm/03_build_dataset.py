#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:20:32 2017


5@author: ad24740

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- population.csv

OUTPUT:
- mask.nii
- y.npy
- X.npy = Age + Gender + Voxels
"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
import brainomics.image_atlas
import shutil
import mulm
import sklearn
import re
from nilearn import plotting
import matplotlib.pyplot as plt
import scipy, scipy.ndimage
#import nilearn.plotting
from nilearn import datasets, plotting, image

# cookiecutter Directory structure
# https://drivendata.github.io/cookiecutter-data-science/

WD = '/neurospin/psy/canbind'
#BASE_PATH = '/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm'
INPUT_CSV= os.path.join(WD, "data", "participants.tsv")

# Voxel size
# vs = "1mm"
vs = "1.5mm-s8mm"
#vs = "1.5mm"

OUTPUT = os.path.join(WD, "models", "vbm_%s" % vs)

#############################################################################
# 1) Build Raw dataset
#
# Read all images
#image_filenames = glob.glob(os.path.join(WD, "data", "derivatives/spmsegment/sub-*/ses-*/anat/mwc1sub-*_ses-*_T1w_%s.nii" % vs))
image_filenames = glob.glob(os.path.join(WD, "data", "derivatives/spmsegment/sub-*/ses-*/anat/*mwc1sub-*_ses-*_T1w_%s.nii" % vs))
regexp = re.compile(".+(sub-.+)/(ses-.+)/anat/.mwc1sub-.+_ses-.+_T1w_.+.nii")
img_tab = pd.DataFrame([list(regexp.findall(f)[0]) + [f] for f in image_filenames], columns=["participant_id", "session", "path"])
assert img_tab.shape == (806, 3)

# Read pop csv
pop = pd.read_csv(INPUT_CSV, sep="\t")

# Merge
pop = pd.merge(pop, img_tab, how="right") # keep only thoses with images
assert pop.shape[0] == 806
assert np.sum(pop["group"] == "Treatment") == 489

#############################################################################
# 2) Read images
n = len(pop)
assert n == 806

gm_imgs = list()
gm_arrs = list()

for index, row in pop.iterrows():
    print(row["participant_id"], row["path"])
    img =  nibabel.load(row["path"])
    gm_imgs.append(img)
    gm_arrs.append(img.get_data().ravel())


shape = img.get_data().shape

#############################################################################
# 3) Compute mask implicit Masking
Xtot = np.vstack(gm_arrs)
#mask_arr = (np.std(Xtot, axis=0) > 1e-7)

mask_arr = (np.mean(Xtot, axis=0) >= 0.1) & (np.std(Xtot, axis=0) >= 1e-6)
mask_arr = mask_arr.reshape(shape)

#nibabel.Nifti1Image(np.min(Xtot, axis=0).reshape(shape).astype(float), affine=img.affine).to_filename(os.path.join(OUTPUT, "min_all.nii.gz"))
nibabel.Nifti1Image(np.mean(Xtot, axis=0).reshape(shape).astype(float), affine=img.affine).to_filename(os.path.join(OUTPUT, "mean.nii.gz"))
nibabel.Nifti1Image(np.std(Xtot, axis=0).reshape(shape).astype(float), affine=img.affine).to_filename(os.path.join(OUTPUT, "std.nii.gz"))

# Avoid isolated clusters: remove all cluster smaller that 5
mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
print([[lab, np.sum(mask_clustlabels_arr == lab)] for lab in  np.unique(mask_clustlabels_arr)[1:]])
# 1.5mm
# [[1, 397559], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 2], [11, 1], [12, 2], [13, 1], [14, 1], [15, 1], [16, 1], [17, 1], [18, 1], [19, 1], [20, 1], [21, 1], [22, 1], [23, 1], [24, 1], [25, 1], [26, 1], [27, 1], [28, 1], [29, 1], [30, 1], [31, 1], [32, 2], [33, 2], [34, 1], [35, 2], [36, 2], [37, 4], [38, 2], [39, 2], [40, 1], [41, 4], [42, 2], [43, 1], [44, 1], [45, 1], [46, 1], [47, 1], [48, 1], [49, 1], [50, 1], [51, 2], [52, 1], [53, 1], [54, 1], [55, 1], [56, 1], [57, 1], [58, 1], [59, 2], [60, 1], [61, 2], [62, 2], [63, 1], [64, 1], [65, 1], [66, 2], [67, 2], [68, 1], [69, 1], [70, 1], [71, 1], [72, 1], [73, 1], [74, 2], [75, 1], [76, 1], [77, 1], [78, 2], [79, 1]]
# 1.5mm-s8mm
# [[1, 528502]]

clust_size_thres = 5
labels = np.unique(mask_clustlabels_arr)[1:]
for lab in labels:
    clust_size = np.sum(mask_clustlabels_arr == lab)
    if clust_size <= clust_size_thres:
        mask_arr[mask_clustlabels_arr == lab] = False


mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
labels = np.unique(mask_clustlabels_arr)[1:]

print([[lab, np.sum(mask_clustlabels_arr == lab)] for lab in labels])
# [[1, 1308898]] 1mm3
# [[1, 397559]] 1.5mm3
# [[1, 528502]] 1.5mm-s8mm
if vs == "1.5mm":
    assert mask_arr.sum() == 397559
# 1.5mm-s8mm
if vs == "1.5mm-s8mm":
    assert mask_arr.sum() == 528502

nibabel.Nifti1Image(mask_arr.astype(int), affine=img.affine).to_filename(os.path.join(OUTPUT, "mask.nii.gz"))
np.save(os.path.join(OUTPUT, "mask.npy"), mask_arr)


# Plot mask
mask_img = nibabel.Nifti1Image(mask_arr.astype(float), affine=img.affine)
nslices = 8
fig = plt.figure(figsize=(19.995, 11.25))
ax = fig.add_subplot(211)
plotting.plot_anat(mask_img, display_mode='z', cut_coords=nslices, figure=fig, axes=ax)
ax = fig.add_subplot(212)
plotting.plot_anat(mask_img, display_mode='y', cut_coords=nslices, figure=fig,axes=ax)
plt.savefig(os.path.join(OUTPUT, "mask.png"))

# apply mask
Xraw = Xtot[:, mask_arr.ravel()]
np.save(os.path.join(OUTPUT, "Xraw.npy"), Xraw)

# Save
set(pop.site)
mean_sites = dict()
for s in set(pop.site):
    m = pop.site == s
    mean_sites[s] = Xraw[m, :].mean(axis=0)

np.savez_compressed(os.path.join(OUTPUT, "mean_sites.npz"), **mean_sites)
d = np.load(os.path.join(OUTPUT, "mean_sites.npz"))
# ['MCU', 'CAM', 'MCM', 'UBC', 'UCA', 'QNS', 'TWH', 'TGH']
assert np.all(np.array([k for k in d]) == np.array(['UCA', 'TWH', 'MCM', 'CAM', 'UBC', 'TGH', 'MCU', 'QNS']))
np.array([mean_sites[k].mean() for k in mean_sites])


if vs == "1.5mm":
    assert np.allclose(np.array([d[k].mean() for k in d]),
                       np.array([ 0.52133739,  0.50380158,  0.51773959,  0.53550243,  0.54018521,
            0.51645875,  0.50088221,  0.51606596], dtype=float))
# 1.5mm-s8mm
if vs == "1.5mm-s8mm":
    assert np.allclose(np.array([d[k].mean() for k in d]),
                       np.array([ 0.38460582,  0.38262334,  0.38161469,  0.39497823,  0.39820373,
        0.36950105,  0.38066757,  0.37143078], dtype=float))


pop.to_csv(os.path.join(OUTPUT, "population.csv"), index=False)

assert pop.shape[0] == Xraw.shape[0]


