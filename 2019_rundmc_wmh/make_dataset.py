#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:02:06 CEST 2019


@author: edouard.duchesnay@cea.fr

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
import brainomics.image_atlas
import nilearn
import parsimony.functions.nesterov.tv as nesterov_tv
#from parsimony.utils.linalgs import LinearOperatorNesterov

# cookiecutter Directory structure
# https://drivendata.github.io/cookiecutter-data-science/

STUDY_PATH = '/neurospin/brainomics/2019_rundmc_wmh'
DATA_PATH = os.path.join(STUDY_PATH, 'sourcedata', 'wmhmask')

ANALYSIS_PATH = os.path.join(STUDY_PATH, 'analysis', '201905_rundmc_wmh_pca')
ANALYSIS_DATA_PATH = os.path.join(ANALYSIS_PATH, "data")
ANALYSIS_MODELS_PATH = os.path.join(ANALYSIS_PATH, "models")

os.makedirs(ANALYSIS_PATH, exist_ok=True)
os.makedirs(ANALYSIS_DATA_PATH, exist_ok=True)
os.makedirs(ANALYSIS_MODELS_PATH, exist_ok=True)

# Voxel size
# vs = "1mm"
#vs = "1.5mm-s8mm"
#vs = "1.5mm"
CONF = dict(voxsize=1.5, smooth=8) # Voxel size, smoothing
#CONF = dict(voxsize=1.5, smooth=8) # Voxel size, smoothing

# "2019_hbn_vbm_predict-ari_vs%.1f-s%i" % (CONF["voxsize"], CONF["smoothing"]
#WD = os.path.join(ANALYSIS_PATH, "data")
#os.makedirs(WD, exist_ok=True)

#############################################################################
# 2) Read images
nii_filenames = glob.glob(os.path.join(DATA_PATH, "wrRUNDMC_*_WMHmask_2006.nii.gz"))

match_participant_id = re.compile("wrRUNDMC_([0-9]+)_WMHmask_2006.nii.gz")

pop = pd.DataFrame([[match_participant_id.findall(os.path.basename(nii_filename))[0],
        os.path.basename(nii_filename)] for nii_filename in nii_filenames], columns=["participant_id", "nii_path"])


pop.to_csv(os.path.join(ANALYSIS_PATH, "participants.csv"), quoting=1, index=False)

gm_imgs = list()
gm_arrs = list()

gm_imgs = [nibabel.load(os.path.join(DATA_PATH, nii_filename)) for nii_filename in pop.nii_path]
ref_img = gm_imgs[0]
shape = ref_img.get_data().shape
shape == (91, 109, 91)


assert np.all([np.all(img.affine == ref_img.affine) for img in gm_imgs])
assert np.all([np.all(img.get_data().shape == ref_img.get_data().shape) for img in gm_imgs])
assert ref_img.header.get_zooms() == (2.0, 2.0, 2.0)

#############################################################################
# Save non smoothed
WMH = nilearn.image.concat_imgs(gm_imgs)
WMH.to_filename(os.path.join(ANALYSIS_DATA_PATH, "WMH.nii.gz"))

#############################################################################
# Smooth image
gms_imgs = nilearn.image.smooth_img(gm_imgs, CONF["smooth"])

WMHs = nilearn.image.concat_imgs(gms_imgs)
WMHs.to_filename(os.path.join(ANALYSIS_DATA_PATH, "WMH_s%s.nii.gz" % CONF["smooth"]))

#############################################################################
# 3) Compute mask implicit mask
WMH_flat = np.vstack([img.get_data().ravel() for img in gm_imgs])
#mask_arr = (np.std(GM, axis=0) > 1e-7)

mask_arr = (np.max(WMH_flat, axis=0) > 0) #& (np.std(GM, axis=0) >= 1e-6)
#mask_arr = (np.mean(GM, axis=0) >= 0.1) & (np.std(GM, axis=0) >= 1e-6)

mask_arr = mask_arr.reshape(shape)
assert mask_arr.sum() == 64332

#############################################################################
# atlas: make sure we are within atlas

atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=os.path.join(DATA_PATH,pop.loc[0, "nii_path"]),
    output=os.path.join(ANALYSIS_DATA_PATH, "atlas_harvard_oxford.nii.gz"))

cereb = brainomics.image_atlas.resample_atlas_bangor_cerebellar(
    ref=os.path.join(DATA_PATH,pop.loc[0, "nii_path"]),
    output=os.path.join(ANALYSIS_DATA_PATH, "atlas_cerebellar.nii.gz"))

mask_arr = ((atlas.get_data() + cereb.get_data()) != 0) & mask_arr
assert mask_arr.sum() == 62695

#############################################################################
# Remove small branches

mask_arr = scipy.ndimage.binary_opening(mask_arr)
assert mask_arr.sum() == 51791

#############################################################################
# Avoid isolated clusters: remove all cluster smaller that 10

mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
print([[lab, np.sum(mask_clustlabels_arr == lab)] for lab in  np.unique(mask_clustlabels_arr)[1:]])
# s8mm
# [[1, 12], [2, 25138], [3, 65], [4, 23], [5, 112], [6, 7], [7, 7], [8, 7], [9, 56], [10, 20], [11, 34], [12, 20], [13, 7], [14, 28], [15, 7], [16, 7], [17, 7], [18, 7], [19, 7], [20, 7], [21, 25875], [22, 7], [23, 7], [24, 7], [25, 14], [26, 17], [27, 16], [28, 29], [29, 7], [30, 7], [31, 7], [32, 21], [33, 117], [34, 12], [35, 7], [36, 7], [37, 7], [38, 28], [39, 7], [40, 7], [41, 7]]

clust_size_thres = 10
labels = np.unique(mask_clustlabels_arr)[1:]
for lab in labels:
    clust_size = np.sum(mask_clustlabels_arr == lab)
    if clust_size <= clust_size_thres:
        mask_arr[mask_clustlabels_arr == lab] = False


mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
labels = np.unique(mask_clustlabels_arr)[1:]

print([[lab, np.sum(mask_clustlabels_arr == lab)] for lab in labels])
# [[1, 12], [2, 25138], [3, 65], [4, 23], [5, 112], [6, 56], [7, 20], [8, 34], [9, 20], [10, 28], [11, 25875], [12, 14], [13, 17], [14, 16], [15, 29], [16, 21], [17, 117], [18, 12], [19, 28]]

assert mask_arr.sum() == 51637

nibabel.Nifti1Image(mask_arr.astype(int), affine=ref_img.affine).to_filename(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))


# Plot mask
mask_img = nibabel.Nifti1Image(mask_arr.astype(float), affine=ref_img.affine)
nslices = 8
fig = plt.figure(figsize=(19.995, 11.25))
ax = fig.add_subplot(211)
plotting.plot_anat(mask_img, display_mode='z', cut_coords=nslices, figure=fig, axes=ax)
ax = fig.add_subplot(212)
plotting.plot_anat(mask_img, display_mode='y', cut_coords=nslices, figure=fig,axes=ax)
plt.savefig(os.path.join(ANALYSIS_DATA_PATH, "mask.png"))


# Check that appyling mask on save 4D GM == mask ravel image

mask_img = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))
mask_arr = mask_img.get_data() == 1
WMH_ = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "WMH.nii.gz"))
WMH_arr_ = WMH_.get_data()[mask_arr].T
assert np.allclose(WMH_arr_, WMH_flat[:, mask_arr.ravel()])

########################################################################################################################
# Save Linearoperator
from parsimony.utils.linalgs import LinearOperatorNesterov

mask_img = nibabel.Nifti1Image(mask_arr.astype(float), affine=ref_img.affine)
Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(ANALYSIS_DATA_PATH, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(ANALYSIS_DATA_PATH, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 11.920102051821967)
