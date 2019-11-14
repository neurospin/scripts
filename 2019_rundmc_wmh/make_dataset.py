#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:02:06 CEST 2019


@author: edouard.duchesnay@cea.fr


## 02 / 2019

The WMH masks are all normalized to MNI space. Most of the masks are binarized. However some may contains other numbers; please includes those voxels other than 0 as WMH. These WMH masks are from the baseline RUN DMC data that have been collected in 2006. Please let me know
whether you are also interested in 2011 and 2015 normalised WMH masks as well.

## 07 / 2019

transfer_212043_files_e940935e.zip

Dear Edouard, Please find attached the segmented WMH masks registered to standard space. This file includes only patients with two follow-up scans, which brings us in total 267 patients spanning a period of 9 years. I figured if we are planning to perform longitudinal analys
es of WMH progression, we would be only interested in patients with baseline WMH scans who have consecutive scanning. I have included the standard brain atlas, which we have used to register. If you think it would be of interest to look at the whole study population (about 4
40 patients) please let me know. I could send you these WMH masks as well. Many thanks so far and I look forward seeing the results, Anil
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

########################################################################################################################
# Configure
# cookiecutter Directory structure
# https://drivendata.github.io/cookiecutter-data-science/

STUDY_PATH = '/neurospin/brainomics/2019_rundmc_wmh'
DATA_PATH = os.path.join(STUDY_PATH, 'sourcedata', 'wmhmask')

ANALYSIS_PATH = os.path.join(STUDY_PATH, 'analyses', '201909_rundmc_wmh_pca')
ANALYSIS_DATA_PATH = os.path.join(ANALYSIS_PATH, "data")
ANALYSIS_MODELS_PATH = os.path.join(ANALYSIS_PATH, "models")

CONF = dict(clust_size_thres = 20, NI="WMH", vs=1, shape=(181, 217, 181))
nii_filenames = glob.glob(os.path.join(DATA_PATH, "*", "Stdt1wSubjTempl_RUNDMC_*_WMH_to_T1_*.nii.gz"))

match_filename_re = re.compile("Stdt1wSubjTempl_RUNDMC_([0-9]+)_WMH_to_T1_(20[0-9]+).nii.gz")
match_columns = ["participant_id", "year", "nii_path"]

"""
nii_filename = nii_filenames[0]
match_filename_re.findall(os.path.basename(nii_filename))
"""

########################################################################################################################

os.makedirs(ANALYSIS_PATH, exist_ok=True)
os.makedirs(ANALYSIS_DATA_PATH, exist_ok=True)
os.makedirs(ANALYSIS_MODELS_PATH, exist_ok=True)

# Voxel size
# vs = "1mm"
#vs = "1.5mm-s8mm"
#vs = "1.5mm"

# CONF = dict(voxsize=1.5, smooth=8) # Voxel size, smoothing
#CONF = dict(voxsize=1.5, smooth=8) # Voxel size, smoothing

########################################################################################################################
# 1) Participants file & list images

"""
pop = pd.DataFrame([[match_participant_id.findall(os.path.basename(nii_filename))[0],
        nii_filename] for nii_filename in nii_filenames], columns=match_columns)
"""
pop = pd.DataFrame([list(match_filename_re.findall(os.path.basename(nii_filename))[0]) + [nii_filename]
    for nii_filename in nii_filenames], columns=match_columns)


print("N images:", pop.shape[0], "N participants:", len(pop.participant_id.unique()))
# N images: 804 N participants: 270

pop.to_csv(os.path.join(ANALYSIS_PATH, "participants.csv"), quoting=1, index=False)


########################################################################################################################
# 2) Load images
ni_imgs = list()
ni_arrs = list()

ni_imgs = [nibabel.load(os.path.join(DATA_PATH, nii_filename)) for nii_filename in pop.nii_path]

#  Fix affine transformation
for img in ni_imgs:
    img.affine[0, 3] *= -1  #  Tr X
    #  mask_img.affine[0, 1] *= -1  # flip X
    img.affine[1, 1] *= -1  #  flip Y
    # mask_img.affine[2, 3] *= -1

ref_img = ni_imgs[0]


assert ref_img.get_data().shape == CONF["shape"]


assert np.all([np.all(img.affine == ref_img.affine) for img in ni_imgs])
assert np.all([np.all(img.get_data().shape == ref_img.get_data().shape) for img in ni_imgs])
assert ref_img.header.get_zooms() == (CONF["vs"], CONF["vs"], CONF["vs"])

#############################################################################
# Save non smoothed

NI = nilearn.image.concat_imgs(ni_imgs)
NI.to_filename(os.path.join(ANALYSIS_DATA_PATH, "%s.nii.gz" % CONF["NI"]))
del NI

# Split by years
NI = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "%s.nii.gz" % CONF["NI"]))

NI_arr = NI.get_data()
NI_arr.shape

for year in set(pop.year):
    print(year)
    msk = pop.year == year
    pop_ = pop[msk]
    NI_arr_ = NI_arr[:, :, :, msk]
    nibabel.Nifti1Image(NI_arr_, affine=ref_img.affine).to_filename(os.path.join(ANALYSIS_DATA_PATH, "%s_%s.nii.gz" % (CONF["NI"], year)))
    pop_.to_csv(os.path.join(ANALYSIS_DATA_PATH, "participants_%s.csv" % year), index=False)

#############################################################################
# 2) Smooth image
if "smooth" in CONF:
    gms_imgs = nilearn.image.smooth_img(ni_imgs, CONF["smooth"])
    NIs = nilearn.image.concat_imgs(gms_imgs)
    NIs.to_filename(os.path.join(ANALYSIS_DATA_PATH, "%s_s%s.nii.gz" % (CONF["NI"], CONF["smooth"])))

#############################################################################
# 3) Compute mask implicit mask

"""
NI_flat = np.vstack([img.get_data().ravel() for img in ni_imgs])
del ni_imgs

mask_arr = (np.max(NI_flat, axis=0) > 0) #& (np.std(GM, axis=0) >= 1e-6)
#mask_arr = (np.mean(GM, axis=0) >= 0.1) & (np.std(GM, axis=0) >= 1e-6)

mask_arr = mask_arr.reshape(CONF['shape'])
#assert mask_arr.sum() == 64332
"""
mask_arr = (np.max(NI_arr, axis=3) > 0)
assert mask_arr.sum() == 298098
assert mask_arr.sum() == 432004

#############################################################################
# atlas: make sure we are within atlas
ref_img.to_filename(os.path.join(ANALYSIS_DATA_PATH, "ref.nii.gz"))

atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=os.path.join(ANALYSIS_DATA_PATH, "ref.nii.gz"),
    output=os.path.join(ANALYSIS_DATA_PATH, "atlas_harvard_oxford.nii.gz"))

cereb = brainomics.image_atlas.resample_atlas_bangor_cerebellar(
    ref=os.path.join(ANALYSIS_DATA_PATH, "ref.nii.gz"),
    output=os.path.join(ANALYSIS_DATA_PATH, "atlas_cerebellar.nii.gz"))

mask_arr = ((atlas.get_data() + cereb.get_data()) != 0) & mask_arr
assert mask_arr.sum() == 423143

#############################################################################
# Remove small branches

mask_arr = scipy.ndimage.binary_opening(mask_arr)
assert mask_arr.sum() == 372353

#############################################################################
# Avoid isolated clusters: remove all cluster smaller that 10

mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
print([[lab, np.sum(mask_clustlabels_arr == lab)] for lab in  np.unique(mask_clustlabels_arr)[1:]])
# [[1, 116276], [2, 7], [3, 7], [4, 19], [5, 28], [6, 13], [7, 72], [8, 7], [9, 16], [10, 51], [11, 12], [12, 7], [13, 35], [14, 12], [15, 30], [16, 12], [17, 28], [18, 62], [19, 7], [20, 7], [21, 12], [22, 12], [23, 7], [24, 7], [25, 7], [26, 7], [27, 7], [28, 14], [29, 146], [30, 7], [31, 12], [32, 12], [33, 12], [34, 7], [35, 7], [36, 7], [37, 16], [38, 12], [39, 85], [40, 42], [41, 30], [42, 7], [43, 7], [44, 7], [45, 32], [46, 54], [47, 300], [48, 7], [49, 7], [50, 12], [51, 16], [52, 17], [53, 491], [54, 7], [55, 7], [56, 7], [57, 7], [58, 7], [59, 14], [60, 17], [61, 7], [62, 118], [63, 7], [64, 12], [65, 12], [66, 23], [67, 12], [68, 7], [69, 7], [70, 14], [71, 12], [72, 30], [73, 122], [74, 7], [75, 29], [76, 32], [77, 51], [78, 24], [79, 17], [80, 7], [81, 57], [82, 12], [83, 19], [84, 266], [85, 7], [86, 32], [87, 538], [88, 7], [89, 27], [90, 13], [91, 31], [92, 27], [93, 12], [94, 12], [95, 12], [96, 129], [97, 125755], [98, 65], [99, 7], [100, 7], [101, 39], [102, 7], [103, 74], [104, 7], [105, 113], [106, 197], [107, 7], [108, 17], [109, 68], [110, 7], [111, 7], [112, 7], [113, 50], [114, 7], [115, 7], [116, 7], [117, 7], [118, 83], [119, 7], [120, 12], [121, 324], [122, 7], [123, 14], [124, 122], [125, 7], [126, 611], [127, 7], [128, 108], [129, 12], [130, 17], [131, 12], [132, 7], [133, 95], [134, 12], [135, 19], [136, 106], [137, 7], [138, 7], [139, 24], [140, 7], [141, 7], [142, 47], [143, 7], [144, 263], [145, 12], [146, 12], [147, 7], [148, 17], [149, 46], [150, 17], [151, 7], [152, 7], [153, 7], [154, 7], [155, 12], [156, 7], [157, 76], [158, 25], [159, 12], [160, 7], [161, 7], [162, 7], [163, 12], [164, 7], [165, 12], [166, 19], [167, 43], [168, 7], [169, 12], [170, 7], [171, 7], [172, 7], [173, 7], [174, 7], [175, 7], [176, 7], [177, 14], [178, 12], [179, 7], [180, 79], [181, 12], [182, 7], [183, 29]]


labels = np.unique(mask_clustlabels_arr)[1:]
for lab in labels:
    clust_size = np.sum(mask_clustlabels_arr == lab)
    if clust_size <= CONF['clust_size_thres']:
        mask_arr[mask_clustlabels_arr == lab] = False


mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
labels = np.unique(mask_clustlabels_arr)[1:]

print([[lab, np.sum(mask_clustlabels_arr == lab)] for lab in labels])
#[[1, 95], [2, 179484], [3, 28], [4, 24], [5, 49], [6, 35], [7, 56], [8, 23], [9, 48], [10, 50], [11, 425], [12, 68], [13, 22], [14, 79], [15, 151], [16, 33], [17, 78], [18, 54], [19, 308], [20, 22], [21, 29], [22, 118], [23, 125], [24, 122], [25, 48], [26, 51], [27, 266], [28, 31], [29, 129], [30, 58], [31, 187287], [32, 27], [33, 24], [34, 39], [35, 29], [36, 68], [37, 30], [38, 21], [39, 83], [40, 324], [41, 122], [42, 108], [43, 37], [44, 48], [45, 106], [46, 24], [47, 63], [48, 47], [49, 263], [50, 37], [51, 46], [52, 23], [53, 76], [54, 43], [55, 22], [56, 68], [57, 70], [58, 34]]

assert mask_arr.sum() == 371278

nibabel.Nifti1Image(mask_arr.astype(int), affine=ref_img.affine).to_filename(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))

# Plot mask
mask_img = nibabel.Nifti1Image(mask_arr.astype(float), affine=ref_img.affine)
nslices = 8
fig = plt.figure(figsize=(19.995, 11.25))
ax = fig.add_subplot(311)
plotting.plot_anat(mask_img, display_mode='z', cut_coords=nslices, figure=fig, axes=ax)
ax = fig.add_subplot(312)
plotting.plot_anat(mask_img, display_mode='y', cut_coords=nslices, figure=fig,axes=ax)
ax = fig.add_subplot(313)
plotting.plot_anat(mask_img, display_mode='x', cut_coords=nslices, figure=fig,axes=ax)
plt.savefig(os.path.join(ANALYSIS_DATA_PATH, "mask.png"))

########################################################################################################################
# Reload All
mask_img = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))
mask_arr = mask_img.get_data() == 1
pop = pd.read_csv(os.path.join(ANALYSIS_PATH, "participants.csv"))
NI = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "%s.nii.gz") % CONF["NI"])
NI_arr =  NI.get_data()
NI_arr_msk = NI_arr[mask_arr].T


# Check shape
assert NI_arr.shape == tuple(list(CONF["shape"]) + [pop.shape[0]])

# Save map
map_arr = np.zeros(CONF["shape"])
map_arr[mask_arr] = NI_arr_msk.mean(axis=0)
map_img = nibabel.Nifti1Image(map_arr, mask_img.affine)
map_img.to_filename(os.path.join(ANALYSIS_DATA_PATH, "%s_mean.nii.gz" % CONF["NI"]))

# Check that appyling mask on save 4D GM == mask ravel image
# assert np.allclose(NI_arr_msk, NI_flat[:, mask_arr.ravel()])


########################################################################################################################
# Reload 2006
YEAR = '2006'
mask_img = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))
mask_arr = mask_img.get_data() == 1
pop = pd.read_csv(os.path.join(ANALYSIS_DATA_PATH, "participants_%s.csv" % YEAR))
NI = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "%s_%s.nii.gz" % (CONF["NI"], YEAR)))
NI_arr =  NI.get_data()
NI_arr_msk = NI_arr[mask_arr].T

# Check shape
assert NI_arr.shape == tuple(list(CONF["shape"]) + [pop.shape[0]])

# Save map
map_arr = np.zeros(CONF["shape"])
map_arr[mask_arr] = NI_arr_msk.mean(axis=0)
map_img = nibabel.Nifti1Image(map_arr, mask_img.affine)
map_img.to_filename(os.path.join(ANALYSIS_DATA_PATH, "%s_%s_mean.nii.gz" % (CONF["NI"], year)))

# Check that appyling mask on save 4D GM == mask ravel image
# assert np.allclose(NI_arr_msk, NI_flat[:, mask_arr.ravel()])

########################################################################################################################
# Save Linear operator
from parsimony.utils.linalgs import LinearOperatorNesterov
mask_img = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "mask.nii.gz"))

Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(ANALYSIS_DATA_PATH, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(ANALYSIS_DATA_PATH, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 11.974760295502465)
