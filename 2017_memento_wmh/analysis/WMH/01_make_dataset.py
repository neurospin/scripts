# -*- coding: utf-8 -*-
"""

Creates a CSV file for the population.

"""
import os
import numpy as np
import pandas as pd
import re
import glob

import nibabel
import matplotlib.pyplot as plt
import scipy, scipy.ndimage
from nilearn import datasets, plotting, image
import brainomics.image_atlas
import nilearn
import parsimony.functions.nesterov.tv as nesterov_tv


SRC_PATH = "/home/ad247405/git/scripts/2017_memento_wmh/analysis"
INPUT_CLINIC_FILENAME = '/neurospin/brainomics/2017_memento/documents/MEMENTO_BASELINE_ML_for_catidb.csv'
INPUT_WMHs = "/neurospin/cati/MEMENTO/WMH_registration_MNI_space/DB_Memento_Recalage_MNI_M000_v1/*/Espace_MNI/wrwmh_lesion_mask_*.nii.gz"
ANALYSIS_PATH = "/neurospin/brainomics/2017_memento/analysis/WMH"

ANALYSIS_DATA_PATH = os.path.join(ANALYSIS_PATH, "data")
ANALYSIS_MODELS_PATH = os.path.join(ANALYSIS_PATH, "models")
PARTICIPANTS_CSV = os.path.join(ANALYSIS_PATH , "population.csv")

CONF = dict(clust_size_thres=20, NI="WMH", vs=1.5, shape=(121, 145, 121))

########################################################################################################################
os.makedirs(ANALYSIS_DATA_PATH, exist_ok=True)
os.makedirs(ANALYSIS_MODELS_PATH, exist_ok=True)

########################################################################################################################
# Build population file

# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
clinic = clinic.rename(columns={"usubjid": "participant_id"})

# list nii files
input_subjects_wmh = dict()
nii_filenames = glob.glob(INPUT_WMHs)
match_filename_re = re.compile(".+_v1/(.+)/Espace_MNI.+")
pop_columns = ["participant_id", "nii_path"]

"""
nii_filename = nii_filenames[0]
match_filename_re.findall(nii_filename)
"""

pop = pd.DataFrame([list(match_filename_re.findall(nii_filename)) + [nii_filename]
    for nii_filename in nii_filenames], columns=pop_columns)

assert pop.shape == (1757, 2)

pop = pd.merge(pop, clinic, on="participant_id", how='inner') # Keep only thos with image
assert pop.shape == (1755, 26)

# Save population information
pop.to_csv(PARTICIPANTS_CSV, index=False)

########################################################################################################################
# 2) Build dataset

pop = pd.read_csv(PARTICIPANTS_CSV)

########################################################################################################################
# 2.1) Load images
ni_imgs = list()
ni_arrs = list()

ni_imgs = [nibabel.load(nii_filename) for nii_filename in pop.nii_path]

"""
#  Fix affine transformation
for img in ni_imgs:
    img.affine[0, 3] *= -1  #  Tr X
    #  mask_img.affine[0, 1] *= -1  # flip X
    img.affine[1, 1] *= -1  #  flip Y
    # mask_img.affine[2, 3] *= -1
"""

ref_img = ni_imgs[0]

# Check
assert ref_img.get_data().shape == CONF["shape"]
assert np.all([np.all(img.affine == ref_img.affine) for img in ni_imgs])
assert np.all([np.all(img.get_data().shape == ref_img.get_data().shape) for img in ni_imgs])
assert ref_img.header.get_zooms() == (CONF["vs"], CONF["vs"], CONF["vs"])

# Check values in 1, 0
assert np.all([(np.nanmax(img.get_data()), np.nanmin(img.get_data())) == (1, 0) for img in ni_imgs])
assert np.all([np.all(np.unique(img.get_data()) == (0, 1)) for img in ni_imgs])

ref_arr = ref_img.get_data()

#############################################################################
# Save non smoothed
NI = nilearn.image.concat_imgs(ni_imgs)
NI.to_filename(os.path.join(ANALYSIS_DATA_PATH, "%s.nii.gz" % CONF["NI"]))
del NI

#############################################################################
# 2.2) Smooth image
if "smooth" in CONF:
    gms_imgs = nilearn.image.smooth_img(ni_imgs, CONF["smooth"])
    NIs = nilearn.image.concat_imgs(gms_imgs)
    NIs.to_filename(os.path.join(ANALYSIS_DATA_PATH, "%s_s%s.nii.gz" % (CONF["NI"], CONF["smooth"])))

#############################################################################
# 3) Compute mask implicit mask

NI_flat = np.vstack([img.get_data().ravel() for img in ni_imgs])
del ni_imgs

thresh = 1 / 1000 * NI_flat.shape[0]
mask_arr = (np.sum(NI_flat, axis=0) > thresh) #& (np.std(GM, axis=0) >= 1e-6)
# mask_arr = (np.max(NI_flat, axis=0) > 0) #& (np.std(GM, axis=0) >= 1e-6)
#mask_arr = (np.mean(GM, axis=0) >= 0.1) & (np.std(GM, axis=0) >= 1e-6)

mask_arr = mask_arr.reshape(CONF['shape'])
#assert mask_arr.sum() == 64332

assert mask_arr.sum() == 122427

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
assert mask_arr.sum() == 121761

#############################################################################
# Remove small branches

mask_arr = scipy.ndimage.binary_opening(mask_arr)
assert mask_arr.sum() == 116095

#############################################################################
# Avoid isolated clusters: remove all cluster smaller that 10

mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
print([[lab, np.sum(mask_clustlabels_arr == lab)] for lab in  np.unique(mask_clustlabels_arr)[1:]])
# [[1, 115781], [2, 94], [3, 7], [4, 7], [5, 7], [6, 7], [7, 21], [8, 22], [9, 48], [10, 7], [11, 7], [12, 22], [13, 49], [14, 16]]

labels = np.unique(mask_clustlabels_arr)[1:]
for lab in labels:
    clust_size = np.sum(mask_clustlabels_arr == lab)
    if clust_size <= CONF['clust_size_thres']:
        mask_arr[mask_clustlabels_arr == lab] = False


mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
labels = np.unique(mask_clustlabels_arr)[1:]

print([[lab, np.sum(mask_clustlabels_arr == lab)] for lab in labels])
# [[1, 115781], [2, 94], [3, 21], [4, 22], [5, 48], [6, 22], [7, 49]]


assert mask_arr.sum() == 116037

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
pop = pd.read_csv(PARTICIPANTS_CSV)
NI = nibabel.load(os.path.join(ANALYSIS_DATA_PATH, "%s.nii.gz") % CONF["NI"])
NI_arr =  NI.get_data()
NI_arr_msk = NI_arr[mask_arr].T

np.save(os.path.join(ANALYSIS_DATA_PATH, "%s_arr_msk.npy" % CONF["NI"]), NI_arr_msk)
NI_arr_msk_ = np.load(os.path.join(ANALYSIS_DATA_PATH, "%s_arr_msk.npy" % CONF["NI"]))
np.all(NI_arr_msk_ == NI_arr_msk)

# Check shape
assert NI_arr.shape == tuple(list(CONF["shape"]) + [pop.shape[0]])

# Save map
map_arr = np.zeros(CONF["shape"])
map_arr[mask_arr] = NI_arr_msk.sum(axis=0) / NI_arr_msk.shape[0]
map_img = nibabel.Nifti1Image(map_arr, mask_img.affine)
map_img.to_filename(os.path.join(ANALYSIS_DATA_PATH, "%s_prop.nii.gz" % CONF["NI"]))

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
assert np.allclose(Atv_.get_singular_values(0), 11.951989954638236)
