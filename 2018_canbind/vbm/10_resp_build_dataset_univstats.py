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


# Voxel size
# vs = "1mm"
vs = "1.5mm-s8mm"
#vs = "1.5mm"

OUTPUT = os.path.join(WD, "models", "vbm_resp_%s" % vs)


#############################################################################
# 1) Build Raw dataset
#
# Read all images
#image_filenames = glob.glob(os.path.join(WD, "data", "derivatives/spmsegment/sub-*/ses-*/anat/mwc1sub-*_ses-*_T1w_%s.nii" % vs))
image_filenames = glob.glob(os.path.join(WD, "data", "derivatives/spmsegment/sub-*/ses-*/anat/*mwc1sub-*_ses-*_T1w_%s.nii" % vs))
regexp = re.compile(".+(sub-.+)/(ses-.+)/anat/.*mwc1sub-.+_ses-.+_T1w_.+.nii")
#f = image_filenames[0]
img_tab = pd.DataFrame([list(regexp.findall(f)[0]) + [f] for f in image_filenames], columns=["participant_id", "session", "path"])
assert img_tab.shape == (806, 3)

# Read participants and volumes
participants = pd.read_csv(os.path.join(WD, "data", "participants.tsv"), sep='\t')
assert participants.shape[0] == 349
vols = pd.read_csv(os.path.join(WD, "data/derivatives/spmsegment/spmsegment_volumes.csv"))
assert vols.shape == (808, 10)

# Merge scans with volume then with participants => img
img = pd.merge(img_tab, vols, on=["participant_id", "session"], how='left')
assert img.shape[0] == 806

# Merge img with participants => pop
pop = pd.merge(participants, img, on=["participant_id"], how='right')
pop["sex_num"] = pop["sex"].map({1:1, 2:0})
pop["respond_wk16_num"] = pop["respond_wk16"].map({"NonResponder":0, "Responder":1})

#
#pop_ = pd.read_csv(os.path.join(OUTPUT, "population.csv"))
pop_orig = pd.read_csv(os.path.join(OUTPUT, "population.csv.bak"))
#pop_.participant_id == pop.participant_id


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
# 3) Compute mask implicit mask
XTot = np.vstack(gm_arrs)
#mask_arr = (np.std(XTot, axis=0) > 1e-7)

mask_arr = (np.mean(XTot, axis=0) >= 0.1) & (np.std(XTot, axis=0) >= 1e-6)
mask_arr = mask_arr.reshape(shape)

#nibabel.Nifti1Image(np.min(XTot, axis=0).reshape(shape).astype(float), affine=img.affine).to_filename(os.path.join(OUTPUT, "min_all.nii.gz"))
nibabel.Nifti1Image(np.mean(XTot, axis=0).reshape(shape).astype(float), affine=img.affine).to_filename(os.path.join(OUTPUT, "mean.nii.gz"))
nibabel.Nifti1Image(np.std(XTot, axis=0).reshape(shape).astype(float), affine=img.affine).to_filename(os.path.join(OUTPUT, "std.nii.gz"))

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
XTot = XTot[:, mask_arr.ravel()]

#############################################################################
# Proportional scaling: remove TIV effect
propscal = pop.TIV_l.mean() / pop.TIV_l
XTotTiv = XTot * np.asarray(propscal)[:, np.newaxis]

#############################################################################
# Remove site effect

XTotTivSite = np.zeros(XTotTiv.shape)
# center data by site
for s in set(pop.site):
    m = pop.site == s
    XTotTivSite[m] = XTotTiv[m] - XTotTiv[m, :].mean(axis=0)

#############################################################################
# Rm few component of PCA
from sklearn import decomposition
pca = decomposition.PCA(n_components=1)
XTotTivSiteCtr = XTotTivSite - np.mean(XTotTivSite, axis=0)
pca.fit(XTotTivSiteCtr)
'''
pca.explained_variance_ratio_
array([ 0.06738766,  0.01289889,  0.01056353,  0.00998285,  0.00903054])
rm only the first component
'''
PC = pca.transform(XTotTivSiteCtr)
XTotTivSitePca = XTotTivSiteCtr - np.dot(PC, pca.components_)

#############################################################################
# Keep only Resp/NoResp
keep = pop.respond_wk16.notnull() & (pop.session == "ses-01")
keep.sum()
pop[keep].respond_wk16.describe()
# count           124
# unique            2
# top       Responder
#freq             92
pop = pop[keep]
pop = pop.reset_index(drop=True)
# Make sure that participants keepts the same order
pop_orig = pd.read_csv(os.path.join(OUTPUT, "population.csv.bak"))
assert np.all(pop_orig.participant_id == pop.participant_id)

XTreatTivSite = XTotTivSite[keep, :]
XTreatTivSitePca = XTotTivSitePca[keep, :]
XTreat = XTot[keep, :]

np.save(os.path.join(OUTPUT, "XTreatTivSitePca.npy"), XTreatTivSitePca)
np.save(os.path.join(OUTPUT, "XTreatTivSite.npy"), XTreatTivSite)
np.save(os.path.join(OUTPUT, "XTreat.npy"), XTreat)
y =  pop['respond_wk16_num']
np.save(os.path.join(OUTPUT, "y.npy"), y)
pop.to_csv(os.path.join(OUTPUT, "population.csv"), index=False)

assert np.sum(pop['respond_wk16_num'] == 0) == 32
assert np.sum(pop['respond_wk16_num'] == 1) == 92

#############################################################################
# Univar stat Full model
Zdf = pd.concat([
        pop[['respond_wk16_num', 'age_onset', 'age', 'sex_num', 'TIV_l']],
        pd.get_dummies(pop[['site']])], axis=1)

print(Zdf.isnull().sum())

Zdf.loc[Zdf["age_onset"].isnull(), "age_onset"] = Zdf["age_onset"].mean()
print(Zdf.isnull().sum())

Z = np.asarray(Zdf)

## OLS with MULM
contrasts = [1] + [0] *(Zdf.shape[1] - 1)

mod = mulm.MUOLS(XTreat, Z)
tvals, pvals, df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)

print([[thres, np.sum(pvals <thres), np.sum(pvals <thres)/pvals.size] for thres in 10. ** np.array([-4, -3, -2])])
# [[0.0001, 34, 8.5521897378753849e-05], [0.001, 333, 0.0008376115243272068], [0.01, 3374, 0.0084867906398798671]]

tstat_arr = np.zeros(mask_arr.shape)
pvals_arr = np.zeros(mask_arr.shape)

pvals_arr[mask_arr] = -np.log10(pvals[0])
tstat_arr[mask_arr] = tvals[0]

pvals_img = nibabel.Nifti1Image(pvals_arr, affine=mask_img.affine)
pvals_img.to_filename(os.path.join(OUTPUT, "XTreat_fulllm_log10pvals.nii.gz"))

tstat_img = nibabel.Nifti1Image(tstat_arr, affine=mask_img.affine)
tstat_img.to_filename(os.path.join(OUTPUT, "XTreat_fulllm_tstat.nii.gz"))

threshold = 3
fig = plt.figure(figsize=(13.33,  7.5 * 4))
ax = fig.add_subplot(411)
ax.set_title("-log pvalues >%.2f"% threshold)
plotting.plot_glass_brain(pvals_img, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(412)
ax.set_title("T-stats T>%.2f" % threshold)
plotting.plot_glass_brain(tstat_img, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(413)
ax.set_title("-log pvalues >%.2f"% threshold)
plotting.plot_stat_map(pvals_img, colorbar=True, draw_cross=False, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(414)
ax.set_title("T-stats T>%.2f" % threshold)
plotting.plot_stat_map(tstat_img, colorbar=True, draw_cross=False, threshold=threshold, figure=fig, axes=ax)
plt.savefig(os.path.join(OUTPUT, "XTreat_fulllm_tstat.png"))


#############################################################################
# Univar stat on data with site effect and TIV removed: use XTreatTivSite
Zdf = pop[['respond_wk16_num', 'age_onset', 'age', 'sex_num']]

print(Zdf.isnull().sum())

Zdf.loc[Zdf["age_onset"].isnull(), "age_onset"] = Zdf["age_onset"].mean()
print(Zdf.isnull().sum())

Z = np.asarray(Zdf)

## OLS with MULM
contrasts = [1] + [0] *(Zdf.shape[1] - 1)

mod = mulm.MUOLS(XTreatTivSite, Z)
tvals, pvals, df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)

print([[thres, np.sum(pvals <thres), np.sum(pvals <thres)/pvals.size] for thres in 10. ** np.array([-4, -3, -2])])
#### [[0.0001, 11, 2.081354469803331e-05], [0.001, 311, 0.00058845567282621444], [0.01, 3606, 0.0068230583801007372]]
#### [[0.0001, 604, 0.001142852817964738], [0.001, 3969, 0.0075099053551358364], [0.01, 17394, 0.032911890588871943]]
# [[0.0001, 85, 0.00021380474344688462], [0.001, 628, 0.0015796397515840416], [0.01, 5355, 0.01346969883715373]]

tstat_arr = np.zeros(mask_arr.shape)
pvals_arr = np.zeros(mask_arr.shape)

pvals_arr[mask_arr] = -np.log10(pvals[0])
tstat_arr[mask_arr] = tvals[0]

pvals_img = nibabel.Nifti1Image(pvals_arr, affine=mask_img.affine)
pvals_img.to_filename(os.path.join(OUTPUT, "XTreatTivSite_lm_log10pvals.nii.gz"))

tstat_img = nibabel.Nifti1Image(tstat_arr, affine=mask_img.affine)
tstat_img.to_filename(os.path.join(OUTPUT, "XTreatTivSite_lm_tstat.nii.gz"))

threshold = 3
fig = plt.figure(figsize=(13.33,  7.5 * 4))
ax = fig.add_subplot(411)
ax.set_title("-log pvalues >%.2f"% threshold)
plotting.plot_glass_brain(pvals_img, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(412)
ax.set_title("T-stats T>%.2f" % threshold)
plotting.plot_glass_brain(tstat_img, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(413)
ax.set_title("-log pvalues >%.2f"% threshold)
plotting.plot_stat_map(pvals_img, colorbar=True, draw_cross=False, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(414)
ax.set_title("T-stats T>%.2f" % threshold)
plotting.plot_stat_map(tstat_img, colorbar=True, draw_cross=False, threshold=threshold, figure=fig, axes=ax)
plt.savefig(os.path.join(OUTPUT, "XTreatTivSite_lm_tstat.png"))

# yeah !!!


#############################################################################
# Univar stat on data with site effect, TIV and Pca removed: use XTreatTivSitePca
Zdf = pop[['respond_wk16_num', 'age_onset', 'age', 'sex_num']]

print(Zdf.isnull().sum())

Zdf.loc[Zdf["age_onset"].isnull(), "age_onset"] = Zdf["age_onset"].mean()
print(Zdf.isnull().sum())

Z = np.asarray(Zdf)

## OLS with MULM
contrasts = [1] + [0] *(Zdf.shape[1] - 1)

mod = mulm.MUOLS(XTreatTivSitePca, Z)
tvals, pvals, df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)

print([[thres, np.sum(pvals <thres), np.sum(pvals <thres)/pvals.size] for thres in 10. ** np.array([-4, -3, -2])])
# [[0.0001, 94, 0.00023644289275302532], [0.001, 726, 0.001826144044028685], [0.01, 5504, 0.013844485975666504]]

tstat_arr = np.zeros(mask_arr.shape)
pvals_arr = np.zeros(mask_arr.shape)

pvals_arr[mask_arr] = -np.log10(pvals[0])
tstat_arr[mask_arr] = tvals[0]

pvals_img = nibabel.Nifti1Image(pvals_arr, affine=mask_img.affine)
pvals_img.to_filename(os.path.join(OUTPUT, "XTreatTivSitePca_lm_log10pvals.nii.gz"))

tstat_img = nibabel.Nifti1Image(tstat_arr, affine=mask_img.affine)
tstat_img.to_filename(os.path.join(OUTPUT, "XTreatTivSitePca_lm_tstat.nii.gz"))

threshold = 3
fig = plt.figure(figsize=(13.33,  7.5 * 4))
ax = fig.add_subplot(411)
ax.set_title("-log pvalues >%.2f"% threshold)
plotting.plot_glass_brain(pvals_img, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(412)
ax.set_title("T-stats T>%.2f" % threshold)
plotting.plot_glass_brain(tstat_img, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(413)
ax.set_title("-log pvalues >%.2f"% threshold)
plotting.plot_stat_map(pvals_img, colorbar=True, draw_cross=False, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(414)
ax.set_title("T-stats T>%.2f" % threshold)
plotting.plot_stat_map(tstat_img, colorbar=True, draw_cross=False, threshold=threshold, figure=fig, axes=ax)
plt.savefig(os.path.join(OUTPUT, "XTreatTivSitePca_lm_tstat.png"))
