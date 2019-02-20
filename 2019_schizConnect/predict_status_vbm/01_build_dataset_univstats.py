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
import brainomics.image_atlas
import nilearn
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

# cookiecutter Directory structure
# https://drivendata.github.io/cookiecutter-data-science/

WD = '/neurospin/psy/schizConnect/studies/2019_predict_status_vbm'
#BASE_PATH = '/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm'


# Voxel size
# vs = "1mm"
#vs = "1.5mm-s8mm"
#vs = "1.5mm"
CONF = dict(voxsize=1.5, smoothing=0, target="dx_num") # Voxel size, smoothing

OUTPUT = os.path.join(WD, "schizConnect_vbm_vs%.1f-s%i" % (CONF["voxsize"], CONF["smoothing"]))
#OUTPUT = /neurospin/psy/schizConnect/studies/2019_predict_status_vbm


#############################################################################
# 1) Build Raw dataset
#
# Read all images
#image_filenames = glob.glob(os.path.join(WD, "data", "derivatives/spmsegment/sub-*/ses-*/anat/mwc1sub-*_ses-*_T1w_%s.nii" % vs))
"""
image_filenames = glob.glob(os.path.join(WD, "data", "derivatives/spmsegment/sub-*/ses-*/anat/*mwc1sub-*_ses-*_T1w_%s.nii" % vs))
regexp = re.compile(".+(sub-.+)/(ses-.+)/anat/.*mwc1sub-.+_ses-.+_T1w_.+.nii")
#f = image_filenames[0]
img_tab = pd.DataFrame([list(regexp.findall(f)[0]) + [f] for f in image_filenames], columns=["participant_id", "session", "path"])
assert img_tab.shape == (806, 3)
"""

# Read participants and volumes
pop = pd.read_csv(os.path.join(WD, "participants.tsv"), sep='\t')
assert pop.shape[0] == 603
"""
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
#pop_orig = pd.read_csv(os.path.join(OUTPUT, "population.csv.bak"))
#pop_.participant_id == pop.participant_id
"""

#############################################################################
# 2) Read images
n = len(pop)
assert n == 603

gm_imgs = list()
gm_arrs = list()

gm_imgs = [nibabel.load(nii_filename) for nii_filename in pop.path]
ref_img = gm_imgs[0]
assert np.all([np.all(img.affine == ref_img.affine) for img in gm_imgs])
shape = ref_img.get_data().shape

#############################################################################
# Smooth image
if CONF["smoothing"] != 0:
    gm_imgs = nilearn.image.smooth_img(gm_imgs, CONF["smoothing"])

#############################################################################
# 3) Compute mask implicit mask
XTot = np.vstack([img.get_data().ravel() for img in gm_imgs])
#mask_arr = (np.std(XTot, axis=0) > 1e-7)

mask_arr = (np.mean(XTot, axis=0) >= 0.1) & (np.std(XTot, axis=0) >= 1e-6)
mask_arr = mask_arr.reshape(shape)


#############################################################################
# atlas: make sure we are within atlas

atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=pop.loc[0, "path"],
    output=os.path.join(OUTPUT, "atlas_harvard_oxford.nii.gz"))

cereb = brainomics.image_atlas.resample_atlas_bangor_cerebellar(
    ref=pop.loc[0, "path"],
    output=os.path.join(OUTPUT, "atlas_cerebellar.nii.gz"))

mask_arr = ((atlas.get_data() + cereb.get_data()) != 0) & mask_arr

#############################################################################
# Remove small branches

mask_arr = scipy.ndimage.binary_opening(mask_arr)

#############################################################################
# Avoid isolated clusters: remove all cluster smaller that 5

mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
print([[lab, np.sum(mask_clustlabels_arr == lab)] for lab in  np.unique(mask_clustlabels_arr)[1:]])
# s0mm
# [[1, 396422], [2, 7], [3, 7]]

clust_size_thres = 10
labels = np.unique(mask_clustlabels_arr)[1:]
for lab in labels:
    clust_size = np.sum(mask_clustlabels_arr == lab)
    if clust_size <= clust_size_thres:
        mask_arr[mask_clustlabels_arr == lab] = False


mask_clustlabels_arr, n_clusts = scipy.ndimage.label(mask_arr)
labels = np.unique(mask_clustlabels_arr)[1:]

print([[lab, np.sum(mask_clustlabels_arr == lab)] for lab in labels])
# [[1, 396422]]

if tuple(CONF.values()) == (1.5, 0):
    assert mask_arr.sum() == 396422


nibabel.Nifti1Image(mask_arr.astype(int), affine=ref_img.affine).to_filename(os.path.join(OUTPUT, "mask.nii.gz"))
#np.save(os.path.join(OUTPUT, "mask.npy"), mask_arr)


# Plot mask
mask_img = nibabel.Nifti1Image(mask_arr.astype(float), affine=ref_img.affine)
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

XTotSite = np.zeros(XTot.shape)
# center data by site
for s in set(pop.site):
    m = pop.site == s
    XTotSite[m] = XTot[m] - XTot[m, :].mean(axis=0)

XTotTivSite = np.zeros(XTotTiv.shape)
XTotTivSite[::] = np.NAN
# center data by site
for s in set(pop.site):
    m = pop.site == s
    XTotTivSite[m] = XTotTiv[m] - XTotTiv[m, :].mean(axis=0)

assert np.sum(np.isnan(XTotTivSite)) == 0

#############################################################################
# Rm few component of PCA
from sklearn import decomposition
pca = decomposition.PCA(n_components=1)
XTotTivSiteCtr = XTotTivSite - np.mean(XTotTivSite, axis=0)
pca.fit(XTotTivSiteCtr)
'''
pca.explained_variance_ratio_
array([ 0.07879986])
rm only the first component
'''
PC = pca.transform(XTotTivSiteCtr)
XTotTivSitePca = XTotTivSiteCtr - np.dot(PC, pca.components_)


assert np.all(pop[CONF["target"]].value_counts() == (330, 273))





#############################################################################
# MULM
def univar_stats(Y, X, prefix, mask_img):
    contrasts = [1] + [0] *(X.shape[1] - 1)
    mod = mulm.MUOLS(Y, X)
    tvals, pvals, df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)

    print([[thres, np.sum(pvals <thres), np.sum(pvals <thres)/pvals.size] for thres in 10. ** np.array([-4, -3, -2])])
    # {'voxsize': 1.5, 'smoothing': 0, 'target': 'dx_num'}
    # [[0.0001, 23068, 0.058190514149063371], [0.001, 47415, 0.11960738808643315], [0.01, 96295, 0.24291033292804132]]

    tstat_arr = np.zeros(mask_arr.shape)
    pvals_arr = np.zeros(mask_arr.shape)

    pvals_arr[mask_arr] = -np.log10(pvals[0])
    tstat_arr[mask_arr] = tvals[0]

    pvals_img = nibabel.Nifti1Image(pvals_arr, affine=mask_img.affine)
    pvals_img.to_filename(os.path.join(OUTPUT, "univ-stat", prefix + "_log10pvals.nii.gz"))

    tstat_img = nibabel.Nifti1Image(tstat_arr, affine=mask_img.affine)
    tstat_img.to_filename(os.path.join(OUTPUT, "univ-stat", prefix + "_tstat.nii.gz"))

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
    plt.savefig(os.path.join(OUTPUT, "univ-stat", prefix + "_tstat.png"))

    return tstat_arr, pvals_arr

def mae(x, y):
    return np.mean(np.abs(x.ravel() -y.ravel()))

def norm_diff(x, y):
    x, y = x.ravel(), y.ravel()
    nrme = np.sqrt(np.sum((x - y) ** 2))
    nrmx = np.sqrt(np.sum((x) ** 2))
    nrmy = np.sqrt(np.sum((y) ** 2))
    return nrme / (1 / 2 * nrmx + 1 / 2 * nrmy)


#############################################################################
# Choose dataset based on MULM
# Summary
# Site as covar or site Centerered almost indentical (With or without TIV)
# Best is Site centered without TIV => XTotSite

# TIV as global scaling or tiv as regressor: moderate diff

#############################################################################
# TIV effect

#############################################################################
# XTot_age+sex+tiv+site # TIV as covariate
Zdf = pd.concat([pop[['dx_num', 'age', 'sex_num', 'TIV_l']],
        pd.get_dummies(pop[['site']])], axis=1)

tmap_siteTivcovar, _= univar_stats(Y=XTot, X=np.asarray(Zdf), prefix="XTot_age+sex+tiv+site", mask_img=mask_img)
# [[0.0001, 23068, 0.058190514149063371], [0.001, 47415, 0.11960738808643315], [0.01, 96295, 0.24291033292804132]]

#############################################################################
# XTot_age+sex+site # No TIV
Zdf = pd.concat([pop[['dx_num', 'age', 'sex_num']],
        pd.get_dummies(pop[['site']])], axis=1)
# [[0.0001, 39880, 0.10059986579957722], [0.001, 75212, 0.18972710898991479], [0.01, 134849, 0.34016527841542599]]
tmap_site, _= univar_stats(Y=XTot, X=np.asarray(Zdf), prefix="XTot_age+sex+site", mask_img=mask_img)
norm_diff(tmap_siteTivcovar, tmap_site)
# 0.1836380452024981

# => TIV no TIV moderate diff

#############################################################################
# XTotTiv_age+sex+site # Tiv as global scaling
Zdf = pd.concat([pop[['dx_num', 'age', 'sex_num']],
        pd.get_dummies(pop[['site']])], axis=1)

tmap_siteTivglobscale, _= univar_stats(Y=XTotTiv, X=np.asarray(Zdf), prefix="XTotTiv_age+sex+site", mask_img=mask_img)
# [[0.0001, 18807, 0.047441867504830711], [0.001, 39451, 0.099517685698573749], [0.01, 83613, 0.2109191719934817]]

norm_diff(tmap_siteTivcovar, tmap_siteTivglobscale)
# 0.12768083340601372
# => TIV as global scaling or tiv as regressor moderate diff

norm_diff(tmap_site, tmap_siteTivglobscale)
# 0.25947152245874594
# => TIV as global scaling or no tiv as regressor moderate++ diff


#############################################################################
# SITE effect

#############################################################################
# XTot_age+sex # No site
Zdf = pop[['dx_num', 'age', 'sex_num']]
Zdf["inter"] = 1
#Zdf = pop[['sex_num', 'dx_num', 'age']]
#Zdf = pop[['age', 'sex_num', 'dx_num']]

tmap_, _ = univar_stats(Y=XTot, X=np.asarray(Zdf), prefix="XTot_age+sex", mask_img=mask_img)
# [[0.0001, 37600, 0.094848419108929369], [0.001, 73000, 0.18414719667425117], [0.01, 133688, 0.33723658121900402]]

norm_diff(tmap_site, tmap_)
# 0.055600430999142574
# Site as covar no Site very similar

#############################################################################
# XTotSiteCtr_age+sex # Site centered
Zdf = pop[['dx_num', 'age', 'sex_num']]
Zdf["inter"] = 1

tmap_siteCtr, _ = univar_stats(Y=XTotSite, X=np.asarray(Zdf), prefix="XTotSiteCtr_age+sex", mask_img=mask_img)
# [[0.0001, 41647, 0.1050572369848293], [0.001, 77501, 0.19550125875960467], [0.01, 137327, 0.34641619284499853]]
norm_diff(tmap_site, tmap_siteCtr)
# 0.015297198733383671
# Without TIV: Site as covar or site Centerered almost indentical

#############################################################################
# XTotTivSite_age+sex+site # Tiv as global scaling + Site centered
Zdf = pop[['dx_num', 'age', 'sex_num']]
Zdf["inter"] = 1

tmap_XTotTivSite_age_sex, _= univar_stats(Y=XTotTivSite, X=np.asarray(Zdf), prefix="XTotTivSite_age+sex", mask_img=mask_img)
# [[0.0001, 20854, 0.052605556704723756], [0.001, 43047, 0.10858882705803412], [0.01, 89149, 0.22488408816866876]]

norm_diff(tmap_siteTivglobscale, tmap_XTotTivSite_age_sex)
# 0.032824845663625236
# With TIV as global scaling: Site as covar or site Centerered almost indentical




#############################################################################
# Save
np.savez_compressed(os.path.join(OUTPUT, "XTotSite.npz"),
                    participant_id = pop["participant_id"],
                    site = pop["site"],
                    #session = pop["session"],
                    age = pop["age"],
                    target = pop[CONF["target"]],
                    XTotSite = XTotSite)


pop.to_csv(os.path.join(OUTPUT, "participants.csv"), index=False)

# QC
pop_= pd.read_csv(os.path.join(OUTPUT, "participants.csv"))
arxiv = np.load(os.path.join(OUTPUT, "XTotSite.npz"))

assert np.all(arxiv['participant_id'] == pop_['participant_id'])

if tuple(CONF.values()) == (1.5, 0):
    arxiv['XTotSite'] == (603, 396422)

###############################################################################
# precompute linearoperator
arxiv = np.load(os.path.join(OUTPUT, "XTotSite.npz"))
X = arxiv['XTotSite']
y = arxiv['target']

mask = nibabel.load(os.path.join(OUTPUT, "mask.nii.gz"))

Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 11.956572163487806)
