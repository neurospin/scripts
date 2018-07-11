#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 18:20:14 2018

@author: ed203246
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

#############################################################################
# Build dataset
INPUT = os.path.join(WD, "models", "vbm_%s" % vs)
OUTPUT = os.path.join(WD, "models", "univstats-RespNoResp_vbm_%s" % vs)

Xraw = np.load(os.path.join(INPUT, "Xraw.npy"))
pop = pd.read_csv(os.path.join(INPUT, "population.csv"))
mask_img =  nibabel.load(os.path.join(INPUT, "mask.nii.gz"))
mask_arr = mask_img.get_data() == True
mean_sites = np.load(os.path.join(INPUT, "mean_sites.npz"))

assert np.all(mask_arr == np.load(os.path.join(INPUT, "mask.npy")))
# 1.5mm

if vs == "1.5mm":
    assert mask_arr.sum() == 397559
# 1.5mm-s8mm
if vs == "1.5mm-s8mm":
    assert mask_arr.sum() == 528502

#############################################################################
# Select ses-01 with Treatment info
mask_subj = pop["group"].notnull() & pop["respond_wk16"].notnull() & (pop.session == "ses-01")
X = Xraw[mask_subj]
mask_subj.sum()
pop = pop[mask_subj]
assert pop.shape[0] == 124
pop["sex_num"] = pop["sex"].map({1:1, 2:0})
pop["respond_wk16_num"] = pop["respond_wk16"].map({"NonResponder":0, "Responder":1})

# som QC
assert pop.shape[0] == pop['ses-01'].sum()
pop = pop[['participant_id', 'age', 'sex_num', 'site', 'respond_wk16_num', 'psyhis_mdd_age', 'path']]

assert np.sum(pop['respond_wk16_num'] == 0) == 32
assert np.sum(pop['respond_wk16_num'] == 1) == 92

# center data by site
for s in set(pop.site):
    print(s, mean_sites[s].mean())
    X[s == pop.site] = X[s == pop.site] - mean_sites[s]

y =  pop['respond_wk16_num']

np.save(os.path.join(OUTPUT, "Xsite.npy"), X)
np.save(os.path.join(OUTPUT, "y.npy"), y)
pop.to_csv(os.path.join(OUTPUT, "population.csv"), index=False)

#############################################################################
# Models of respond_wk16_num + psyhis_mdd_age + age + sex_num + site

"""
# load data
X = np.load(os.path.join(OUTPUT, "Xsite.npy"))
pop = pd.read_csv(os.path.join(OUTPUT, "population.csv"))
assert X.shape == (124, 397559)

mask_img = nibabel.load(os.path.join(OUTPUT, "mask.nii.gz"))
mask_arr = mask_img.get_data()
mask_arr = mask_arr == 1
"""

#############################################################################
# Model 1: full-lm: MRI ~ respond_wk16_num + psyhis_mdd_age + age + sex_num

# load data

pop = pop[['participant_id', 'age', 'sex_num', 'site', 'respond_wk16_num', 'psyhis_mdd_age', 'path']]


Zdf = pd.concat([
        pop[['respond_wk16_num', 'psyhis_mdd_age', 'age', 'sex_num']],
        pd.get_dummies(pop[['site']])], axis=1)

print(Zdf.isnull().sum())

Zdf.loc[Zdf["psyhis_mdd_age"].isnull(), "psyhis_mdd_age"] = Zdf["psyhis_mdd_age"].mean()
print(Zdf.isnull().sum())

Z = np.asarray(Zdf)

## OLS with MULM
contrasts = [1] + [0] *(Zdf.shape[1] - 1)

mod = mulm.MUOLS(X, Z)
tvals, pvals, df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)

print([[thres, np.sum(pvals <thres), np.sum(pvals <thres)/pvals.size] for thres in 10. ** np.array([-4, -3, -2])])
# 1mm
# [[0.0001, 189, 0.00014439627839602474], [0.001, 1448, 0.0011062741328965282], [0.01, 13813, 0.010553152346477725]]
# 1.5mm
# [[0.0001, 48, 0.0001207367962994172], [0.001, 414, 0.0010413548680824733], [0.01, 3947, 0.0099280861457041597]]
# 1.5mm-s8mm
# [[0.0001, 30, 5.6764212812818116e-05], [0.001, 432, 0.0008174046645045809], [0.01, 5028, 0.0095136820674283154]]

tstat_arr = np.zeros(mask_arr.shape)
pvals_arr = np.zeros(mask_arr.shape)

pvals_arr[mask_arr] = -np.log10(pvals[0])
tstat_arr[mask_arr] = tvals[0]

pvals_img = nibabel.Nifti1Image(pvals_arr, affine=mask_img.affine)
pvals_img.to_filename(os.path.join(OUTPUT, "resp-fulllm_log10pvals.nii.gz"))

tstat_img = nibabel.Nifti1Image(tstat_arr, affine=mask_img.affine)
tstat_img.to_filename(os.path.join(OUTPUT, "resp-fulllm_tstat.nii.gz"))

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
plt.savefig(os.path.join(OUTPUT, "resp-fulllm_tstat.png"))


nperms = 1000
tvals, pvalsTmax, _ = mod.t_test_maxT(contrasts=contrasts, nperms=nperms, two_tailed=True)
print([[thres, np.sum(pvalsTmax <thres), np.sum(pvalsTmax <thres)/pvals.size] for thres in 10. ** np.array([-4, -3, -2, -1])])
# 1.5mm
# [[0.0001, 0, 0.0], [0.001, 0, 0.0], [0.01, 0, 0.0], [0.10000000000000001, 1, 2.5153499229045248e-06]]
# 1mm
# [[0.0001, 0, 0.0], [0.001, 0, 0.0], [0.01, 0, 0.0], [0.1, 0, 0.0]]
# => got for 1.5mm





















#############################################################################
# Model 2: resid-lm: MRI residual = MRI ~ psyhis_mdd_age + age + sex_num
#                    MRI residual ~ respond_wk16_num

# load data
X = np.load(os.path.join(OUTPUT, "Xraw.npy"))
pop = pd.read_csv(os.path.join(OUTPUT, "population.csv"))

pop = pop[['participant_id', 'age', 'sex_num', 'site', 'respond_wk16_num', 'psyhis_mdd_age', 'path']]


Zdf = pd.concat([
        pop[['psyhis_mdd_age', 'age', 'sex_num']],
        pd.get_dummies(pop[['site']])], axis=1)

print(Zdf.isnull().sum())

Zdf.loc[Zdf["psyhis_mdd_age"].isnull(), "psyhis_mdd_age"] = Zdf["psyhis_mdd_age"].mean()
print(Zdf.isnull().sum())

Z = np.asarray(Zdf)

## OLS with MULM
contrasts = [1] + [0] *(Zdf.shape[1] - 1)

mod = mulm.MUOLS(X, Z)
mod.fit()
residuals = X - mod.predict(Z)


# residuals is the new X
Xres = residuals
np.save(os.path.join(OUTPUT, "Xres.npy"), residuals)


Zdf = pop[['respond_wk16_num']]
Zdf["intercept"] = 1

print(Zdf.isnull().sum())

Z = np.asarray(Zdf)

## OLS with MULM
contrasts = [1] + [0] *(Zdf.shape[1] - 1)

mod = mulm.MUOLS(Xres, Z)
tvals, pvals, df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)

print([[thres, np.sum(pvals <thres), np.sum(pvals <thres)/pvals.size] for thres in 10. ** np.array([-4, -3, -2])])


# 1.5mm
# [[0.0001, 36, 9.070523318803699e-05], [0.001, 313, 0.000788631610773766], [0.01, 3305, 0.008327244324623952]]
# to be compared with full-lm
# 1.5mm
# [[0.0001, 48, 0.00012094031091738265], [0.001, 414, 0.0010431101816624254], [0.01, 3951, 0.00995489934238706]]

tstat_arr = np.zeros(mask_arr.shape)
pvals_arr = np.zeros(mask_arr.shape)

pvals_arr[mask_arr] = -np.log10(pvals[0])
tstat_arr[mask_arr] = tvals[0]

pvals_img = nibabel.Nifti1Image(pvals_arr, affine=mask_img.affine)
pvals_img.to_filename(os.path.join(OUTPUT, "resid-lm_log10pvals.nii.gz"))

tstat_img = nibabel.Nifti1Image(tstat_arr, affine=mask_img.affine)
tstat_img.to_filename(os.path.join(OUTPUT, "resid-lm_tstat.nii.gz"))

threshold = 3
fig = plt.figure(figsize=(13.33,  7.5 * 4))
ax = fig.add_subplot(411)
ax.set_title("p-values")
plotting.plot_glass_brain(pvals_img, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(412)
ax.set_title("T-stats")
plotting.plot_glass_brain(tstat_img, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(413)
ax.set_title("p-values")
plotting.plot_stat_map(pvals_img, colorbar=True, draw_cross=False, threshold=threshold, figure=fig, axes=ax)

ax = fig.add_subplot(414)
ax.set_title("T-stats")
plotting.plot_stat_map(tstat_img, colorbar=True, draw_cross=False, threshold=threshold, figure=fig, axes=ax)
plt.savefig(os.path.join(OUTPUT, "resid-lm_stat_response.png"))


fullm_arr = nibabel.load(os.path.join(OUTPUT, "full-lm_tstat.nii.gz")).get_data()[mask_arr]
resiflm_arr = nibabel.load(os.path.join(OUTPUT, "resid-lm_tstat.nii.gz")).get_data()[mask_arr]
from scipy import stats
beta, beta0, r_value, p_value, std_err = stats.linregress(fullm_arr, resiflm_arr)
yhat = beta * fullm_arr + beta0 # regression line
plt.plot(fullm_arr, resiflm_arr,'o', fullm_arr, yhat, 'r-')
plt.xlabel('full-lm_tstat')
plt.ylabel('resid-lm_tstat = %.2f * full-lm_tstat + %.2f' % (beta, beta0))
plt.savefig(os.path.join(OUTPUT, "resid-lm_vs_full-lm_stat_response.png"))
print(beta, beta0, r_value, p_value, std_err)
# 0.9774301944741465 3.900719991595247e-05 0.9999992146142297 0.0 1.944502629602345e-06

#nperms = 1000
#tvals, pvalsTmax, _ = mod.t_test_maxT(contrasts=contrasts, nperms=nperms, two_tailed=True)
#print([[thres, np.sum(pvalsTmax <thres), np.sum(pvals <thres)/pvals.size] for thres in 10. ** np.array([-4, -3, -2, -1])])

assert np.all(pop['respond_wk16_num'] == y)


#############################################################################
# Model 3: center-site: MRI = MRI - site

# load data
Xraw = np.load(os.path.join(OUTPUT, "Xraw.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
pop = pd.read_csv(os.path.join(OUTPUT, "population.csv"))
assert np.all(pop['respond_wk16_num'] == y)

site = pop.site
X = np.zeros(Xraw.shape)
X[:] = np.nan
np.isnan(X).sum()
set(site)
# {'CAM', 'MCU', 'QNS', 'TGH', 'UBC', 'UCA'}


X[site=='CAM',:] = Xraw[site=='CAM',:] - Xraw[site=='CAM',:].mean(axis=0)
X[site=='MCU',:] = Xraw[site=='MCU',:] - Xraw[site=='MCU',:].mean(axis=0)
X[site=='QNS',:] = Xraw[site=='QNS',:] - Xraw[site=='QNS',:].mean(axis=0)
X[site=='TGH',:] = Xraw[site=='TGH',:] - Xraw[site=='TGH',:].mean(axis=0)
X[site=='UBC',:] = Xraw[site=='UBC',:] - Xraw[site=='UBC',:].mean(axis=0)
X[site=='UCA',:] = Xraw[site=='UCA',:] - Xraw[site=='UCA',:].mean(axis=0)
np.isnan(X).sum()

np.save(os.path.join(OUTPUT, "Xcentersite.npy"), X)

###############################################################################
###############################################################################
# precompute linearoperatorSS
#X = np.load("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/data/X.npy")
#y = np.load("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/data/y.npy")

mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))

import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)