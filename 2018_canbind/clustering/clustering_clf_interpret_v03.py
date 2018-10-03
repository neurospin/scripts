#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:29:19 2018

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

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
import sklearn.metrics as metrics

# cookiecutter Directory structure
# https://drivendata.github.io/cookiecutter-data-science/

# OUTPUT = os.path.join(WD, "models", "vbm_resp_%s" % vs)
WD = os.path.join('/neurospin/psy/canbind/models/clustering_v03')
CLUST = 1

###############################################################################
# Population
###############################################################################
img_tot = np.load(os.path.join(WD, "XTotTivSiteCtr.npz"))
pop_tot = pd.read_csv(os.path.join(WD, "population.csv"))

# QC
assert np.all(img_tot['participant_id'] == pop_tot['participant_id'])
assert np.all(img_tot["treat_ses01_mask"] == pop_tot.respond_wk16.notnull() & (pop_tot.session == "ses-01"))

mask_ctl = pop_tot.treatment == "Control"
mask_ctl_ses01 = (pop_tot.treatment == "Control") & (pop_tot.session == "ses-01")
mask_treat_ses01 = pop_tot.respond_wk16.notnull() & (pop_tot.session == "ses-01")


###############################################################################
# Clustering
###############################################################################

cluster_tot = pd.read_csv(os.path.join(WD, "clusters.csv"))
assert np.all(cluster_tot['participant_id'] == pop_tot['participant_id'])

centers = np.load(os.path.join(WD, "clusters_centers.npz"))
'''
cluster_tot["proj_XCtlSes1TivSiteCtr"]
cluster_tot["cluster_XCtlSes1TivSiteCtr"]
cluster_tot["proj_XCtlSes1TivSiteCtr_all"]
cluster_tot["cluster_XCtlSes1TivSiteCtr_all"]
'''

mask_gp1 = (cluster_tot["cluster_XCtlSes1TivSiteCtr"] == 1) & mask_treat_ses01
mask_gp0 = (cluster_tot["cluster_XCtlSes1TivSiteCtr"] == 0) & mask_treat_ses01
mask_gp1_withintreat = (cluster_tot.loc[mask_treat_ses01, "cluster_XCtlSes1TivSiteCtr"] == 1)
mask_gp0_withintreat = (cluster_tot.loc[mask_treat_ses01, "cluster_XCtlSes1TivSiteCtr"] == 0)

assert mask_gp1.sum() == 66
assert mask_gp1.sum() + mask_gp0.sum() == mask_treat_ses01.sum()

assert np.all(
    metrics.confusion_matrix(pop_tot['respond_wk16_num'][mask_treat_ses01],
                             cluster_tot["cluster_XCtlSes1TivSiteCtr"][mask_treat_ses01]) == \
    np.array([[16, 16],
              [42, 50]]))

###############################################################################
# Imaging data
###############################################################################


#######################
# 1) Keep only subgroup

if CLUST == 1:
    Xim_g1 = img_tot['XTotTivSiteCtr'][mask_gp1]
    assert Xim_g1.shape == (66, 397559)
    y_g1 = np.array(pop_tot['respond_wk16_num'][mask_gp1], dtype=int)
elif CLUST == 0:
    Xim_g0 = img_tot['XTotTivSiteCtr'][mask_gp0]
    assert Xim_g0.shape == (58, 397559)
    y_g0 = np.array(pop_tot['respond_wk16_num'][mask_gp0], dtype=int)

###############################################################################
# Clinical data: imput missing
###############################################################################

###############
# 1) Imputation

# use data from all patients
democlin = pop_tot.loc[mask_treat_ses01, ['participant_id', 'age', 'sex_num', 'educ', 'age_onset',
                'respond_wk16',
                'mde_num', 'madrs_Baseline', 'madrs_Screening']]
democlin.describe()

"""
              age     sex_num        educ   age_onset    mde_num  madrs_Baseline  madrs_Screening
count  124.000000  124.000000  123.000000  118.000000  88.000000      120.000000       117.000000
mean    35.693548    0.620968   16.813008   20.983051   3.840909       29.975000        30.427350
std     12.560214    0.487114    2.255593    9.964881   2.495450        5.630742         5.234692
min     18.000000    0.000000    9.000000    5.000000   1.000000       21.000000        22.000000
25%     25.000000    0.000000   16.000000   14.250000   2.000000       25.750000        27.000000
50%     33.000000    1.000000   17.000000   18.000000   3.000000       29.000000        29.000000
75%     46.000000    1.000000   19.000000   25.750000   5.000000       34.000000        33.000000
max     61.000000    1.000000   21.000000   55.000000  10.000000       47.000000        46.000000
"""
democlin.isnull().sum()
"""
age                 0
sex_num             0
educ                1
age_onset           6
respond_wk16        0
mde_num            36
madrs_Baseline      4
madrs_Screening     7
"""

# Imput missing value with the median
democlin.loc[democlin["educ"].isnull(), "educ"] = democlin["educ"].median()
democlin.loc[democlin["age_onset"].isnull(), "age_onset"] = democlin["age_onset"].median()
democlin.loc[democlin["mde_num"].isnull(), "mde_num"] = democlin["mde_num"].median()


democlin.loc[democlin["madrs_Baseline"].isnull(), "madrs_Baseline"] = democlin.loc[democlin["madrs_Baseline"].isnull(), "madrs_Screening"]
assert democlin["madrs_Baseline"].isnull().sum() == 0

democlin.pop("madrs_Screening")
assert(np.all(democlin.isnull().sum() == 0))

# add duration
democlin["duration"] = democlin["age"] - democlin["age_onset"]


#######################
# 2) Keep only subgroup

# Grp1
democlin_g1 = democlin[mask_gp1_withintreat]
# Drop participant_id & response with some QC
assert np.all(democlin_g1.pop('participant_id').values == pop_tot['participant_id'][mask_gp1].values)
assert np.all(democlin_g1.pop("respond_wk16").values == pop_tot['respond_wk16'][mask_gp1].values)

democlin_g1.columns
"""
['age', 'sex_num', 'educ', 'age_onset', 'mde_num', 'madrs_Baseline', 'duration']
"""
Xclin_g1 = np.asarray(democlin_g1)

###############################################################################
# ML
###############################################################################

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
import copy
import sklearn.linear_model as lm
from sklearn.pipeline import make_pipeline

#clustering = pd.read_csv(os.path.join(WD, DATASET+"-clust.csv"))
#cluster_labels = clustering.cluster
C = 0.1
NFOLDS = 5
#cv = StratifiedKFold(n_splits=NFOLDS)
# cv = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42+1)
cv = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=24)

#model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)
scaler = preprocessing.StandardScaler()
def balanced_acc(estimator, X, y):
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()
scorers = {'auc': 'roc_auc', 'bacc':balanced_acc, 'acc':'accuracy'}


###############################################################################
# Clustering Im classifiy ClinImEnettv

import nibabel
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov


if CLUST == 1:
    assert Xim_g1.shape == (66, 397559)
    y_g1 = np.array(pop_tot['respond_wk16_num'][mask_gp1], dtype=int)
    Xim, y = Xim_g1, y_g1
    Xclin = Xclin_g1
    assert np.all(np.array([np.sum(lab==y) for lab in np.unique(y)]) == (16, 50))
elif CLUST == 0:
    assert Xim_g0.shape == (58, 397559)
    y_g0 = np.array(pop_tot['respond_wk16_num'][mask_gp0], dtype=int)
    X, y = Xim_g0, y_g0


mask = nibabel.load(os.path.join(WD, "mask.nii.gz"))
# Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
# Atv.save(os.path.join(WD, "Atv.npz"))
Atv = LinearOperatorNesterov(filename=os.path.join(WD, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv.get_singular_values(0), 11.956104408414376)


# Load models Coeficients
#modelscv = np.load(os.path.join(WD, "XCtlSes1TivSiteCtr-clust-Ctl-%i"%CLUST +"_enettv_0.1_0.1_0.8_%icv.npz" % NFOLDS))
modelscv = np.load(os.path.join(WD, "rcv_gp1/rcv_024.npz")) # ['beta_cv', 'beta_refit', 'y_pred', 'y_true', 'proba_pred']

# Parameters
key = 'enettv_0.1_0.1_0.8'.split("_")
algo, alpha, l1l2ratio, tvratio = key[0], float(key[1]), float(key[2]), float(key[3])
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)
# print(key, algo, alpha, l1, l2, tv)

# CV loop
y_test_pred_img = np.zeros(len(y))
y_test_prob_pred_img = np.zeros(len(y))
y_test_decfunc_pred_img = np.zeros(len(y))
y_train_pred_img = np.zeros(len(y))
coefs_cv_img = np.zeros((NFOLDS, Xim.shape[1]))
auc_test_img = list()
recalls_test_img = list()
acc_test_img = list()

y_test_pred_clin = np.zeros(len(y))
y_test_prob_pred_clin = np.zeros(len(y))
y_test_decfunc_pred_clin = np.zeros(len(y))
y_train_pred_clin = np.zeros(len(y))
coefs_cv_clin = np.zeros((NFOLDS, Xclin.shape[1]))
auc_test_clin = list()
recalls_test_clin = list()
acc_test_clin = list()

y_test_pred_stck = np.zeros(len(y))
y_test_prob_pred_stck = np.zeros(len(y))
y_test_decfunc_pred_stck = np.zeros(len(y))
y_train_pred_stck = np.zeros(len(y))
coefs_cv_stck = np.zeros((NFOLDS, 2))
auc_test_stck = list()
recalls_test_stck = list()
acc_test_stck = list()

CV = list()
for cv_i, (train, test) in enumerate(cv.split(Xim, y)):
    CV.append([train.tolist(), test.tolist()])
    #for train, test in cv.split(X, y, None):
    print(cv_i)
    X_train_img, X_test_img, y_train, y_test = Xim[train, :], Xim[test, :], y[train], y[test]
    X_train_clin, X_test_clin = Xclin[train, :], Xclin[test, :]

    X_train_img = scaler.fit_transform(X_train_img)
    X_test_img = scaler.transform(X_test_img)
    X_train_clin = scaler.fit_transform(X_train_clin)
    X_test_clin = scaler.transform(X_test_clin)

    # Im
    conesta = algorithms.proximal.CONESTA(max_iter=10000)
    estimator_img = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                    class_weight="auto", penalty_start=0)
    estimator_img.beta = modelscv["beta_cv"][cv_i][:, None]
    # Store prediction for micro avg
    y_test_pred_img[test] = estimator_img.predict(X_test_img).ravel()
    y_test_prob_pred_img[test] = estimator_img.predict_probability(X_test_img).ravel()#[:, 1]
    y_test_decfunc_pred_img[test] = np.dot(X_test_img, estimator_img.beta).ravel()
    y_train_pred_img[train] = estimator_img.predict(X_train_img).ravel()
    # Compute score for macro avg
    auc_test_img.append(metrics.roc_auc_score(y_test, estimator_img.predict_probability(X_test_img).ravel()))
    recalls_test_img.append(metrics.recall_score(y_test, estimator_img.predict(X_test_img).ravel(), average=None))
    acc_test_img.append(metrics.accuracy_score(y_test, estimator_img.predict(X_test_img).ravel()))
    coefs_cv_img[cv_i, :] = estimator_img.beta.ravel()

    # Clin
    estimator_clin = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)
    estimator_clin.fit(X_train_clin, y_train)
    y_test_pred_clin[test] = estimator_clin.predict(X_test_clin).ravel()
    y_test_prob_pred_clin[test] =  estimator_clin.predict_proba(X_test_clin)[:, 1]
    y_test_decfunc_pred_clin[test] = estimator_clin.decision_function(X_test_clin)
    y_train_pred_clin[train] = estimator_clin.predict(X_train_clin).ravel()
    # Compute score for macro avg
    auc_test_clin.append(metrics.roc_auc_score(y_test, estimator_clin.predict_proba(X_test_clin)[:, 1]))
    recalls_test_clin.append(metrics.recall_score(y_test, estimator_clin.predict(X_test_clin).ravel(), average=None))
    acc_test_clin.append(metrics.accuracy_score(y_test, estimator_clin.predict(X_test_clin).ravel()))
    coefs_cv_clin[cv_i, :] = estimator_clin.coef_.ravel()

    # Stacking
    X_train_stck = np.c_[
            np.dot(X_train_img, estimator_img.beta).ravel(),
            estimator_clin.decision_function(X_train_clin).ravel()]
    X_test_stck = np.c_[
            np.dot(X_test_img, estimator_img.beta).ravel(),
            estimator_clin.decision_function(X_test_clin).ravel()]
    X_train_stck = scaler.fit(X_train_stck).transform(X_train_stck)
    X_test_stck = scaler.transform(X_test_stck)

    #
    estimator_stck = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=100)
    estimator_stck.fit(X_train_stck, y_train)
    y_test_pred_stck[test] = estimator_stck.predict(X_test_stck).ravel()
    y_test_prob_pred_stck[test] =  estimator_stck.predict_proba(X_test_stck)[:, 1]
    y_test_decfunc_pred_stck[test] = estimator_stck.decision_function(X_test_stck)
    y_train_pred_stck[train] = estimator_stck.predict(X_train_stck).ravel()
    # Compute score for macro avg
    auc_test_stck.append(metrics.roc_auc_score(y_test, estimator_stck.predict_proba(X_test_stck)[:, 1]))
    recalls_test_stck.append(metrics.recall_score(y_test, estimator_stck.predict(X_test_stck).ravel(), average=None))
    acc_test_stck.append(metrics.accuracy_score(y_test, estimator_stck.predict(X_test_stck).ravel()))
    coefs_cv_stck[cv_i, :] = estimator_stck.coef_.ravel()


#print("#", IMADATASET+"-clust%i"%CLUST, Ximg.shape)

# Micro Avg Img
recall_test_img_microavg = metrics.recall_score(y, y_test_pred_img, average=None)
recall_train_img_microavg = metrics.recall_score(y, y_train_pred_img, average=None)
bacc_test_img_microavg = recall_test_img_microavg.mean()
auc_test_img_microavg = metrics.roc_auc_score(y, y_test_prob_pred_img)
acc_test_img_microavg = metrics.accuracy_score(y, y_test_pred_img)

print("#", auc_test_img_microavg, bacc_test_img_microavg, acc_test_img_microavg)
print("#", auc_test_img)
print("#", recalls_test_img)
print("#", acc_test_img)
# seed 42 + 1
# 0.70625 0.66375 0.651515151515
# [0.57499999999999996, 0.96666666666666679, 0.73333333333333339, 0.93333333333333335, 0.53333333333333344]
# [array([ 0.5,  0.6]), array([ 1. ,  0.6]), array([ 0.66666667,  0.6       ]), array([ 1. ,  0.5]), array([ 0.33333333,  0.9       ])]
# [0.5714285714285714, 0.69230769230769229, 0.61538461538461542, 0.61538461538461542, 0.76923076923076927]

# Seed 24 : switch 42
# 0.70375 0.66375 0.651515151515
# [0.625, 0.53333333333333344, 0.76666666666666672, 0.8666666666666667, 0.66666666666666674]
# [array([ 0.75,  0.6 ]), array([ 0.33333333,  0.4       ]), array([ 0.66666667,  0.7       ]), array([ 1. ,  0.7]), array([ 0.66666667,  0.8       ])]
# [0.6428571428571429, 0.38461538461538464, 0.69230769230769229, 0.76923076923076927, 0.76923076923076927]

# Micro Avg Clin
recall_test_clin_microavg = metrics.recall_score(y, y_test_pred_clin, average=None)
recall_train_clin_microavg = metrics.recall_score(y, y_train_pred_clin, average=None)
bacc_test_clin_microavg = recall_test_clin_microavg.mean()
auc_test_clin_microavg = metrics.roc_auc_score(y, y_test_prob_pred_clin)
acc_test_clin_microavg = metrics.accuracy_score(y, y_test_pred_clin)

print("#", auc_test_clin_microavg, bacc_test_clin_microavg, acc_test_clin_microavg)
# 0.61375 0.60125 0.621212121212
# 0.6375 0.61125 0.636363636364

# Micro Avg Stacking
recall_test_stck_microavg = metrics.recall_score(y, y_test_pred_stck, average=None)
recall_train_stck_microavg = metrics.recall_score(y, y_train_pred_stck, average=None)
bacc_test_stck_microavg = recall_test_stck_microavg.mean()
auc_test_stck_microavg = metrics.roc_auc_score(y, y_test_prob_pred_stck)
acc_test_stck_microavg = metrics.accuracy_score(y, y_test_pred_stck)

print("#", auc_test_stck_microavg, bacc_test_stck_microavg, acc_test_stck_microavg)
# 0.73 0.65375 0.636363636364
# 0.72375 0.66375 0.651515151515

"""
import json
with open(os.path.join(WD, 'XCtlSes1TivSiteCtr-clust-Ctl-1_5cv.json'), "w") as outfile:
    json.dump(CV, outfile)
"""

###############################################################################
# Caracterize Cluster centers

from nilearn import plotting, image
import  nibabel
from matplotlib.backends.backend_pdf import PdfPages


scaler = preprocessing.StandardScaler()
Xim_ = scaler.fit_transform(img_tot['XTotTivSiteCtr'][mask_treat_ses01])
cluster_ = cluster_tot[mask_treat_ses01]
cluster_centers = centers['XCtlSes1TivSiteCtr']
mean = scaler.mean_

c0 = cluster_centers[0, :]
c1 = cluster_centers[1, :]
# Some QC
proj_c1c0  = np.dot(Xim_, c1 - c0)
assert np.allclose(cluster_tot['proj_XCtlSes1TivSiteCtr'][mask_treat_ses01], proj_c1c0)


"""
df = imgscores.copy()
df["respond_wk16"] = pop["respond_wk16"]
df["sex"] = pop["sex"]
df["age"] = pop["age"]

plt.plot(imgscores.GMratio, imgscores.proj_c1c0, 'o')

ICI
sns.lmplot(x="GMratio", y="proj_c1c0", hue="respond_wk16" , data=df, fit_reg=False)
sns.lmplot(x="GMratio", y="proj_c1c0", hue="cluster" , data=df, fit_reg=False)
sns.lmplot(x="GMratio", y="proj_c1c0", hue="age" , data=df, fit_reg=False)
sns.lmplot(x="GMratio", y="age", hue="cluster" , data=df, fit_reg=False)
sns.lmplot(x="age", y="GMratio", hue="cluster" , data=df, fit_reg=False)
sns.lmplot(x="age", y="proj_c1c0", hue="cluster" , data=df, fit_reg=False)
"""

n0 = np.sum(cluster_["cluster_XCtlSes1TivSiteCtr"] == 0)
n1 = np.sum(cluster_["cluster_XCtlSes1TivSiteCtr"] == 1)
assert n0, n1 == (58, 66)

X0 = Xim[cluster_["cluster_XCtlSes1TivSiteCtr"] == 0, ]
X1 = Xim[cluster_["cluster_XCtlSes1TivSiteCtr"] == 1, ]
X0c = X0 - c0
X1c = X1 - c1

s = np.sqrt((np.sum(X0c ** 2, axis=0) * (n0 - 1) + np.sum(X1c ** 2, axis=0) * (n1 - 1)) / (n0 + n1 -2))

tmap = (c1 - c0) / (s * np.sqrt(1 / n1 + 1 / n0))
zmap = (c1 - c0) / s


mask_img = nibabel.load(os.path.join(WD, "mask.nii.gz"))
coef_arr = np.zeros(mask_img.get_data().shape)


figure_filename = os.path.join(WD, "clusters_centers.pdf")
pdf = PdfPages(figure_filename)

fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = c1
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='Group 1 center (centered and scaled)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = c0
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='Group 0 center (centered and scaled)', colorbar=True)
pdf.savefig(); plt.close()


c0_ = scaler.inverse_transform(c0)
coef_arr[mask_img.get_data() != 0] = c0_ - c0_.min()
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_anat(coef_img, display_mode='ortho', cut_coords=(5, -13, 1), black_bg=True,  draw_cross=False,
                       title='Group 0 center')
pdf.savefig(); plt.close()

c1_ = scaler.inverse_transform(c1)
coef_arr[mask_img.get_data() != 0] = c1_ - c1_.min()
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_anat(coef_img, display_mode='ortho', cut_coords=(5, -13, 1), black_bg=True, draw_cross=False,
                       title='Group 1 center')
pdf.savefig(); plt.close()


fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = c1 - c0
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='Difference of the centers: center 1 - center 2', colorbar=True)
pdf.savefig(); plt.close()

coef_arr[mask_img.get_data() != 0] = zmap
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
coef_img.to_filename(os.path.join(WD, "clusters_centers_zmap-diff.nii.gz"))

"""
L/R Thalami (medial)
*L/R Caudates, Putamen, Insular cortex
*L/R Cingulate gyrus Anterior and posterior, Precuneus, anterior part Calcarine fissure
*L/R Hypocampus, Amygdala, Fusiform, Lingual gyrus

*L/R Cerebellum VI

*L/R Temporal pole
*L/R Postcentral, Precentral

"""
fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='ortho', draw_cross=True, cut_coords=(-42, -16, 39),threshold=0.1,
                       title='Postcentral, Precentral (Z map)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='ortho', draw_cross=True, cut_coords=(25, 12, -40),threshold=0.1,
                       title='Temporal poles (Z map)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='ortho', draw_cross=True, cut_coords=(-26, -47, -23),threshold=0.1,
                       title='Cerebellum VI (Z map)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='ortho', draw_cross=True, cut_coords=(-30, -32, -10),threshold=0.1,
                       title='Hypocampus (Z map)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='ortho', draw_cross=True, cut_coords=(-21, -5, -18),threshold=0.1,
                       title='Amygdala (Z map)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='ortho', draw_cross=True, cut_coords=(-20, -54, -16),threshold=0.1,
                       title='Fusiform, Lingual gyrus (Z map)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='ortho', draw_cross=True, cut_coords=(-5, -40, 43),threshold=0.1,
                       title='Cingulate gyrus (ant./post.), Precuneus, Calcarine fissure (Z map)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='ortho', draw_cross=True, cut_coords=(-23, 8, -6),threshold=0.1,
                       title='Putamen,Caudates, Insular cortex (Z map)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='ortho', draw_cross=True, cut_coords=(1, -9, 2),threshold=0.1,
                       title='Thalami (Z map)', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_glass_brain(coef_img,   colorbar=True, plot_abs=False, threshold=0.15, title='Z map of the difference')
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='Z map of the difference', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='y', cut_coords=7,
                       title='Z map of the difference', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='x', cut_coords=7,
                       title='Z map of the difference', colorbar=True)
pdf.savefig(); plt.close()

fig = plt.figure()
coef_arr[mask_img.get_data() != 0] = tmap
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='T map of the difference', colorbar=True)
pdf.savefig(); plt.close()

pdf.close()


###############################################################################
# Signature
from nilearn import plotting, image
import  nibabel
from matplotlib.backends.backend_pdf import PdfPages
CLUST = 1

modelscv = np.load(os.path.join(WD, "rcv_gp1/rcv_024.npz")) # ['beta_cv', 'beta_refit', 'y_pred', 'y_true', 'proba_pred']

coef_refit = np.load(os.path.join(WD, "rcv_gp1/rcv_000.npz"))['beta_refit'] # ['beta_cv', 'beta_refit', 'y_pred', 'y_true', 'proba_pred']

if "beta_refit" in modelscv:
    np.sum((modelscv["beta_refit"] - coef_refit) ** 2) / np.sum(modelscv["beta_refit"] ** 2)
    plt.plot(modelscv["beta_refit"], coef_refit)
    np.corrcoef(modelscv["beta_refit"], coef_refit)[0, 1] # 0.99972945758044729

# CV
# --

mask_img = nibabel.load(os.path.join(WD, "mask.nii.gz"))
coef_arr = np.zeros(mask_img.get_data().shape)

#coef_refit = modelscv['coef_refitall_clust1']
#coef_refit0 = modelscv['coef_refitall_clust0']

coef_cv = modelscv['beta_cv']
#coef_avgcv = coef_cv.mean(axis=0)

pd.Series(coef_refit).describe(percentiles=[0.01, 0.05, 0.1, .25, .5, .75, 0.9])
"""
count    3.975590e+05
mean     4.122286e-06
std      3.906128e-04
min     -6.788251e-02
1%      -1.227855e-04
5%      -8.786389e-07
10%     -4.223494e-07
25%     -5.630435e-08
50%      1.234659e-07
75%      7.178728e-07
90%      2.282067e-06
max      4.815641e-02
"""
pd.Series(np.abs(coef_refit)).describe(percentiles=[0.01, 0.05, 0.1, .25, .5, .75, 0.9])

"""
count    3.975590e+05
mean     1.617330e-05
std      3.902996e-04
min      0.000000e+00
1%       0.000000e+00
5%       8.097614e-09
10%      2.370446e-08
25%      8.860295e-08
50%      3.224114e-07
75%      9.784164e-07
90%      2.997795e-06
max      6.788251e-02
"""

# Bootstrap
# ---------

import glob


coef_boot_img = [np.load(f)["beta_cv"] for f in glob.glob(os.path.join(WD, "rcv_gp1/rcv_*.npz"))]
coef_boot_img = np.concatenate(coef_boot_img)

print(coef_boot_img.shape)
# (735, 397559)
# (825, 397559)
# (940, 397559)

# 0.970044675253 0.979496828144 0.985384311961 0.836103443538
# 0.970044675253 0.979400439765 0.985496055984 0.835362864937
# 0.970044675253 0.979532698861 0.985607539235 0.833696216845

# old
# 0.942233238632 0.745839768565 0.806542301258 0.73123706908
# 0.942233238632 0.755023156032 0.834868776487 0.742830307793
# 0.942233238632 0.797933102049 0.878795596685 0.721951036469
# 0.942233238632 0.824568882048 0.896801442198 0.759808244301
# 0.942233238632 0.872093385138 0.940329152839 0.78211136126
# 0.942233238632 0.86648106966 0.939997706992 0.782311038707
#0.942233238632 0.867677242763 0.945523048513 0.797198890482

prefix_filename = os.path.join(WD,  "coeffs-map-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5")
pdf = PdfPages(prefix_filename+".pdf")

mask_img = nibabel.load(os.path.join(WD, "mask.nii.gz"))
coef_arr = np.zeros(mask_img.get_data().shape)

# Boot
fig = plt.figure()
coef_boot_img_avg = coef_boot_img.mean(axis=0)
coef_boot_img_std = coef_boot_img.std(axis=0)

coef_boot_img_avg[np.abs(coef_boot_img_avg) < 1 * coef_boot_img_std] = 0
coef_arr[mask_img.get_data() != 0] = coef_boot_img_avg
#coef_arr[mask_img.get_data() != 0] = coef_boot_img_avg / coef_boot_img_std
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
plotting.plot_glass_brain(coef_img,  vmax=5e-4, cmap=plt.cm.bwr, colorbar=True, plot_abs=False,
                          title='Signature avg boot where avg > sd')#, figure=fig, axes=ax)
pdf.savefig(); plt.close()
coef_img.to_filename(prefix_filename + "_boot-mean-sup-std.nii.gz")

# Refit all
coef_arr[mask_img.get_data() != 0] = coef_refit
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
#coef_img.to_filename(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5_refit.nii.gz"))
fig = plt.figure()
plotting.plot_glass_brain(coef_img,  vmax=5e-4, cmap=plt.cm.bwr, colorbar=True, plot_abs=False,
                          title='Signature refit')#, figure=fig, axes=ax)
pdf.savefig(); plt.close()
coef_img.to_filename(prefix_filename + "_refit-all.nii.gz")

# CV
coef = coef_cv.mean(axis=0)
coef[np.abs(coef) < coef_cv.std(axis=0)] = 0

coef_arr[mask_img.get_data() != 0] =  coef
#coef_arr[np.abs(coef_arr)<=1e-9] = np.nan
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)
fig = plt.figure()
plotting.plot_glass_brain(coef_img,  vmax=5e-4, cmap=plt.cm.bwr, colorbar=True, plot_abs=False, title='Signature avg 5CV, where avg > sd')#, figure=fig, axes=ax)
#coef_img.to_filename(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_enettv_0.1_0.1_0.8_5_c.nii.gz"))
pdf.savefig(); plt.close()
coef_img.to_filename(prefix_filename + "_refit-all-mean-sup-std.nii.gz")

# Refit all slices
coef_arr[mask_img.get_data() != 0] = coef_refit
coef_img = nibabel.Nifti1Image(coef_arr, affine=mask_img.affine)

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='z', cut_coords=7,
                       title='Signature',  vmax=1e-4, colorbar=True, cmap=plt.cm.bwr, threshold=1e-6)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='y', cut_coords=7,
                       title='Signature',  vmax=1e-4, colorbar=True, cmap=plt.cm.bwr, threshold=1e-6)
pdf.savefig(); plt.close()

fig = plt.figure()
plotting.plot_stat_map(coef_img, display_mode='x', cut_coords=7,
                       title='Signature',  vmax=1e-4, colorbar=True, cmap=plt.cm.bwr, threshold=1e-6)

pdf.savefig(); plt.close()

pdf.close()

"""
Manual inspection fsleye

# Positive clusters

R clusters
R
39.0% Postcentral Gyrus (Show/Hide)
16.0% Supramarginal Gyrus, anterior division
0.00052 (5e-4)

Regional increases of cortical thickness in untreated, first-episode major depressive disorder
ranslational Psychiatry 4(4):e378
"Areas with cortical thickness differences between healthy controls and patients with major depression (left) after FDR correction.
Scatterplots show the negative correlation between HDRS with right rostral middle frontal gyrus and right supramarginal gyrus (right).
Warmer colors (positive values) represent cortical thickening; cooler colors (negative values) represent signi
ficant cortical thinning in MDD patients.

R
52.0% Precuneous Cortex (Show/Hide)
+0.00013 (1e-4)

53.0% Middle Temporal Gyrus, temporooccipital part
+0.00017 (1e-4)

Right Amygdala (Show/Hide)
Right Parahippocampal Gyrus, anterior division
Rigth and Left Temporal Fusiform
+0.0003 (3e-4)

Left Cerebelum anterior parts of VIIIa and VIIIb
+0.00084

Voxel-based lesion symptom mapping analysis of depressive mood in patients with isolated cerebellar stroke: A pilot study
https://www.sciencedirect.com/science/article/pii/S2213158216302170
Voxel-wise subtraction and χ (Ayerbe et al., 2014) analyses indicated that damage to the left posterior cerebellar hemisphere was associated with depression. Significant correlations were also found between the severity of depressive symptoms and lesions in lobules VI, VIIb, VIII, Crus I, and Crus II of the left cerebellar hemisphere (Pcorrected = 0.045). Our results suggest that damage to the left posterior cerebellum is associated with increased depressive mood severity in patients with isolated cerebellar stroke.

# Negative

R (L)
100.0% Left Thalamus (Show/Hide)
R:-0.0009
L:-0.00003

R (L)
90.6% Brain-Stem (Show/Hide)
R:-0.0009
L:-0.0009


de Brouwer E.J.M., Kockelkoren R., Claus J.J., et al
"Hippocampal Calcifications: Risk Factors and Association with Cognitive Function.”
Radiology, June 12, 2018. https://doi.org/10.1148/radiol.2018172588

convert XTreatTivSite-clust1_enettv_0.1_0.1_0.8_5.pdf[1] images/signature_glassview_refit.png
convert XTreatTivSite-clust1_enettv_0.1_0.1_0.8_5.pdf[2] images/signature_glassview_agv5cv-thresh-sd.png
convert XTreatTivSite-clust1_enettv_0.1_0.1_0.8_5.pdf[3] images/signature_axial_refit.png
convert XTreatTivSite-clust1_enettv_0.1_0.1_0.8_5.pdf[4] images/signature_coronal_refit.png


# General Neuroimaging biomarkers associated to response

Fonseka TM, MacQueen GM and Kennedy SH (2017)
Neuroimaging biomarkers as predictors of treatment outcome in major depressive disorder.
Journal of Affective Disorders.

"""
