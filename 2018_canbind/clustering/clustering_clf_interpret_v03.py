#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:29:19 2018

@author: ed203246

laptop to desktop
rsync -azvun /home/edouard/data/psy/canbind/models/clustering_v03/* ed203246@is234606.intra.cea.fr:/neurospin/psy/canbind/models/clustering_v03/

desktop to laptop
rsync -azvu ed203246@is234606.intra.cea.fr:/neurospin/psy/canbind/models/clustering_v03/* /home/edouard/data/psy/canbind/models/clustering_v03/

WD = '/neurospin/psy/canbind'
WD = '/home/edouard/data/psy/canbind'
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
from nilearn import datasets, plotting, image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import scipy, scipy.ndimage
#import nilearn.plotting

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing
import sklearn.metrics as metrics

import getpass

# cookiecutter Directory structure
# https://drivendata.github.io/cookiecutter-data-science/
if getpass.getuser() == 'ed203246':
    BASEDIR = '/neurospin/psy/canbind'
elif getpass.getuser() == 'edouard':
    BASEDIR = '/home/edouard/data/psy/canbind'

# OUTPUT = os.path.join(WD, "models", "vbm_resp_%s" % vs)
#WD = os.path.join('/neurospin/psy/canbind/models/clustering_v03')
WD = os.path.join(BASEDIR, "models", "clustering_v03")

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
# Graphics
###############################################################################

sns.set(style="whitegrid")

palette_resp = {"NonResponder":sns.color_palette()[0],
           "Responder":sns.color_palette()[2]}
alphas_clust = {1:1, 0:.5}

###############################################################################
# Clustering
###############################################################################

cluster_tot = pd.read_csv(os.path.join(WD, "clusters.csv"))
# cluster_tot = cluster_tot[cluster_tot.columns[:8]]

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

Xim_g1 = img_tot['XTotTivSiteCtr'][mask_gp1]
assert Xim_g1.shape == (66, 397559)
y_g1 = np.array(pop_tot['respond_wk16_num'][mask_gp1], dtype=int)

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
# ['age', 'sex_num', 'educ', 'age_onset', 'mde_num', 'madrs_Baseline', 'duration']
Xclin_g1 = np.asarray(democlin_g1)

# Grp0
democlin_g0 = democlin[mask_gp0_withintreat]
assert np.all(democlin_g0.pop('participant_id').values == pop_tot['participant_id'][mask_gp0].values)
assert np.all(democlin_g0.pop("respond_wk16").values == pop_tot['respond_wk16'][mask_gp0].values)
Xclin_g0 = np.asarray(democlin_g0)

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

y_g1 = np.array(pop_tot['respond_wk16_num'][mask_gp1], dtype=int)
y_g0 = np.array(pop_tot['respond_wk16_num'][mask_gp0], dtype=int)
xgrps_g1 = cluster_tot["proj_XCtlSes1TivSiteCtr"][mask_gp1].values
xgrps_g0 = cluster_tot["proj_XCtlSes1TivSiteCtr"][mask_gp0].values

if CLUST == 1:
    assert Xim_g1.shape == (66, 397559)
    Xim, y = Xim_g1, y_g1
    Xclin = Xclin_g1
    xgrps = xgrps_g1
    assert np.all(np.array([np.sum(lab==y) for lab in np.unique(y)]) == (16, 50))
elif CLUST == 0:
    assert Xim_g0.shape == (58, 397559)
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
coefs_cv_stck = np.zeros((NFOLDS, 3))
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
    xgrps_train, xgrps_test = xgrps[train][:, None], xgrps[test][:, None]

    X_train_img = scaler.fit_transform(X_train_img)
    X_test_img = scaler.transform(X_test_img)
    X_train_clin = scaler.fit_transform(X_train_clin)
    X_test_clin = scaler.transform(X_test_clin)
    xgrps_train = scaler.fit_transform(xgrps_train)
    xgrps_test = scaler.transform(xgrps_test)

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
    estimator_clin = lm.LogisticRegression(class_weight='balanced',
                                           fit_intercept=True, C=1)
#                                           fit_intercept=False, C=C)
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
            estimator_clin.decision_function(X_train_clin).ravel(),
            xgrps_train
            ]
    X_test_stck = np.c_[
            np.dot(X_test_img, estimator_img.beta).ravel(),
            estimator_clin.decision_function(X_test_clin).ravel(),
            xgrps_test
            ]
    X_train_stck = scaler.fit(X_train_stck).transform(X_train_stck)
    X_test_stck = scaler.transform(X_test_stck)

    #
    estimator_stck = lm.LogisticRegression(class_weight='balanced',
                                           fit_intercept=True, C=1)
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

print("#", auc_test_img_microavg, bacc_test_img_microavg, acc_test_img_microavg, recall_test_img_microavg)
print("#", auc_test_img)
print("#", recalls_test_img)
print("#", acc_test_img)

# 0.70375 0.6637500000000001 0.6515151515151515 [0.6875 0.64  ]
# [0.625, 0.53333333333333344, 0.76666666666666672, 0.8666666666666667, 0.66666666666666674]
# [array([ 0.75,  0.6 ]), array([ 0.33333333,  0.4       ]), array([ 0.66666667,  0.7       ]), array([ 1. ,  0.7]), array([ 0.66666667,  0.8       ])]
# [0.6428571428571429, 0.38461538461538464, 0.69230769230769229, 0.76923076923076927, 0.76923076923076927]

# Micro Avg Clin
recall_test_clin_microavg = metrics.recall_score(y, y_test_pred_clin, average=None)
recall_train_clin_microavg = metrics.recall_score(y, y_train_pred_clin, average=None)
bacc_test_clin_microavg = recall_test_clin_microavg.mean()
auc_test_clin_microavg = metrics.roc_auc_score(y, y_test_prob_pred_clin)
acc_test_clin_microavg = metrics.accuracy_score(y, y_test_pred_clin)

print("#", auc_test_clin_microavg, bacc_test_clin_microavg, acc_test_clin_microavg, recall_test_clin_microavg)
# 0.6387499999999999 0.6 0.6515151515151515 [0.5 0.7]

# Micro Avg Stacking
recall_test_stck_microavg = metrics.recall_score(y, y_test_pred_stck, average=None)
recall_train_stck_microavg = metrics.recall_score(y, y_train_pred_stck, average=None)
bacc_test_stck_microavg = recall_test_stck_microavg.mean()
auc_test_stck_microavg = metrics.roc_auc_score(y, y_test_prob_pred_stck)
acc_test_stck_microavg = metrics.accuracy_score(y, y_test_pred_stck)

print("#", auc_test_stck_microavg, bacc_test_stck_microavg, acc_test_stck_microavg, recall_test_stck_microavg)
# 0.72625 0.7537499999999999 0.7878787878787878 [0.6875 0.82  ]

"""
import json
with open(os.path.join(WD, 'XCtlSes1TivSiteCtr-clust-Ctl-1_5cv.json'), "w") as outfile:
    json.dump(CV, outfile)
"""
# Save test probabilities

# Img
pred = np.full(cluster_tot.shape[0], np.nan)
pred[mask_gp1] = y_test_pred_img
cluster_tot["y_cv_test_pred_img"] = pred

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp1] = y_test_decfunc_pred_img
cluster_tot["y_cv_test_decfunc_pred_img"] = proj

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp1] = y_test_prob_pred_img
cluster_tot["y_cv_test_prob_pred_img"] = proj

# Clin
pred = np.full(cluster_tot.shape[0], np.nan)
pred[mask_gp1] = y_test_pred_clin
cluster_tot["y_cv_test_pred_clin"] = pred

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp1] = y_test_prob_pred_clin
cluster_tot["y_cv_test_prob_pred_clin"] = proj

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp1] = y_test_decfunc_pred_clin
cluster_tot["y_cv_test_decfunc_pred_clin"] = proj

# Stacked
pred = np.full(cluster_tot.shape[0], np.nan)
pred[mask_gp1] = y_test_pred_stck
cluster_tot["y_cv_test_pred_stck"] = pred

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp1] = y_test_prob_pred_stck
cluster_tot["y_cv_test_prob_pred_stck"] = proj

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp1] = y_test_decfunc_pred_stck
cluster_tot["y_cv_test_decfunc_pred_stck"] = proj


# Provide scores for other subjects that group 1: refitt on all group1
# ----------------------------------------------------------
assert np.all(y == y_g1)
y_g0 = np.array(pop_tot['respond_wk16_num'][mask_gp0], dtype=int)

# Img

Xim_g1_sdtz = scaler.fit_transform(img_tot['XTotTivSiteCtr'][mask_gp1])
Xim_g0_sdtz = scaler.transform(img_tot['XTotTivSiteCtr'][mask_gp0])

coef_refit = np.load(os.path.join(WD, "rcv_gp1/rcv_000.npz"))['beta_refit'] # ['beta_cv', 'beta_refit', 'y_pred', 'y_true', 'proba_pred']
estimator_img.beta = coef_refit[:, None]

y_refitg1_g0_prob_pred_img = estimator_img.predict_probability(Xim_g0_sdtz).ravel()
y_refitg1_g0_decfunc_pred_img = np.dot(Xim_g0_sdtz, estimator_img.beta).ravel()
y_refitg1_g0_pred_img = estimator_img.predict(Xim_g0_sdtz).ravel()

# Clin

Xclin_g1_sdtz = scaler.fit_transform(Xclin_g1)
Xclin_g0_sdtz = scaler.fit_transform(Xclin_g0)

estimator_clin = lm.LogisticRegression(class_weight='balanced',
                                           fit_intercept=True, C=1)
estimator_clin.fit(Xclin_g1_sdtz, y_g1)

y_refitg1_g0_prob_pred_clin = estimator_clin.predict_proba(Xclin_g0_sdtz)[:, 1]
y_refitg1_g0_decfunc_pred_clin = estimator_clin.decision_function(Xclin_g0_sdtz)
y_refitg1_g0_pred_clin = estimator_clin.predict(Xclin_g0_sdtz).ravel()


# Stack
X_g0_stck = np.c_[
        np.dot(Xim_g0_sdtz, estimator_img.beta).ravel(),
        estimator_clin.decision_function(Xclin_g0_sdtz).ravel(),
        xgrps_g0
        ]
X_g0_stk_sdtz = scaler.fit_transform(X_g0_stck)

X_g1_stck = np.c_[
        np.dot(Xim_g1_sdtz, estimator_img.beta).ravel(),
        estimator_clin.decision_function(Xclin_g1_sdtz).ravel(),
        xgrps_g1
        ]
X_g1_stk_sdtz = scaler.fit_transform(X_g1_stck)

estimator_stck = lm.LogisticRegression(class_weight='balanced', fit_intercept=True, C=1)
estimator_stck.fit(X_g1_stk_sdtz, y_g1)

y_refitg1_g0_prob_pred_stk = estimator_stck.predict_proba(X_g0_stk_sdtz)[:, 1]
y_refitg1_g0_decfunc_pred_stk = estimator_stck.decision_function(X_g0_stk_sdtz)
y_refitg1_g0_pred_stk = estimator_stck.predict(X_g0_stk_sdtz).ravel()

print("# Refit G1 predict G0")
print("# Imaging")
print("# ", metrics.roc_auc_score(y_g0, y_refitg1_g0_prob_pred_img),
      metrics.recall_score(y_g0, y_refitg1_g0_pred_img, average=None))

print("# Clinic")
print("# ", metrics.roc_auc_score(y_g0, y_refitg1_g0_prob_pred_clin),
      metrics.recall_score(y_g0, y_refitg1_g0_pred_clin, average=None))

print("# Stacked")
print("# ", metrics.roc_auc_score(y_g0, y_refitg1_g0_prob_pred_stk),
      metrics.recall_score(y_g0, y_refitg1_g0_pred_stk, average=None))

# Refit G1 predict G0
# Imaging
#  0.42708333333333337 [0.6875     0.28571429]
# Clinic
#  0.46726190476190477 [0.3125     0.61904762]
# Stacked
#  0.43898809523809523 [0.3125     0.66666667]

# Save test probabilities

# Img
pred = np.full(cluster_tot.shape[0], np.nan)
pred[mask_gp0] = y_refitg1_g0_pred_img
cluster_tot["y_refitg1_g0_pred_img"] = pred

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp0] = y_refitg1_g0_prob_pred_img
cluster_tot["y_refitg1_g0_prob_pred_img"] = proj

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp0] = y_refitg1_g0_decfunc_pred_img
cluster_tot["y_refitg1_g0_decfunc_pred_img"] = proj

# Clin
pred = np.full(cluster_tot.shape[0], np.nan)
pred[mask_gp0] = y_refitg1_g0_pred_clin
cluster_tot["y_refitg1_g0_pred_clin"] = pred

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp0] = y_refitg1_g0_prob_pred_clin
cluster_tot["y_refitg1_g0_prob_pred_clin"] = proj

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp0] = y_refitg1_g0_decfunc_pred_clin
cluster_tot["y_refitg1_g0_decfunc_pred_clin"] = proj

# Stacked
pred = np.full(cluster_tot.shape[0], np.nan)
pred[mask_gp0] = y_refitg1_g0_pred_stk
cluster_tot["y_refitg1_g0_pred_stk"] = pred

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp0] = y_refitg1_g0_prob_pred_stk
cluster_tot["y_refitg1_g0_prob_pred_stk"] = proj

proj = np.full(cluster_tot.shape[0], np.nan)
proj[mask_gp0] = y_refitg1_g0_decfunc_pred_stk
cluster_tot["y_refitg1_g0_decfunc_pred_stk"] = proj

cluster_tot.to_csv(os.path.join(WD, "clusters.csv"), index=False)

###############################################################################
# Caracterize Cluster centers

from nilearn import plotting, image
import  nibabel


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
L/R Caudates, Putamen, Insular cortex
L/R Cingulate gyrus Anterior and posterior, Precuneus, anterior part Calcarine fissure
L/R Hypocampus, Amygdala, Fusiform, Lingual gyrus

L/R Cerebellum VI

L/R Temporal pole
L/R Postcentral, Precentral

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
###############################################################################
# Caracterize Cluster Statistics

# A) Cluster association with clinical variables
import os
import seaborn as sns
#import pandas as pd
import scipy.stats as stats

#CLUST=1

xls_filename = os.path.join(WD, "clusters_stats_demo-clin-vs-cluster.xlsx")

# add cluster information
#pop = pd.read_csv(os.path.join(WD, "population.csv"))
#clust = pd.read_csv(os.path.join(WD,  IMADATASET+"-clust%i"%CLUST +"_img-scores.csv"))
#pop = pd.merge(pop, clust[["participant_id", 'cluster']], on='participant_id')

assert np.all(cluster_tot[mask_treat_ses01]['participant_id'] == pop_tot[mask_treat_ses01]['participant_id'])
df = pd.merge(pop_tot[mask_treat_ses01], cluster_tot[mask_treat_ses01])#, on='participant_id')
assert df.shape[0] == 124

df["duration"] = df['age'] - df['age_onset']
df["group"] = df['cluster_XCtlSes1TivSiteCtr']

# cluster effect on dem/clinique
"""
['age', 'sex_num', 'educ']
['age_onset', 'mde_num', 'madrs_Baseline', 'madrs_Screening']
'respond_wk16'
"""
cols_ = ['age', 'educ', 'age_onset', "duration", 'mde_num', 'madrs_Baseline',
         'GMratio', 'WMratio', 'TIV_l',
         "group"]

means = df[cols_].groupby(by="group").mean().T.reset_index()
means.columns = ['var', 'mean_0', 'mean_1']

stds = df[cols_].groupby(by="group").std().T.reset_index()
stds.columns = ['var', 'std_0', 'std_1']
desc = pd.merge(means, stds)

stat_p = pd.DataFrame(
[[col] + list(stats.ttest_ind(
        df.loc[df.group==1, col],
        df.loc[df.group==0, col],
        equal_var=False, nan_policy='omit'))
    for col in cols_], columns=['var', 'T', 'T-pval'])

stat_clust_vs_var = pd.merge(desc, stat_p)


stat_np = list()
for col in cols_:
    clust = df["group"][df[col].notnull()]
    val = df.loc[df[col].notnull(), col]
    auc = metrics.roc_auc_score(clust, val)
    auc = max(auc, 1 - auc)
    wilcox = stats.mannwhitneyu(*[val[clust == r] for r in np.unique(clust)])
    stat_np.append([col, auc, wilcox.statistic, wilcox.pvalue])

stat_np = pd.DataFrame(stat_np,  columns=['var', 'auc', 'Mann–Whitney-U', 'MW-U-pval'])

stat_clust_vs_var = pd.merge(stat_clust_vs_var, stat_np)

print(stat_clust_vs_var)
"""
              var     mean_0     mean_1      std_0     std_1          T  \
0             age  41.580645  29.806452  12.277593  9.844427  -5.891231
1            educ  17.000000  16.622951   2.522034  1.950760  -0.928299
2       age_onset  23.779661  18.186441  11.365507  7.431236  -3.163806
3        duration  17.762712  11.932203  14.392693  9.428249  -2.602892
4         mde_num   4.047619   3.652174   2.251854  2.709796  -0.746835
5  madrs_Baseline  30.083333  29.866667   5.790968  5.512595  -0.209911
6         GMratio   0.459701   0.519362   0.031557  0.032438  10.380524
7         WMratio   0.299968   0.296910   0.023441  0.022140  -0.746770
8           TIV_l   1.503500   1.392831   0.155763  0.134993  -4.227700

         T-pval       auc  Mann–Whitney-U     MW-U-pval
0  3.796567e-08  0.774584           866.5  6.584649e-08
1  3.552030e-01  0.569011          1630.0  8.997774e-02
2  2.063409e-03  0.659868          1184.0  1.365327e-03
3  1.064894e-02  0.599971          1392.5  3.055953e-02
4  4.572176e-01  0.601190           770.5  4.942305e-02
5  8.340998e-01  0.510417          1762.5  4.228655e-01
6  1.788653e-18  0.901925           377.0  5.886210e-15
7  4.566439e-01  0.551249          1725.0  1.630527e-01
8  4.642436e-05  0.699272          1156.0  6.525133e-05

Grp 1 are younger (6 years), earlier onset (3 years) and shorter duration
"""


# B) Effect of stratification to disantangle Resp/NoResp (ROC)
df_ = df.copy()
df_["group"] = "All"
df = df.copy().append(df_)


variables = ['age', 'educ'] +\
    ['age_onset', 'mde_num', 'madrs_Baseline', 'madrs_Screening', "duration"]

res = list()
for var in variables:
    for lab in df["group"].unique():
        resp = df.loc[df.group == lab, "respond_wk16_num"]
        val = df.loc[df.group == lab, var]
        mask = val.notnull()
        resp, val = resp[mask], val[mask]
        auc = metrics.roc_auc_score(resp, val)
        auc = max(auc, 1 - auc)
        wilcox = stats.mannwhitneyu(*[val[resp == r] for r in np.unique(resp)])
        res.append([var, lab, auc, wilcox.statistic, wilcox.pvalue])

roc_clust_on_var = pd.DataFrame(res, columns=['var', 'clust', 'auc', 'Mann–Whitney-U', 'pval'])

with pd.ExcelWriter(xls_filename) as writer:
    stat_clust_vs_var.to_excel(writer, sheet_name='stat_clust_vs_var', index=False)
    roc_clust_on_var.to_excel(writer, sheet_name='roc_clust_on_var', index=False)


###############################################################################
# Caracterize Cluster Plot
import statsmodels.formula.api as smfrmla



pdf_filename = os.path.join(WD, "clusters_plot_demo-clin-vs-cluster.pdf")

df = pd.merge(pop_tot[mask_treat_ses01], cluster_tot[mask_treat_ses01])#, on='participant_id')
assert df.shape[0] == 124

df["duration"] = df['age'] - df['age_onset']
df["group"] = [str(int(g)) for g in df['cluster_XCtlSes1TivSiteCtr']]


pdf = PdfPages(pdf_filename)


# Caracterize Cluster: GM ~ age + group

model = smfrmla.ols("GMratio ~ group + age ", df).fit()
print(model.summary())

'''
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.4879      0.011     43.216      0.000       0.466       0.510
group          0.0519      0.006      8.246      0.000       0.039       0.064
age           -0.0007      0.000     -2.896      0.004      -0.001      -0.000
'''

sns.lmplot(x="age", y="GMratio", hue="group", data=df)
plt.ylabel("Gray matter / TIV")
plt.savefig(os.path.join(WD, "clusters_plot_GM~grp+age.svg"))

pdf.savefig(); plt.close()

df["group"] = [int(g) for g in df['cluster_XCtlSes1TivSiteCtr']]

# A) group association with clinical variables
#f = pop.copy()

xy_cols = [
        ["age_onset", "GMratio"],
        ["duration", "GMratio"],
        ["age", "GMratio"]]

for x_col, y_col in xy_cols:
    print(x_col, y_col)
    fig = plt.figure()
    fig.suptitle('%s by %s' % (x_col, y_col))
    for lab in df.group.unique():
        resp = df.loc[df.group == lab, "respond_wk16"]
        x = df.loc[df.group == lab, x_col]
        y = df.loc[df.group == lab, y_col]
        for r in resp.unique():
            plt.plot(x[resp == r], y[resp == r], "o", markeredgewidth=int(lab), markeredgecolor='black', color=palette_resp[r],
                     alpha=alphas_clust[lab], label="grp %i / %s" % (lab, r))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    pdf.savefig(); plt.close()


# B) Effect of stratification to disantangle Resp/NoResp
df_ = df.copy()
df_.group = "All"
df = df.copy().append(df_)

fig = plt.figure()
fig.suptitle('Duration, group and response ')
ax = sns.violinplot(x="group", y="duration", hue="respond_wk16", data=df,
               split=True, label="", legend_out = True, palette=palette_resp)
handles, labels = ax.get_legend_handles_labels()
ax = sns.swarmplot(x="group", y="duration", hue="respond_wk16", data=df,
              dodge=True, linewidth=1, edgecolor='black', palette=palette_resp)
ax.legend(handles, labels)
pdf.savefig(); plt.close()


fig = plt.figure()
fig.suptitle('Age, group and response ')
ax = sns.violinplot(x="group", y="age", hue="respond_wk16", data=df,
               split=True, palette=palette_resp)
handles, labels = ax.get_legend_handles_labels()
ax = sns.swarmplot(x="group", y="age", hue="respond_wk16", data=df,
              dodge=True, linewidth=1, edgecolor='black', palette=palette_resp)
ax.legend(handles, labels)
pdf.savefig(); plt.close()

fig = plt.figure()
fig.suptitle('Age onset, group and response ')
ax = sns.violinplot(x="group", y="age_onset", hue="respond_wk16", data=df,
               split=True, palette=palette_resp)
handles, labels = ax.get_legend_handles_labels()
ax = sns.swarmplot(x="group", y="age_onset", hue="respond_wk16", data=df,
              dodge=True, linewidth=1, edgecolor='black', palette=palette_resp)
ax.legend(handles, labels)
pdf.savefig(); plt.close()

fig = plt.figure()
fig.suptitle('Grey matter, group and response ')
ax = sns.violinplot(x="group", y="GMratio", hue="respond_wk16", data=df,
               split=True, label=None, legend_out = True, palette=palette_resp)
handles, labels = ax.get_legend_handles_labels()
ax = sns.swarmplot(x="group", y="GMratio", hue="respond_wk16", data=df,
              dodge=True, linewidth=1, edgecolor='black', palette=palette_resp)
ax.legend(handles, labels)
pdf.savefig(); plt.close()

pdf.close()


###############################################################################
# Discriminative pattern: predicted probability

df = pd.merge(pop_tot[mask_treat_ses01], cluster_tot[mask_treat_ses01])#, on='participant_id')
assert df.shape[0] == 124
df["duration"] = df['age'] - df['age_onset']
df["group"] = df['cluster_XCtlSes1TivSiteCtr']
# df["prob_img"] = df["y_tot-refit_prob_pred_img"]


pdf = PdfPages(os.path.join(WD, "clusters_discrimative-scatter.pdf"))

# --------------------------------
# 1) discriminative = clinic + img

# for group 1: stacked LR on the top of y_test_prob_pred_clin, y_test_prob_pred_img
# logistic regression(y_test_prob_pred_clin, y_test_prob_pred_img)


dfclust1 = df.loc[df["group"] == 1, ['y_cv_test_decfunc_pred_clin', 'y_cv_test_decfunc_pred_img', "respond_wk16"]]
X = np.array(dfclust1[['y_cv_test_decfunc_pred_clin', 'y_cv_test_decfunc_pred_img']])
y = np.array(dfclust1.respond_wk16.map({'NonResponder':0, 'Responder':1}))
scaler = preprocessing.StandardScaler()
X = scaler.fit(X).transform(X)
estimator_stck = lm.LogisticRegression(class_weight='balanced', fit_intercept=True, C=1)
estimator_stck.fit(X, y)

recall_post_stck_microavg = metrics.recall_score(y, estimator_stck.predict(X), average=None)
bacc_post_stck_microavg = recall_post_stck_microavg.mean()
auc_post_stck_microavg = metrics.roc_auc_score(y, estimator_stck.predict_proba(X)[:, 1])
acc_post_stck_microavg = metrics.accuracy_score(y,  estimator_stck.predict(X))

print("#", auc_post_stck_microavg, bacc_post_stck_microavg, acc_post_stck_microavg, recall_post_stck_microavg)
# 0.7462500000000001 0.70375 0.7121212121212122 [0.6875 0.72  ]

print("# ", estimator_stck.coef_)
#  [[0.3012868 0.7138052]]

print("# ", estimator_stck.intercept_)
#  [0.10000225]

df.loc[df["group"] == 1, "y_post_decfunc_pred_stck"] = estimator_stck.decision_function(X)
df.loc[df["group"] == 1, "y_post_prob_pred_stck"] = estimator_stck.predict_proba(X)[:, 1]


# contour
# https://matplotlib.org/1.3.0/examples/pylab_examples/contour_demo.html

nx = ny = 100
x = np.linspace(0.1, 0.7, num=nx)
y = np.linspace(0.0, 0.9, num=ny)
xx, yy = np.meshgrid(x, y)
z_proba = estimator_stck.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
z_proba = z_proba.reshape(xx.shape)

fig = plt.figure()
fig.suptitle('Discriminative pattern (Clinic+Imag.)')

plt.scatter(df["y_cv_test_decfunc_pred_clin"], df["y_cv_test_decfunc_pred_img"], c=[palette[res] for res in df.respond_wk16])
#sns.lmplot(x="y_test_prob_pred_clin", y="y_test_prob_pred_img", hue="respond_wk16" , data=df, fit_reg=False, palette=palette, axis=g.ax_joint)
CS1 = plt.contour(xx, yy, z_proba, 6, levels=[0.5], colors='k')#, axis=g.ax_joint)
CS2 = plt.contour(xx, yy, z_proba, 6, levels=[0.25, 0.75], linestyles="dashed", colors='grey')#, axis=g.ax_joint)
plt.clabel(CS1, CS1.levels, fontsize=9, inline=1)
plt.clabel(CS2, CS2.levels, fontsize=9, inline=1)
plt.xlabel("Clinic desision function.")
plt.ylabel("Imaging proba.")
pdf.savefig(); plt.close()



fig = plt.figure(figsize=(9, 8))
ax1 = plt.subplot(211)
fig.suptitle('Treatment response prediction')
sns.kdeplot(df["y_cv_test_decfunc_pred_stck"][(df["group"] == 1) & (df["respond_wk16"] == "Responder")],
            color=palette_resp["Responder"], shade=True, label="Responder")
sns.kdeplot(df["y_cv_test_decfunc_pred_stck"][(df["group"] == 1) & (df["respond_wk16"] == "NonResponder")],
            color=palette_resp["NonResponder"], shade=True, label="NonResponder")
plt.setp(ax1.get_xticklabels(), visible=False)

"""
sns.kdeplot(df["y_cv_test_prob_pred_stck"][(df["group"] == 1) & (df["respond_wk16"] == "Responder")],
            color=palette["Responder"], shade=True, label="Responder")
sns.kdeplot(df["y_cv_test_prob_pred_stck"][(df["group"] == 1) & (df["respond_wk16"] == "NonResponder")],
            color=palette["NonResponder"], shade=True, label="NonResponder", clip=(0, 1))
"""

#clip
#plt.xlim(0, 1)
#pdf.savefig(); plt.close()


# 2) inter group vs supervised
#fig = plt.figure()
#fig.suptitle('Inter group vs supervised prediction on imaging')

ax2 = plt.subplot(212, sharex=ax1)
df["prediction"] =  df["y_cv_test_decfunc_pred_stck"].copy()
df.loc[df["prediction"].isnull(), "prediction"] = df.loc[df["prediction"].isnull(),
       "y_refitg1_g0_decfunc_pred_stk"]

#for (lab, resp), d in df.groupby(["group", "respond_wk16"]):
for (lab, resp), d in df.sort_values(["group", "respond_wk16"], ascending=False).groupby(["group", "respond_wk16"], sort=False):
    print(lab, resp)
    plt.plot(
            d["prediction"],
            d["proj_XCtlSes1TivSiteCtr"], "o", markeredgewidth=int(lab), markeredgecolor='black', color=palette_resp[resp],
                     alpha=alphas_clust[lab], label="grp %i / %s" % (lab, resp))

#plt.xlim(0, 1)

plt.axhline(y=0, ls='--', color="k")
plt.plot([0., 0.], [0, df["proj_XCtlSes1TivSiteCtr"].max()], ls='--', color="k")

plt.xlabel("Predicted decision function")
plt.ylabel("Inter-group axis")

plt.tight_layout()
plt.savefig(os.path.join(WD, "clusters_discrimative-scatter.svg"))
#plt.legend()
pdf.savefig(); plt.close()

pdf.close()
