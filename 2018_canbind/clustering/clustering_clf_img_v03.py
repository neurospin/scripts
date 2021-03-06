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
scaler = preprocessing.StandardScaler()
import sklearn.metrics as metrics

# cookiecutter Directory structure
# https://drivendata.github.io/cookiecutter-data-science/


# OUTPUT = os.path.join(WD, "models", "vbm_resp_%s" % vs)
WD = os.path.join('/neurospin/psy/canbind/models/clustering_v03')


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

# centers_ = np.load(os.path.join(WD, "clusters_centers.npz"))
# cluster_tot_ = pd.read_csv(os.path.join(WD, "clusters.csv"))
###############################################################################
# Clustering
###############################################################################

centers = dict()
XTotTivSiteCtr = img_tot['XTotTivSiteCtr']

# XTreatTivSiteCtr
XTreatSes1TivSiteCtrStdz = scaler.fit_transform(XTotTivSiteCtr[mask_treat_ses01])
clusterer_treat = KMeans(n_clusters=2, random_state=10)#, n_jobs=5)
clusterer_treat.fit(XTreatSes1TivSiteCtrStdz)
centers["XTreatSes1TivSiteCtr"] = np.copy(clusterer_treat.cluster_centers_)
#np.all(centers_["XTreatSes1TivSiteCtr"] == centers["XTreatSes1TivSiteCtr"])

# Check non regression on clustering based on Patients
orig = pd.read_csv("/neurospin/psy/canbind/models/clustering_v02/XTreatTivSite-clust1_img-scores.csv")[["participant_id", "respond_wk16", "cluster"]]
target = np.array(orig.cluster)
assert np.all(np.array(orig["participant_id"]) == np.array(pop_tot[mask_treat_ses01]["participant_id"]))
assert np.all(clusterer_treat.predict(XTreatSes1TivSiteCtrStdz) == target)


# XCtlSes1TivSiteCtr
XCtlSes1TivSiteCtrStdz = scaler.fit_transform(XTotTivSiteCtr[mask_ctl_ses01])
clusterer_ctl = KMeans(n_clusters=2, random_state=0, n_jobs=5)
clusterer_ctl.fit(XCtlSes1TivSiteCtrStdz)
# permute
clusterer_ctl.cluster_centers_ = clusterer_ctl.cluster_centers_[[1, 0]]
centers["XCtlSes1TivSiteCtr"] = np.copy(clusterer_ctl.cluster_centers_)
#np.all(centers_["XCtlSes1TivSiteCtr"] == centers["XCtlSes1TivSiteCtr"])

assert np.sum(target != clusterer_ctl.predict(XTreatSes1TivSiteCtrStdz)) == 4
metrics.confusion_matrix(pop_tot.respond_wk16_num[mask_treat_ses01], clusterer_ctl.predict(XTreatSes1TivSiteCtrStdz))
"""
array([[16, 16],
       [42, 50]])
"""

# Classify/Project controls and patients Standardized individually
grp = np.full(pop.shape[0], np.nan)
grp[mask_treat_ses01] = clusterer_ctl.predict(XTreatSes1TivSiteCtrStdz)
grp[mask_ctl_ses01] = clusterer_ctl.predict(XCtlSes1TivSiteCtrStdz)
cluster["cluster_XCtlSes1TivSiteCtr"] = grp

proj = np.full(pop.shape[0], np.nan)
diff = clusterer_ctl.cluster_centers_[1, :] - clusterer_ctl.cluster_centers_[0, :]
proj[mask_treat_ses01] = np.dot(XTreatSes1TivSiteCtrStdz, diff)
proj[mask_ctl_ses01] = np.dot(XCtlSes1TivSiteCtrStdz, diff)
cluster["proj_XCtlSes1TivSiteCtr"] = proj

# Classify/Project all subjects
XTotTivSiteCtrStdz = scaler.fit(XTotTivSiteCtr).transform(XTotTivSiteCtr)
cluster["cluster_XCtlSes1TivSiteCtr_all"] = clusterer_ctl.predict(XTotTivSiteCtrStdz)
cluster["proj_XCtlSes1TivSiteCtr_all"] = np.dot(XTotTivSiteCtrStdz, diff)


cluster.to_csv(os.path.join(WD, "clusters.csv"), index=False)
np.savez_compressed(os.path.join(WD, "clusters_centers.npz"),  **centers)

######################################
# Some QC with the previous clustering

orig = pd.read_csv("/neurospin/psy/canbind/models/clustering_v02/XTreatTivSite-clust1_img-scores.csv")[["participant_id", "respond_wk16", "cluster"]]
target = np.array(orig.cluster)
assert np.all(np.array(orig["participant_id"]) == np.array(pop_tot[mask_treat_ses01]["participant_id"]))
missmatch = cluster["cluster_XCtlSes1TivSiteCtr"][mask_treat_ses01] != target
pop_ = pop_tot[mask_treat_ses01]
pop_["orig"] = target
pop_["new"] = cluster["cluster_XCtlSes1TivSiteCtr"]
print(pop_[missmatch][['participant_id', 'age', 'sex', 'educ', 'respond_wk16', 'orig', 'new']])
"""
    participant_id   age  sex  educ  respond_wk16  orig  new
293   sub-TGH-0040  20.0  1.0  17.0     Responder     0  1.0
364   sub-TGH-0072  46.0  1.0  17.0     Responder     0  1.0
427   sub-UBC-0012  21.0  1.0  19.0  NonResponder     0  1.0
537   sub-UBC-0057  53.0  1.0  16.0     Responder     0  1.0
"""
assert np.sum(missmatch) == 4

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
# Shuffle for the site effect + use 42 (magic number) + 1 (Prime number) as seed
cv = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42+1)
#model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=C)
scaler = preprocessing.StandardScaler()
def balanced_acc(estimator, X, y):
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()
scorers = {'auc': 'roc_auc', 'bacc':balanced_acc, 'acc':'accuracy'}

###############################################################################
# Re-load Clustering
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
    X, y = Xim_g1, y_g1
    assert np.all(np.array([np.sum(lab==y) for lab in np.unique(y)]) == (16, 50))
elif CLUST == 0:
    Xim_g0 = img_tot['XTotTivSiteCtr'][mask_gp0]
    assert Xim_g0.shape == (58, 397559)
    y_g0 = np.array(pop_tot['respond_wk16_num'][mask_gp0], dtype=int)
    X, y = Xim_g0, y_g0


###############################################################################
# EnetTV
###############################################################################

import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

mask = nibabel.load(os.path.join(WD, "mask.nii.gz"))
# Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
# Atv.save(os.path.join(WD, "Atv.npz"))
Atv = LinearOperatorNesterov(filename=os.path.join(WD, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv.get_singular_values(0), 11.956104408414376)

# parameters
key = 'enettv_0.1_0.1_0.8'.split("_")
algo, alpha, l1l2ratio, tvratio = key[0], float(key[1]), float(key[2]), float(key[3])
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)

print(key, algo, alpha, l1, l2, tv)

y_test_pred = np.zeros(len(y))
y_test_prob_pred = np.zeros(len(y))
y_test_decfunc_pred = np.zeros(len(y))
y_train_pred = np.zeros(len(y))
coefs_cv = np.zeros((NFOLDS, X.shape[1]))

auc_test = list()
recalls_test = list()
acc_test = list()

for cv_i, (train, test) in enumerate(cv.split(X, y)):
    #for train, test in cv.split(X, y, None):
    print(cv_i)
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    conesta = algorithms.proximal.CONESTA(max_iter=10000)
    estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                    class_weight="auto", penalty_start=0)
    estimator.fit(X_train, y_train.ravel())
    # Store prediction for micro avg
    y_test_pred[test] = estimator.predict(X_test).ravel()
    y_test_prob_pred[test] = estimator.predict_probability(X_test).ravel()#[:, 1]
    #y_test_decfunc_pred[test] = estimator.decision_function(X_test)
    y_train_pred[train] = estimator.predict(X_train).ravel()
    # Compute score for macro avg
    auc_test.append(metrics.roc_auc_score(y_test, estimator.predict_probability(X_test).ravel()))
    recalls_test.append(metrics.recall_score(y_test, estimator.predict(X_test).ravel(), average=None))
    acc_test.append(metrics.accuracy_score(y_test, estimator.predict(X_test).ravel()))

    coefs_cv[cv_i, :] = estimator.beta.ravel()

# Micro Avg
recall_test_microavg = metrics.recall_score(y, y_test_pred, average=None)
recall_train_microavg = metrics.recall_score(y, y_train_pred, average=None)
bacc_test_microavg = recall_test_microavg.mean()
auc_test_microavg = metrics.roc_auc_score(y, y_test_prob_pred)
acc_test_microavg = metrics.accuracy_score(y, y_test_pred)

print("#", auc_test_microavg, bacc_test_microavg, acc_test_microavg)
print("#", auc_test)
print("#", recalls_test)
print("#", acc_test)

# Refit
conesta = algorithms.proximal.CONESTA(max_iter=10000)
estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                                class_weight="auto", penalty_start=0)
estimator.fit(scaler.fit_transform(X), y)

np.savez_compressed(os.path.join(WD, "XCtlSes1TivSiteCtr-clust-Ctl-%i"%CLUST +"_enettv_0.1_0.1_0.8_%icv.npz" % NFOLDS),
                    beta_cv=coefs_cv, beta_refit=estimator.beta.ravel(), y_pred=y_test_pred, y_true=y,
                    proba_pred=y_test_prob_pred)

# YEAH !
# 0.70625 0.66375 0.651515151515
# [0.57499999999999996, 0.96666666666666679, 0.73333333333333339, 0.93333333333333335, 0.53333333333333344]
# [array([ 0.5,  0.6]), array([ 1. ,  0.6]), array([ 0.66666667,  0.6       ]), array([ 1. ,  0.5]), array([ 0.33333333,  0.9       ])]
# [0.5714285714285714, 0.69230769230769229, 0.61538461538461542, 0.61538461538461542, 0.76923076923076927]


###############################################################################
# SVM

model = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=1)
estimator = make_pipeline(preprocessing.StandardScaler(), model)

%time cv_results = cross_validate(estimator=estimator, X=X, y=y, cv=cv, scoring=scorers, n_jobs=-1)
print(
      cv_results["test_auc"], cv_results["test_auc"].mean(), "\n",
      cv_results["test_bacc"], cv_results["test_bacc"].mean(), "\n",
      cv_results["test_acc"], cv_results["test_acc"].mean())

# Tune rnd
res = list()
for rnd in range(100):
    print(rnd)
    cv = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=rnd)
    #estimator = lm.LogisticRegression(class_weight='balanced', fit_intercept=False, C=10)
    cv_results = cross_validate(estimator=estimator, X=X, y=y, cv=cv, scoring=scorers, n_jobs=-1)
    r_ = [rnd, cv_results["test_auc"].mean(), cv_results["test_bacc"].mean()]
    print(r_)
    res.append(r_)

res = pd.DataFrame(res, columns=["rnd", "test_auc", "test_bacc"])

res["test_min"] = res[[ "test_auc", "test_bacc"]].min(axis=1)
res = res.sort_values("test_min", ascending=False)

"""
[ 0.3         0.8         0.56666667  0.6         0.2       ] 0.493333333333
 [ 0.375       0.8         0.51666667  0.7         0.3       ] 0.538333333333
 [ 0.42857143  0.69230769  0.61538462  0.53846154  0.46153846] 0.547252747253
"""

