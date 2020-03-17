#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:06:56 2020

@author: edouard.duchesnay@cea.fr
"""
%load_ext autoreload
%autoreload 2

import os
import numpy as np
import glob
import pandas as pd
import nibabel
import brainomics.image_preprocessing as preproc
from brainomics.image_statistics import univ_stats, plot_univ_stats, residualize, ml_predictions
import shutil
# import mulm
# import sklearn
# import re
# from nilearn import plotting
import nilearn.image
import matplotlib
# matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
import re
# import glob
import seaborn as sns
import copy
import pickle

INPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/'
OUTPUT_PATH = '/home/ed203246/data/psy_sbox/analyses/202003_start-icaar_cat12vbm_predict-transition'

def PATH(dataset, modality='t1mri', mri_preproc='mwp1', scaling=None, harmo=None,
    type=None, ext=None, basepath=""):
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32

    return os.path.join(basepath, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "_" + scaling) +
                 ("" if harmo is None else "-" + harmo) +
                 ("" if type is None else "_" + type) + "." + ext)

def INPUT(*args, **kwargs):
    return PATH(*args, **kwargs, basepath=INPUT_PATH)

def OUTPUT(*args, **kwargs):
    return PATH(*args, **kwargs, basepath=OUTPUT_PATH)

################################################################################
#
# Dataset
#
################################################################################

dataset, target, target_num = 'icaar-start', "diagnosis", "diagnosis_num"
scaling, harmo = 'gs', 'raw'

# Create dataset if needed
if not os.path.exists(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy")):
    pop = pd.read_csv(INPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"))
    assert pop.shape == (167, 52)

    from  nitk.image import img_to_array, global_scaling
    imgs_arr, df_, target_img = img_to_array(pop.ni_path)
    df_ = df_.rename(columns=dict(path='ni_path'))
    assert np.all(pop[['participant_id', 'ni_path']] == df_[['participant_id', 'ni_path']])
    imgs_arr = global_scaling(imgs_arr, axis0_values=np.array(pop.tiv), target=1500)

    # Save all data to output
    mask_img = nibabel.load(INPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    mask_img.to_filename(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
    pop.to_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"), index=False)
    np.save(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), imgs_arr)

    del imgs_arr, df_, target_img, mask_img

pop = pd.read_csv(OUTPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"))
imgs_arr = np.load(OUTPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"))
mask_img = nibabel.load(OUTPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
mask_arr = mask_img.get_fdata() != 0
assert mask_arr.sum() == 368680

pop[target_num] = pop[target].map({'UHR-C': 1, 'UHR-NC': 0}).values
pop["GM_frac"] = pop.gm / pop.tiv
pop["sex_c"] = pop["sex"].map({0: "M", 1: "F"})
pop[target_num] = pop[target].map({'UHR-C': 1, 'UHR-NC': 0}).values

Xim = imgs_arr.squeeze()[:, mask_arr]


###############################################################################
# Select participants

msk = pop["diagnosis"].isin(['UHR-C', 'UHR-NC']) &  pop["irm"].isin(['M0'])
assert msk.sum() == 80

# Working population df
pop_w = pop.copy()

# Explore data
tab = pd.crosstab(pop["sex_c"][msk], pop["diagnosis"][msk], rownames=["sex_c"],
    colnames=["diagnosis"])
print(pop["diagnosis"][msk].describe())
print(pop["diagnosis"][msk].isin(['UHR-C']).sum())
print(tab)
"""
count         80
unique         2
top       UHR-NC
freq          53

27

diagnosis  UHR-C  UHR-NC
sex_c                   
F              8      23
M             19      30
"""
vars_clinic = []
vars_demo = ['age', 'sex']

# Finally, extract blocs
Xclin = pop_w[vars_clinic].values
Xdemo = pop_w[vars_demo].values
Xsite = pd.get_dummies(pop_w.site).values
Xdemoclin = np.concatenate([Xdemo, Xclin], axis=1)

# Some plot
df_ = pop[msk]
fig = plt.figure()
sns.lmplot("age", "GM_frac", hue=target, data=df_)
fig.suptitle("Aging is faster in Convertors")

fig = plt.figure()
sns.lmplot("age", "GM_frac", hue=target, data=df_[df_.age <= 24])
fig.suptitle("Aging is faster in Convertors in <= 24 years")
del df_

################################################################################
#
# PCA
#
################################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
pca_im = PCA(n_components=None)
PC_im_ = pca_im.fit_transform(scaler.fit_transform(Xim[msk, :]))
print("EV", pca_im.explained_variance_ratio_[:10])
"""
EV [0.03382146 0.02762525 0.02570336 0.01992885 0.01976731 0.01800009
 0.01731018 0.01645601 0.01613953 0.01606488]
"""

fig = plt.figure()
sns.scatterplot(pop["GM_frac"][msk], PC_im_[:, 0], hue=pop[target][msk])
fig.suptitle("PC1 capture global GM atrophy")

# sns.scatterplot(pop["GM_frac"][msk_tgt], PC_tgt_[:, 1], hue=pop[target][msk_tgt])

fig = plt.figure()
sns.scatterplot(PC_im_[:, 0], PC_im_[:, 1], hue=pop[target][msk])
fig.suptitle("PC1-PC2 no specific pattern")

del PC_im_

################################################################################
#
# Cross-Validation
#
################################################################################

################################################################################
# ML Utils

import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from nitk.stats import Residualizer

import sklearn.metrics as metrics
from nitk.utils import dict_product, parallel, aggregate_cv

def fit_predict(estimator_img, split):
    #residualizer = copy.deepcopy(residualizer)
    estimator_img = copy.deepcopy(estimator_img)
    train, test = split
    Xim_train, Xim_test, Xdemoclin_train, Xdemoclin_test, Z_train, Z_test, y_train =\
        Xim_[train, :], Xim_[test, :], Xdemoclin_[train, :], Xdemoclin_[test, :], Z_[train, :], Z_[test, :], y_[train]
    
    # Images based predictor

    # Residualization
    if RES_MOD is not None:
        if RES_MOD == 'RES-ALL':
            residualizer.fit(Xim_, Z_)
        elif RES_MOD == 'RES-TRAIN':
            residualizer.fit(Xim_train, Z_train)
        Xim_train = residualizer.transform(Xim_train, Z_train)
        Xim_test = residualizer.transform(Xim_test, Z_test)

    scaler = StandardScaler()
    Xim_train = scaler.fit_transform(Xim_train)
    Xim_test = scaler.transform(Xim_test)
    estimator_img.fit(Xim_train, y_train)

    y_test_img = estimator_img.predict(Xim_test)
    score_test_img = estimator_img.decision_function(Xim_test)
    score_train_img = estimator_img.decision_function(Xim_train)

    # Demographic/clinic based predictor
    estimator_democlin = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False)
    scaler = StandardScaler()
    Xdemoclin_train = scaler.fit_transform(Xdemoclin_train)
    Xdemoclin_test = scaler.transform(Xdemoclin_test)
    estimator_democlin.fit(Xdemoclin_train, y_train)
    y_test_democlin = estimator_democlin.predict(Xdemoclin_test)
    score_test_democlin = estimator_democlin.decision_function(Xdemoclin_test)
    score_train_democlin = estimator_democlin.decision_function(Xdemoclin_train)

    # STACK DEMO + IMG
    estimator_stck = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=True)
    # SVC
    # from sklearn.svm import SVC
    # estimator_stck = SVC(kernel='rbf', probability=True, gamma=1 / 100)
    # GB
    # from sklearn.ensemble import GradientBoostingClassifier
    # estimator_stck = GradientBoostingClassifier()

    Xstck_train = np.c_[score_train_democlin, score_train_img]
    Xstck_test = np.c_[score_test_democlin, score_test_img]
    scaler = StandardScaler()
    Xstck_train = scaler.fit_transform(Xstck_train)
    Xstck_test = scaler.transform(Xstck_test)
    estimator_stck.fit(Xstck_train, y_train)

    y_test_stck = estimator_stck.predict(Xstck_test)
    score_test_stck = estimator_stck.predict_log_proba(Xstck_test)[:, 1]
    score_train_stck = estimator_stck.predict_log_proba(Xstck_train)[:, 1]

    return dict(y_test_img=y_test_img, score_test_img=score_test_img,
                y_test_democlin=y_test_democlin, score_test_democlin=score_test_democlin,
                y_test_stck=y_test_stck, score_test_stck=score_test_stck,
                coef_img=estimator_img.coef_)


################################################################################
# Settings

#-------------------------------------------------------------------------------
SETTING = "RES-TRAIN_LR_5CV"
RES_MOD = 'RES-TRAIN'
NSPLITS = 5

Xim_ = Xim[msk, :]
Xdemoclin_ = Xdemoclin[msk, :]
Xsite_ = Xsite[msk, :]
y_ = pop[target + "_num"][msk].values
formula_res, formula_full = "site + age + sex", "site + age + sex + " + target_num
residualizer = Residualizer(data=pop_w[msk], formula_res=formula_res, formula_full=formula_full)
Z_ = residualizer.get_design_mat()

# CV
cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=3)

# estimators
estimators_dict = dict(lr=lm.LogisticRegression(C=1e6, class_weight='balanced', fit_intercept=False))

#-------------------------------------------------------------------------------
SETTING = "RES-TRAIN_GB_5CV"
RES_MOD = 'RES-TRAIN'
NSPLITS = 5

Xim_ = Xim[msk, :]
Xdemoclin_ = Xdemoclin[msk, :]
Xsite_ = Xsite[msk, :]
y_ = pop[target + "_num"][msk].values
formula_res, formula_full = "site + age + sex", "site + age + sex + " + target_num
residualizer = Residualizer(data=pop_w[msk], formula_res=formula_res, formula_full=formula_full)
Z_ = residualizer.get_design_mat()

# CV
cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=3)

# estimators
estimators_dict = dict(gb=ensemble.GradientBoostingClassifier()) # 2min 15s / run

#-------------------------------------------------------------------------------
SETTING = "RES-TRAIN-PCA_LR_5CV"
RES_MOD = 'RES-TRAIN'
NSPLITS = 5
from sklearn.pipeline import Pipeline

Xim_ = Xim[msk, :]
Xdemoclin_ = Xdemoclin[msk, :]
Xsite_ = Xsite[msk, :]
y_ = pop[target + "_num"][msk].values
formula_res, formula_full = "site + age + sex", "site + age + sex + " + target_num
residualizer = Residualizer(data=pop_w[msk], formula_res=formula_res, formula_full=formula_full)
Z_ = residualizer.get_design_mat()

# CV
cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=3)

# estimators
lr = lm.LogisticRegression(C=1e6, class_weight='balanced', fit_intercept=False)
pca_ = PCA(n_components=50)
pca_lr = Pipeline([('pca', pca_), ('lr', lr)])
estimators_dict = dict(pcalr=pca_lr)

#-------------------------------------------------------------------------------
SETTING = "RES-ALL_LR_5CV"
RES_MOD = 'RES-ALL'
NSPLITS = 5

Xim_ = Xim[msk, :]
Xdemoclin_ = Xdemoclin[msk, :]
Xsite_ = Xsite[msk, :]
y_ = pop[target + "_num"][msk].values
formula_res, formula_full = "site + age + sex", "site + age + sex + " + target_num
residualizer = Residualizer(data=pop_w[msk], formula_res=formula_res, formula_full=formula_full)
Z_ = residualizer.get_design_mat()

# CV
cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=3)

# estimators
estimators_dict = dict(lr=lm.LogisticRegression(C=1e6, class_weight='balanced', fit_intercept=False))


#-------------------------------------------------------------------------------
SETTING = "RES-TRAIN_ENETTV_5CV"
RES_MOD = 'RES-TRAIN'
NSPLITS = 5

Xim_ = Xim[msk, :]
Xdemoclin_ = Xdemoclin[msk, :]
Xsite_ = Xsite[msk, :]
y_ = pop[target + "_num"][msk].values
formula_res, formula_full = "site + age + sex", "site + age + sex + " + target_num
residualizer = Residualizer(data=pop_w[msk], formula_res=formula_res, formula_full=formula_full)
Z_ = residualizer.get_design_mat()

# CV
cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=3)

# estimators
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov


if not os.path.exists(OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz")):
    Atv = nesterov_tv.linear_operator_from_mask(mask_img.get_fdata(), calc_lambda_max=True)
    Atv.save(OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz"))

Atv = LinearOperatorNesterov(filename=OUTPUT(dataset, scaling=None, harmo=None, type="Atv", ext="npz"))
assert np.allclose(Atv.get_singular_values(0), 11.942012807930546)

# Parameters
mod_str = 'enettv_0.1_0.1_0.8'
keys_ = mod_str.split("_")
algo, alpha, l1l2ratio, tvratio = keys_[0], float(keys_[1]), float(keys_[2]), float(keys_[3])
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)

conesta = algorithms.proximal.CONESTA(max_iter=10000)
estimator = estimators.LogisticRegressionL1L2TV(l1, l2, tv, Atv, algorithm=conesta,
                                            class_weight="auto", penalty_start=0)

estimators_dict = {mod_str:estimator}

"""
estimator.fit(X_train, y_train.ravel())
# Store prediction for micro avg
y_test_pred[test] = estimator.predict(X_test).ravel()
y_test_prob_pred[test] = estimator.predict_probability(X_test).ravel()#[:, 1]
"""

################################################################################
# RUN FOR EACH SETTING

cv_dict = {fold:split for fold, split in enumerate(cv.split(Xim_, y_))}
args_collection = dict_product(estimators_dict, cv_dict)

cv_res = parallel(fit_predict, args_collection, n_jobs=min(NSPLITS, 8))

#-------------------------------------------------------------------------------
# Save results

mod_str = list(estimators_dict.keys())[0]

# pickle results
result_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo,
    type="results-_%s" % mod_str, ext="pkl")
with open(result_filename, 'wb') as fd:
    pickle.dump(cv_dict, fd)

# Save model coeficients
mod_str = list(estimators_dict.keys())[0]
cv_coefs_ = {str(keys[1]):val['coef_img'] for keys, val in cv_res.items()}
modelcoef_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo,
    type="modelcoef-%s" % mod_str, ext="npz")
np.savez_compressed(modelcoef_filename, **cv_coefs_)

#-------------------------------------------------------------------------------
# Scores

aggregate = aggregate_cv(cv_res, args_collection, 1)

"""
a_ = np.load(modelcoef_filename)
[k for k in a.keys()]
"""

# Results

scores_ = dict(N=len(y_))
scores_.update({"count_%i"%lab:np.sum(y_ == lab) for lab in np.unique(y_)})
for mod_key, pred_key in aggregate.keys():
    #print(mod_key, pred_key)
    scores_[mod_key] = mod_key
    if "y_" in pred_key:
        scores_[pred_key.replace("y_", "bacc_")] = metrics.recall_score(y_, aggregate[(mod_key, pred_key)], average=None).mean(),
    elif "score_" in pred_key:
        scores_[pred_key.replace("score_", "auc_")] = metrics.roc_auc_score(y_, aggregate[(mod_key, pred_key)])

print("*** %s" % SETTING)
pd.DataFrame(scores_)

################################################################################
"""
*** RES-TRAIN_ENETTV_5CV
N	count_0	count_1	enettv_0.1_0.1_0.8	auc_test_democlin	auc_test_img	auc_test_stck	bacc_test_democlin	bacc_test_img	bacc_test_stck
80	53	    27	    enettv_0.1_0.1_0.8	0.673655	        0.777778	    0.792453	    0.606569	        0.691474	    0.720475

￼*** RES-TRAIN_LR_5CV
N	count_0	count_1	lr	auc_test_democlin	auc_test_img	auc_test_stck	bacc_test_democlin	bacc_test_img	bacc_test_stck
80	53	    27	    lr	0.673655	        0.680643	    0.690426	    0.606569	        0.616352	    0.626834

*** RES-ALL_LR_5CV (JUST TO SEE HOW MUCH IT IS BIASED)
N	count_0	count_1	lr	auc_test_democlin	auc_test_img	auc_test_stck	bacc_test_democlin	bacc_test_img	bacc_test_stck
80	53	    27	    lr	0.673655	        0.874214    	0.908456	    0.606569	        0.79385	        0.768344

*** PCA_LR_5CV (Koutsouleris)
N	count_0	count_1	pcalr	auc_test_democlin	auc_test_img	auc_test_stck	bacc_test_democlin	bacc_test_img	bacc_test_stck
80	53	    27	    pcalr	0.673655	        0.500349	    0.630328	    0.606569	        0.50559	        0.518519

*** RES-TRAIN-PCA_LR_5CV (all PCs same results as without PCA)
N	count_0	count_1	pcalr	auc_test_democlin	auc_test_img	auc_test_stck	bacc_test_democlin	bacc_test_img	bacc_test_stck
80	53	    27	    pcalr	0.673655	        0.680643	    0.690426	    0.606569	        0.616352	    0.626834

*** RES-TRAIN-PCA(50 PCs)_LR_5CV
N	count_0	count_1	pcalr	auc_test_democlin	auc_test_img	auc_test_stck	bacc_test_democlin	bacc_test_img	bacc_test_stck
0	80	53	27	pcalr	    0.673655	        0.675052	    0.689029	    0.606569	        0.616352	    0.607617

*** RES-TRAIN_GB_5CV
N	count_0	count_1	gb	auc_test_democlin	auc_test_img	auc_test_stck	bacc_test_democlin	bacc_test_img	bacc_test_stck
80	53	    27	    gb	0.673655	        0.614955	    0.628931	    0.606569	        0.543326	    0.580363
"""

###############################################################################
# Explore results

result_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo,
    type="results-_%s" % mod_str, ext="pkl")

with open(result_filename, 'rb') as fd:
    cv_dict = pickle.load(fd)

aggregate = aggregate_cv(cv_res, args_collection, 1)

#keys = list(zip(*[k for k in aggregate.keys()]))
res = pd.DataFrame({"_".join(k):val for k, val in aggregate.items()})
res["participant_id"] = pop_w["participant_id"][msk].values
res["diagnosis"] = pop_w[target][msk].values
res["y"] = pop_w[target + "_num"][msk].values
res["sex"] = pop_w["sex_c"][msk].values
res["age"] = pop_w["age"][msk].values
print(res)
res["prediction"] = res['%s_y_test_img' % mod_str].map({0:'UHR-NC', 1:'UHR-C'})
res['err'] = res['prediction'] != res["diagnosis"]
res['correct'] = res['prediction'] == res["diagnosis"]

prediction_filename = OUTPUT(dataset, scaling=scaling, harmo=harmo,
    type="results-_%s" % mod_str, ext="csv")
res.to_csv(prediction_filename, index=False)

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib qt
sns.scatterplot('%s_score_test_democlin' % mod_str, '%s_score_test_img' % mod_str, hue="diagnosis", style="err", data=res)
sns.scatterplot('age', '%s_score_test_img' % mod_str, hue="diagnosis", style="err", data=res)
sns.pairplot(hue="diagnosis", vars=['%s_score_test_democlin' % mod_str, '%s_score_test_img' % mod_str, 'age'], data=res)

# Classif rate p-value
import scipy.stats
acc, N = 0.691474, len(res[target])
pval = scipy.stats.binom_test(x=int(acc * N), n=N, p=0.5) / 2
print(pval)
# 0.0005263674286766717

# AUC p-value
HERE
score_ = res['%s_score_test_img' % mod_str]
scipy.stats.mannwhitneyu(score_[res[target] == 'UHR-NC'],
                         score_[res[target] == 'UHR-C'])

# MannwhitneyuResult(statistic=318.0, pvalue=2.6791990948692184e-05)


# Check prediction and sex
tab_true = pd.crosstab(res["sex"], res["diagnosis"], rownames=["sex"], colnames=["diagnosis"])
print(tab_true)
print(tab_true / tab_true.values.sum())
print(tab_true["UHR-C"] / tab_true.sum(axis=1))

"""
diagnosis  UHR-C  UHR-NC
sex                     
F              8      23
M             19      30

diagnosis   UHR-C  UHR-NC
sex                      
F          0.1000  0.2875
M          0.2375  0.3750

sex
F    0.258065
M    0.387755

Larger rate of convertion in Male
"""

tab_pred = pd.crosstab(res["sex"], res["prediction"], rownames=["sex"], colnames=["prediction"])
print(tab_pred)
print(tab_pred / tab_pred.values.sum())
print(tab_pred["UHR-C"] / tab_pred.sum(axis=1))
"""
prediction  UHR-C  UHR-NC
sex                      
F              14      17
M              22      27

prediction  UHR-C  UHR-NC
sex                      
F           0.175  0.2125
M           0.275  0.3375

sex
F    0.451613
M    0.448980

Image based prediction over-estimate transition in both sex
"""
err = res[res["err"] == True]
tab_err = pd.crosstab(err["sex"], err["diagnosis"], rownames=["sex"], colnames=["diagnosis"])
tab = pd.crosstab(res["sex"], res["diagnosis"], rownames=["sex"], colnames=["diagnosis"])

print(tab_err)
print(tab_err / tab_err.values.sum())
print(tab_err /tab)

print(tab_err["UHR-C"] / tab_err.sum(axis=1))
"""
diagnosis  UHR-C  UHR-NC
sex                     
F              1       7
M              7      10

diagnosis  UHR-C  UHR-NC
sex                     
F           0.04    0.28
M           0.28    0.40

diagnosis     UHR-C    UHR-NC
sex                          
F          0.125000  0.304348
M          0.368421  0.333333

sex
F    0.125000
M    0.411765
"""

HERE

tab_pred = pd.crosstab(res["sex"], res["prediction"], rownames=["sex"], colnames=["prediction"])
tab_true = pd.crosstab(res["sex"], res["diagnosis"], rownames=["sex"], colnames=["diagnosis"])

print("PREDICTIONS")
print(tab_pred)
print("TRUE")
print(tab_true)
print("RATIO PREDICTIONS / TRUE")
ratio = tab_pred / tab_true
print(tab_pred / tab_true)
"""
PREDICTIONS
prediction  UHR-C  UHR-NC
sex                      
F               9      22
M              32      17

TRUE
diagnosis  UHR-C  UHR-NC
sex                     
F              8      23
M             19      30

RATIO PREDICTIONS / TRUE
prediction     UHR-C    UHR-NC
sex                           
F           1.125000  0.956522
M           1.684211  0.566667

En général sur-estime le rique de transition

Chez les femmes OK
Chez les hommes:
- sur-estime le risque tansition
- sous-estime la non-tansition

OR: a/c / b/d
a = Number of exposed cases
b = Number of exposed non-cases
c = Number of unexposed cases
d = Number of unexposed non-cases

cases = UHR-C
exposed = predicted 
"""

OR = ratio.iloc[:, 0] / ratio.iloc[:, 1]
print(OR)
"""
sex
F    1.176136
M    2.972136
"""