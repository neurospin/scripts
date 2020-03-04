#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:06:56 2020

@author: edouard.duchesnay@cea.fr
"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
# import brainomics.image_atlas
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
# import scipy, scipy.ndimage
#import xml.etree.ElementTree as ET
import re
# import glob
import seaborn as sns
import copy


# 1) Inputs: phenotype
# 
# for the phenotypes
# INPUT_CSV_icaar_bsnip_biobd = '/neurospin/psy_sbox/start-icaar-eugei/phenotype'
#INPUT_CSV_schizconnect = '/neurospin/psy/schizconnect-vip-prague/participants_schizconnect-vip.tsv'
#INPUT_CSV_prague = '/neurospin/psy/schizconnect-vip-prague/participants_prague.tsv'

INPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/'
INPUT_PATH = '/home/ed203246/data/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm'

# PHENOTYPE_CSV = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv"

def INPUT(dataset, modality='t1mri', mri_preproc='mwp1', scaling=None, harmo=None, type=None, ext=None):
    # scaling: global scaling? in "raw", "gs"
    # harmo (harmonization): in [raw, ctrsite, ressite, adjsite]
    # type data64, or data32
    return os.path.join(INPUT_PATH, dataset + "_" + modality+ "_" + mri_preproc +
                 ("" if scaling is None else "_" + scaling) +
                 ("" if harmo is None else "-" + harmo) +
                 ("" if type is None else "_" + type) + "." + ext)


###############################################################################
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from nitk.stats import Residualizer

dataset, target, target_num = 'icaar-start', "diagnosis", "diagnosis_num"

# scaling, harmo = 'gs', 'res:site+age+sex(diag)'
scaling, harmo = 'gs', 'raw'

ni_arr = np.load(INPUT(dataset, scaling=scaling, harmo=harmo, type="data64", ext="npy"), mmap_mode='r')
pop = pd.read_csv(INPUT(dataset, scaling=None, harmo=None, type="participants", ext="csv"))
pop[target_num] = pop[target].map({'UHR-C': 1, 'UHR-NC': 0}).values

mask_img = nibabel.load(INPUT(dataset, scaling=None, harmo=None, type="mask", ext="nii.gz"))
mask_arr = mask_img.get_data() != 0


###############################################################################
# Select participants

msk = pop["diagnosis"].isin(['UHR-C', 'UHR-NC']) &  pop["irm"].isin(['M0'])
assert msk.sum() == 80
# msk = pop["diagnosis"].isin(['UHR-C', 'UHR-NC']) &  pop["irm"].isin(['M0']) & clin_df["sex"].isin([0])
# assert msk.sum() == 49
clin_df = pop[msk]
ni_arr = ni_arr[msk]

clin_df["sex_c"] = clin_df["sex"].map({0: "M", 1: "F"})
tab = pd.crosstab(clin_df["sex_c"], clin_df["diagnosis"], rownames=["sex_c"], colnames=["diagnosis"])
print(clin_df["diagnosis"].describe())
print(clin_df["diagnosis"].isin(['UHR-C']).sum())
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

###############################################################################
# Datasets

Xim = ni_arr[:, 0, mask_arr].astype('float64')

# Anton, want to do a PCA on the data (it is biased, should be perfformed within the CV)
# from sklearn.decomposition import PCA
# pca = PCA()
# Xim = pca.fit_transform(Xim)

clin_df["diagnosis_num"] = clin_df["diagnosis"].map({'UHR-C': 1, 'UHR-NC': 0}).values
clin_df = clin_df.reset_index(drop=True)
clin_df_ = clin_df.copy()

# Randomize target ?
# clin_df_[target_num] = np.random.permutation(clin_df_[target_num].values)

Xdemo = clin_df_[["sex", "age"]].values
#Xdemo = clin_df_[["age"]].values

y = clin_df_[target_num].values


###############################################################################
# ML utils
import sklearn.metrics as metrics
from nitk.utils import dict_product, parallel, aggregate_cv

def fit_predict(estimator_img, split):
    #residualizer = copy.deepcopy(residualizer)
    estimator_img = copy.deepcopy(estimator_img)
    train, test = split
    Xim_train, Xim_test, Xdemo_train, Xdemo_test, Z_train, Z_test, y_train =\
        Xim[train, :], Xim[test, :], Xdemo[train, :], Xdemo[test, :], Z[train, :], Z[test, :], y[train]
    
    # Images based predictor
    if RES_MOD is not None:
        if RES_MOD == 'RES_ALL':
            residualizer.fit(Xim, Z)
        elif RES_MOD == 'RES_TRAIN':
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

    # Demographic based predictor
    estimator_demo = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False)
    scaler = StandardScaler()
    Xdemo_train = scaler.fit_transform(Xdemo_train)
    Xdemo_test = scaler.transform(Xdemo_test)
    estimator_demo.fit(Xdemo_train, y_train)
    y_test_demo = estimator_demo.predict(Xdemo_test)
    score_test_demo = estimator_demo.decision_function(Xdemo_test)
    score_train_demo = estimator_demo.decision_function(Xdemo_train)

    # STACK DEMO + IMG
    estimator_stck = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=True)
    # SVC
    # from sklearn.svm import SVC
    # estimator_stck = SVC(kernel='rbf', probability=True, gamma=1 / 100)
    # GB
    # from sklearn.ensemble import GradientBoostingClassifier
    # estimator_stck = GradientBoostingClassifier()

    Xstck_train = np.c_[score_train_demo, score_train_img]
    Xstck_test = np.c_[score_test_demo, score_test_img]
    scaler = StandardScaler()
    Xstck_train = scaler.fit_transform(Xstck_train)
    Xstck_test = scaler.transform(Xstck_test)
    estimator_stck.fit(Xstck_train, y_train)

    y_test_stck = estimator_stck.predict(Xstck_test)
    score_test_stck = estimator_stck.predict_log_proba(Xstck_test)[:, 1]
    score_train_stck = estimator_stck.predict_log_proba(Xstck_train)[:, 1]

    return dict(y_test_img=y_test_img, score_test_img=score_test_img,
                y_test_demo=y_test_demo, score_test_demo=score_test_demo,
                y_test_stck=y_test_stck, score_test_stck=score_test_stck)


###############################################################################
# CV

# config
#clin_df_ = clin_df.sample(frac=1)
formula_res, formula_full = "site + age + sex", "site + age + sex + " + target_num
# formula_res, formula_full = "site", "site + age + sex"
# formula_res, formula_full = "site", None
# formula_res, formula_full = "site", "site + age + sex"
# formula_res, formula_full = "site + age + sex", None

residualizer = Residualizer(data=clin_df_, formula_res=formula_res, formula_full=formula_full)
Z = residualizer.get_design_mat()

#RES_MOD = 'RES_ALL'
RES_MOD = 'RES_TRAIN'
# RES_MOD = None
NSPLITS = 5

# CV
cv = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=3)
cv_dict = {fold:split for fold, split in enumerate(cv.split(Xim, y))}

# estimators
estimators_dict = dict(lr=lm.LogisticRegression(C=1e6, class_weight='balanced', fit_intercept=False))
estimators_dict = dict(gb=ensemble.GradientBoostingClassifier()) # 2min 15s / run

# Run
args_collection = dict_product(estimators_dict, cv_dict)
cv_res = parallel(fit_predict, args_collection, n_jobs=min(NSPLITS, 8))

###############################################################################
# Results

aggregate = aggregate_cv(cv_res, args_collection, 1)

mod_keys = set(list(zip(*[k for k in aggregate.keys()]))[0])

print("%s %iCV" % (RES_MOD, cv.n_splits), "res:%s(%s)" % (formula_res, formula_full))
for mod_key in mod_keys:
    print(mod_key, "IMG ", dict(
    baccs_test = metrics.recall_score(y, aggregate[(mod_key, 'y_test_img')], average=None).mean(),
    aucs_test = metrics.roc_auc_score(y, aggregate[(mod_key,  'score_test_img')]),
    recalls_test = metrics.recall_score(y, aggregate[(mod_key, 'y_test_img')], average=None)))
    print(mod_key, "DEMO", dict(
    baccs_test = metrics.recall_score(y, aggregate[(mod_key, 'y_test_demo')], average=None).mean(),
    aucs_test = metrics.roc_auc_score(y, aggregate[(mod_key,  'score_test_demo')]),
    recalls_test = metrics.recall_score(y, aggregate[(mod_key, 'y_test_demo')], average=None)))
    print(mod_key, "STCK", dict(
    baccs_test = metrics.recall_score(y, aggregate[(mod_key, 'y_test_stck')], average=None).mean(),
    aucs_test = metrics.roc_auc_score(y, aggregate[(mod_key,  'score_test_stck')]),
    recalls_test = metrics.recall_score(y, aggregate[(mod_key, 'y_test_stck')], average=None)))

"""
RES_TRAIN 5CV res:site + age + sex(site + age + sex + diagnosis_num)

lr IMG  {'baccs_test': 0.6163522012578616, 'aucs_test': 0.6806429070580013, 'recalls_test': array([0.56603774, 0.66666667])}
lr DEMO {'baccs_test': 0.6065688329839274, 'aucs_test': 0.673654786862334, 'recalls_test': array([0.50943396, 0.7037037 ])}
lr STCK {'baccs_test': 0.6268343815513626, 'aucs_test': 0.6904262753319357, 'recalls_test': array([0.69811321, 0.55555556])}

gb IMG  {'baccs_test': 0.5527603074772885, 'aucs_test': 0.5988819007686932, 'recalls_test': array([0.69811321, 0.40740741])}
gb DEMO {'baccs_test': 0.6065688329839274, 'aucs_test': 0.673654786862334, 'recalls_test': array([0.50943396, 0.7037037 ])}
gb STCK {'baccs_test': 0.5618448637316562, 'aucs_test': 0.6114605171208944, 'recalls_test': array([0.67924528, 0.44444444])}
"""
###############################################################################
# Explore results

#keys = list(zip(*[k for k in aggregate.keys()]))
res = pd.DataFrame({"_".join(k):val for k, val in aggregate.items()})
res["diagnosis"] = clin_df_["diagnosis"]
res["sex"] = clin_df_["sex_c"]
res["age"] = clin_df_["age"]
print(res)
res["prediction"] = res['lr_y_test_img'].map({0:'UHR-NC', 1:'UHR-C'})
res['err'] = res['prediction'] != res["diagnosis"]

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib qt
sns.scatterplot('lr_score_test_demo', 'lr_score_test_img', hue="diagnosis", style="err", data=res)
sns.scatterplot('age', 'lr_score_test_img', hue="diagnosis", style="err", data=res)
sns.pairplot(hue="diagnosis", vars=['lr_score_test_demo', 'lr_score_test_img', 'age'], data=res)

# Classif rate p-value
from scipy import stats
acc, N = 0.6163522012578616, len(y)
pval = scipy.stats.binom_test(x=int(acc * N), n=N, p=0.5) / 2
print(pval)
# 0.028332213172560558

# AUC p-value
print(stats.mannwhitneyu(res['lr_score_test_img'][y == 0].values, res['lr_score_test_img'][y == 1].values))
# MannwhitneyuResult(statistic=457.0, pvalue=0.004331010744889353)

# Check prediction and sex
tab = pd.crosstab(res["sex"], res["diagnosis"], rownames=["sex"], colnames=["diagnosis"])
print(tab)
print(tab / tab.values.sum())
print(tab["UHR-C"] / tab.sum(axis=1))

"""
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

tab = pd.crosstab(res["sex"], res["prediction"], rownames=["sex"], colnames=["prediction"])
print(tab)
print(tab / tab.values.sum())
print(tab["UHR-C"] / tab.sum(axis=1))
"""
prediction  UHR-C  UHR-NC
sex                      
F               9      22
M              32      17

prediction   UHR-C  UHR-NC
sex                       
F           0.1125  0.2750
M           0.4000  0.2125

sex
F    0.290323
M    0.653061
"""

err = res[res["err"] == True]
tab_err = pd.crosstab(err["sex"], err["diagnosis"], rownames=["sex"], colnames=["diagnosis"])
tab = pd.crosstab(res["sex"], res["diagnosis"], rownames=["sex"], colnames=["diagnosis"])

print(tab_err)
print(tab_err / tab_err.values.sum())
print(tab_err /tab)

print(tab_err["UHR-C"] / tab_err.sum(axis=1))
"""

print(tab_err)...
diagnosis  UHR-C  UHR-NC
sex                     
F              4       5
M              5      18

diagnosis    UHR-C   UHR-NC
sex                        
F          0.12500  0.15625
M          0.15625  0.56250

diagnosis     UHR-C    UHR-NC
sex                          
F          0.500000  0.217391
M          0.263158  0.600000

Male UHR-C are correclty classifed (err = 26%)
Female UHR-NC are correclty classifed (err = 26%)
Male UHR-NC are missclassified (err 60%) 
Female UHR-C are missclassified (err  50%)
"""

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