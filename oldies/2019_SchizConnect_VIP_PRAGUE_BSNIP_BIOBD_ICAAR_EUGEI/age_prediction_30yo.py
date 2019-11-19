#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:32:50 2019

@author: anton
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import sklearn.linear_model as lm
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import is_classifier, clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import  make_scorer,accuracy_score,recall_score,precision_score
from sklearn import datasets
from sklearn.metrics import recall_score
from sklearn import svm, metrics, linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import  make_scorer,accuracy_score,recall_score,precision_score
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import LinearSVR
from sklearn.model_selection import LeaveOneGroupOut

PATH = '/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data'

ROI = pd.read_csv(os.path.join(PATH,'ROI_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep = '\t')
pheno = pd.read_csv(os.path.join(PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep='\t')
assert ROI.shape == (2697, 143)
assert pheno.shape == (3871, 46)

"""
SELECT PATIENTS OF ALL AGES
"""

## TRAIN CONTROL DATA (SchizConnect, VIP, PRAGUE, BSNIP)
#
#pheno_controls_train = pheno[(pheno.diagnosis == 'control') & (pheno.study.isin(['SCHIZCONNECT-VIP','PRAGUE','BSNIP']))]
#assert pheno_controls_train.shape == (843, 46)
#controls_train = pd.merge(pheno_controls_train, ROI, on='participant_id', how='left')
#assert controls_train.shape == (843, 188)
#controls_train = controls_train[controls_train.rTemTraGy.notna() & controls_train.age.notna() & controls_train.sex.notna() & controls_train.site.notna()]
#assert controls_train.shape == (621, 188)
#
#assert list(controls_train).index('l3thVen') == 46
#assert list(controls_train).index('rTemTraGy') == 187
#
#X_train_ctrl = controls_train.iloc[:,np.r_[46:188]]
#assert X_train_ctrl[X_train_ctrl.isnull().any(axis=1)].shape == (0, 142)
#assert X_train_ctrl.duplicated().sum() == 0
#X_train_ctrl = np.array(X_train_ctrl)
#assert X_train_ctrl.shape == (621, 142)
## info pour moi: il y a dans X_train_ctrl un sujet en plus que dans mon ancien X_train; il s'agit de 'INVFHF33VYJ' qui provient du fichier BSNIP_all_clinical_data,
## qui n'avait pas été intégré
#age_train_ctrl = np.array(controls_train.age)
#sex_train_ctrl = np.array(controls_train.sex)
#site_train_ctrl = np.array(controls_train.site)
#assert age_train_ctrl.shape == sex_train_ctrl.shape == site_train_ctrl.shape == (621,)
#
## centering by site
#X_train_ctrl_bySite = np.zeros(X_train_ctrl.shape)
#for s in set(site_train_ctrl):
#    m = site_train_ctrl == s
#    X_train_ctrl_bySite[m] = X_train_ctrl[m] - X_train_ctrl[m, :].mean(axis=0)
#    
#
## TEST CONTROL DATA (BIOBD)
#
#pheno_controls_test = pheno[(pheno.diagnosis == 'control') & (pheno.study == 'BIOBD')]
#assert pheno_controls_test.shape == (370, 46)
#controls_test = pd.merge(pheno_controls_test, ROI, on='participant_id', how='left')
#assert controls_test.shape == (370, 188)
#controls_test = controls_test[controls_test.rTemTraGy.notna() & controls_test.age.notna() & controls_test.sex.notna() & controls_test.site.notna()]
#assert controls_test.shape == (370, 188)
#
#assert list(controls_test).index('l3thVen') == 46
#assert list(controls_test).index('rTemTraGy') == 187
#
#X_test_ctrl = controls_test.iloc[:,np.r_[46:188]]
#assert X_test_ctrl[X_test_ctrl.isnull().any(axis=1)].shape == (0, 142)
#assert X_test_ctrl.duplicated().sum() == 0
#X_test_ctrl = np.array(X_test_ctrl)
#assert X_test_ctrl.shape == (370, 142)
#age_test_ctrl = np.array(controls_test.age)
#sex_test_ctrl = np.array(controls_test.sex)
#site_test_ctrl = np.array(controls_test.site)
#assert age_test_ctrl.shape == sex_test_ctrl.shape == site_test_ctrl.shape == (370,)
#
## centering by site
#X_test_ctrl_bySite = np.zeros(X_test_ctrl.shape)
#for s in set(site_test_ctrl):
#    m = site_test_ctrl == s
#    X_test_ctrl_bySite[m] = X_test_ctrl[m] - X_test_ctrl[m, :].mean(axis=0)
#
#
## TEST UHR DATA (ICAAR-START)
#
#pheno_uhr_test = pheno[(pheno.diagnosis.isin(['UHR-C','UHR-NC'])) & (pheno.irm == 'M0')]
#assert pheno_uhr_test.shape == (80, 46)
#uhr_test = pd.merge(pheno_uhr_test, ROI, on='participant_id', how='left')
#assert uhr_test.shape == (80, 188)
#uhr_test = uhr_test[uhr_test.rTemTraGy.notna() & uhr_test.age.notna() & uhr_test.sex.notna() & uhr_test.site.notna()]
#assert uhr_test.shape == (80, 188)
#
#assert list(uhr_test).index('l3thVen') == 46
#assert list(uhr_test).index('rTemTraGy') == 187
#
#X_test_uhr = uhr_test.iloc[:,np.r_[46:188]]
#assert X_test_uhr[X_test_uhr.isnull().any(axis=1)].shape == (0, 142)
#assert X_test_uhr.duplicated().sum() == 0
#X_test_uhr = np.array(X_test_uhr)
#assert X_test_uhr.shape == (80, 142)
#age_test_uhr = np.array(uhr_test.age)
#sex_test_uhr = np.array(uhr_test.sex)
#site_test_uhr = np.array(uhr_test.site)
#assert age_test_uhr.shape == sex_test_uhr.shape == site_test_uhr.shape == (80,)
#
#X_test_uhr_bySite = np.zeros(X_test_uhr.shape)
#for s in set(site_test_uhr):
#    m = site_test_uhr == s
#    X_test_uhr_bySite[m] = X_test_uhr[m] - X_test_uhr[m, :].mean(axis=0)


"""
SELECT PATIENTS WITH AGE <= 30
"""

pheno = pheno[pheno.age <= 30]


# TRAIN CONTROL DATA (SchizConnect, VIP, PRAGUE, BSNIP)

pheno_controls_train = pheno[(pheno.diagnosis == 'control') & (pheno.study.isin(['SCHIZCONNECT-VIP','PRAGUE','BSNIP']))]
assert pheno_controls_train.shape == (319, 46)
controls_train = pd.merge(pheno_controls_train, ROI, on='participant_id', how='left')
assert controls_train.shape == (319, 188)
controls_train = controls_train[controls_train.rTemTraGy.notna() & controls_train.age.notna() & controls_train.sex.notna() & controls_train.site.notna()]
assert controls_train.shape == (313, 188)

assert list(controls_train).index('l3thVen') == 46
assert list(controls_train).index('rTemTraGy') == 187

X_train_ctrl = controls_train.iloc[:,np.r_[46:188]]
assert X_train_ctrl[X_train_ctrl.isnull().any(axis=1)].shape == (0, 142)
assert X_train_ctrl.duplicated().sum() == 0
X_train_ctrl = np.array(X_train_ctrl)
assert X_train_ctrl.shape == (313, 142)
# info pour moi: il y a dans X_train_ctrl un sujet en plus que dans mon ancien X_train; il s'agit de 'INVFHF33VYJ' qui provient du fichier BSNIP_all_clinical_data,
# qui n'avait pas été intégré
age_train_ctrl = np.array(controls_train.age)
sex_train_ctrl = np.array(controls_train.sex)
site_train_ctrl = np.array(controls_train.site)
assert age_train_ctrl.shape == sex_train_ctrl.shape == site_train_ctrl.shape == (313,)

# centering by site
X_train_ctrl_bySite = np.zeros(X_train_ctrl.shape)
for s in set(site_train_ctrl):
    m = site_train_ctrl == s
    X_train_ctrl_bySite[m] = X_train_ctrl[m] - X_train_ctrl[m, :].mean(axis=0)
    

# TEST CONTROL DATA (BIOBD)

pheno_controls_test = pheno[(pheno.diagnosis == 'control') & (pheno.study == 'BIOBD')]
assert pheno_controls_test.shape == (101, 46)
controls_test = pd.merge(pheno_controls_test, ROI, on='participant_id', how='left')
assert controls_test.shape == (101, 188)
controls_test = controls_test[controls_test.rTemTraGy.notna() & controls_test.age.notna() & controls_test.sex.notna() & controls_test.site.notna()]
assert controls_test.shape == (101, 188)

assert list(controls_test).index('l3thVen') == 46
assert list(controls_test).index('rTemTraGy') == 187

X_test_ctrl = controls_test.iloc[:,np.r_[46:188]]
assert X_test_ctrl[X_test_ctrl.isnull().any(axis=1)].shape == (0, 142)
assert X_test_ctrl.duplicated().sum() == 0
X_test_ctrl = np.array(X_test_ctrl)
assert X_test_ctrl.shape == (101, 142)
age_test_ctrl = np.array(controls_test.age)
sex_test_ctrl = np.array(controls_test.sex)
site_test_ctrl = np.array(controls_test.site)
assert age_test_ctrl.shape == sex_test_ctrl.shape == site_test_ctrl.shape == (101,)

# centering by site
X_test_ctrl_bySite = np.zeros(X_test_ctrl.shape)
for s in set(site_test_ctrl):
    m = site_test_ctrl == s
    X_test_ctrl_bySite[m] = X_test_ctrl[m] - X_test_ctrl[m, :].mean(axis=0)


# TEST UHR DATA (ICAAR-START)

pheno_uhr_test = pheno[(pheno.diagnosis.isin(['UHR-C','UHR-NC'])) & (pheno.irm == 'M0')]
assert pheno_uhr_test.shape == (80, 46)
uhr_test = pd.merge(pheno_uhr_test, ROI, on='participant_id', how='left')
assert uhr_test.shape == (80, 188)
uhr_test = uhr_test[uhr_test.rTemTraGy.notna() & uhr_test.age.notna() & uhr_test.sex.notna() & uhr_test.site.notna()]
assert uhr_test.shape == (80, 188)

assert list(uhr_test).index('l3thVen') == 46
assert list(uhr_test).index('rTemTraGy') == 187

X_test_uhr = uhr_test.iloc[:,np.r_[46:188]]
assert X_test_uhr[X_test_uhr.isnull().any(axis=1)].shape == (0, 142)
assert X_test_uhr.duplicated().sum() == 0
X_test_uhr = np.array(X_test_uhr)
assert X_test_uhr.shape == (80, 142)
age_test_uhr = np.array(uhr_test.age)
sex_test_uhr = np.array(uhr_test.sex)
site_test_uhr = np.array(uhr_test.site)
assert age_test_uhr.shape == sex_test_uhr.shape == site_test_uhr.shape == (80,)

X_test_uhr_bySite = np.zeros(X_test_uhr.shape)
for s in set(site_test_uhr):
    m = site_test_uhr == s
    X_test_uhr_bySite[m] = X_test_uhr[m] - X_test_uhr[m, :].mean(axis=0)


# DUMMIES FOR SITE

site_train_ctrl_DUM = pd.get_dummies(site_train_ctrl)
assert site_train_ctrl_DUM.shape == (313, 10)

site_test_ctrl_DUM = pd.get_dummies(site_test_ctrl)
assert site_test_ctrl_DUM.shape == (101, 7)
# sandiego is a site missing from BIOBD as all subjects from its site are > 30 yo
site_test_uhr_DUM = pd.get_dummies(site_test_uhr)
assert site_test_uhr_DUM.shape == (80, 2)

r = (313, 9)
X_train_ctrl_DUM = np.concatenate((X_train_ctrl, site_train_ctrl_DUM, np.zeros(r)), axis=1)
assert X_train_ctrl_DUM.shape == (313, 161)
s = (101, 10)
t = (101, 2)
X_test_ctrl_DUM = np.concatenate((X_test_ctrl, np.zeros(s), site_test_ctrl_DUM, np.zeros(t)), axis=1)
assert X_test_ctrl_DUM.shape == (101, 161)
u = (80, 17)
X_test_uhr_DUM = np.concatenate((X_test_uhr, np.zeros(u), site_test_uhr_DUM), axis=1)
assert X_test_uhr_DUM.shape == (80, 161)

# et en centrant par site:

r = (313, 9)
X_train_ctrl_bySite_DUM = np.concatenate((X_train_ctrl_bySite, site_train_ctrl_DUM, np.zeros(r)), axis=1)
assert X_train_ctrl_bySite_DUM.shape == (313, 161)
s = (101, 10)
t = (101, 2)
X_test_ctrl_bySite_DUM = np.concatenate((X_test_ctrl_bySite, np.zeros(s), site_test_ctrl_DUM, np.zeros(t)), axis=1)
assert X_test_ctrl_bySite_DUM.shape == (101, 161)
u = (80, 17)
X_test_uhr_bySite_DUM = np.concatenate((X_test_uhr_bySite, np.zeros(u), site_test_uhr_DUM), axis=1)
assert X_test_uhr_bySite_DUM.shape == (80, 161)




"""
ANALYSIS

on a, au total:

sans les sites en dummies dans la matrice X initiale
    X_train_ctrl # non centré par site
    X_train_ctrl_bySite # centré par site
    X_test_ctrl 
    X_test_ctrl_bySite 
    X_test_uhr
    X_test_uhr_bySite

avec les sites en dummies dans la matrice X initiale (n, 161)
    X_train_ctrl_DUM
    X_train_ctrl_bySite_DUM
    X_test_ctrl_DUM
    X_test_ctrl_bySite_DUM 
    X_test_uhr_DUM
    X_test_uhr_bySite_DUM

et age, sex, site pour chaque _train et _test
"""


###############################################################################

###############################################################################

lr = lm.LinearRegression()

##### LOGO #####

#X_tr = X_train_ctrl
#X_te = X_test_ctrl
#X_uhr = X_test_uhr
X_tr = X_train_ctrl_bySite
X_te = X_test_ctrl_bySite
X_uhr = X_test_uhr_bySite
#X_tr = X_train_ctrl_DUM
#X_te = X_test_ctrl_DUM
#X_uhr = X_test_uhr_DUM
#X_tr = X_train_ctrl_bySite_DUM
#X_te = X_test_ctrl_bySite_DUM
#X_uhr = X_test_uhr_bySite_DUM


Sex_train_ctrl = pd.get_dummies(sex_train_ctrl)
assert Sex_train_ctrl.shape == (313, 2)
Residuals_train = np.array([X_tr[:, j] - lr.fit(Sex_train_ctrl, X_tr[:, j]).predict(Sex_train_ctrl) for j in range(X_tr.shape[1])]).T
assert Residuals_train.shape == (313, 142) # when site is not as a dummy in X
#assert Residuals_train.shape == (313, 161) # when site as dummy in X

X = Residuals_train
y = age_train_ctrl
groups = site_train_ctrl
logo = LeaveOneGroupOut()
assert logo.get_n_splits(X, y, groups) == 10

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

r2_train, r2_test = list(), list()

for train, test in logo.split(X, y, groups):
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    r2_train.append(metrics.r2_score(y[train], model.predict(X_train)))
    r2_test.append(metrics.r2_score(y[test], model.predict(X_test)))

print("Train r2:%.2f" % np.mean(r2_train))
print("Test r2:%.2f" % np.mean(r2_test))
print(model.best_params_)
#Train r2:0.43
#Test r2:-0.50
#{'alpha': 100.0}
#Train r2:0.23
#Test r2:-0.63
#{'alpha': 1000.0}
#Train r2:0.42
#Test r2:-0.56
#{'alpha': 100.0}
#Train r2:0.39
#Test r2:-0.67
#{'alpha': 100.0}


# validation on other cohort (BIOBD) also CbS but not accounting for site (no dummies for site)

# for controls (BIOBD) target set (CbS)
Sex_test_ctrl = pd.get_dummies(sex_test_ctrl)
assert Sex_test_ctrl.shape == (101, 2)
Residuals_test = np.array([X_te[:, j] - lr.fit(Sex_test_ctrl, X_te[:, j]).predict(Sex_test_ctrl) for j in range(X_te.shape[1])]).T
assert Residuals_test.shape == (101, 142)
#assert Residuals_test.shape == (101, 161)

# for UHR targets (CbS)
Sex_test_uhr = pd.get_dummies(sex_test_uhr)
assert Sex_test_uhr.shape == (80, 2)
Residuals_uhr= np.array([X_uhr[:, j] - lr.fit(Sex_test_uhr, X_uhr[:, j]).predict(Sex_test_uhr) for j in range(X_uhr.shape[1])]).T
assert Residuals_uhr.shape == (80, 142) 
#assert Residuals_test.shape == (101, 161)

# Fit the whole model on the control population

# check alpha is as in model !
Residuals_train_s = scaler.fit_transform(Residuals_train)
model = lm.Ridge(alpha = 1000, random_state = 42)
model.fit(Residuals_train_s, age_train_ctrl)
age_train_ctrl_predicted = model.predict(Residuals_train_s)
print("Train r2:%.2f" % metrics.r2_score(age_train_ctrl, age_train_ctrl_predicted))
#Train r2:0.63
#Train r2:0.31
#Train r2:0.65
#Train r2:0.66

# Apply the model to the control BIOBD population

Residuals_test_s = scaler.transform(Residuals_test)
age_test_ctrl_predicted = model.predict(Residuals_test_s)
print("Test r2:%.2f" % metrics.r2_score(age_test_ctrl, age_test_ctrl_predicted))
#Test r2:0.09
#Test r2:0.13
#Test r2:0.08

# Apply the model to the uhr population

Residuals_uhr_s = scaler.transform(Residuals_uhr)
age_test_uhr_predicted = model.predict(Residuals_uhr_s)
print("Test r2:%.2f" % metrics.r2_score(age_test_uhr, age_test_uhr_predicted)) 
#Test r2:-0.48
#Test r2:-0.34
#Test r2:-0.44
#Test r2:-0.46




##### LOGO + Dummies for Site #####

#X_tr = X_train_ctrl
#X_te = X_test_ctrl
#X_uhr = X_test_uhr
X_tr = X_train_ctrl_bySite
X_te = X_test_ctrl_bySite
X_uhr = X_test_uhr_bySite
#X_tr = X_train_ctrl_DUM
#X_te = X_test_ctrl_DUM
#X_uhr = X_test_uhr_DUM
#X_tr = X_train_ctrl_bySite_DUM
#X_te = X_test_ctrl_bySite_DUM
#X_uhr = X_test_uhr_bySite_DUM

SiteSex_train_ctrl = pd.get_dummies(site_train_ctrl)
Sex_train_ctrl = pd.get_dummies(sex_train_ctrl)
SiteSex_train_ctrl['M'] = Sex_train_ctrl[0.0]
SiteSex_train_ctrl['F'] = Sex_train_ctrl[1.0]
assert SiteSex_train_ctrl.shape == (313, 12)
Residuals_train = np.array([X_tr[:, j] - lr.fit(SiteSex_train_ctrl, X_tr[:, j]).predict(SiteSex_train_ctrl) for j in range(X_tr.shape[1])]).T
assert Residuals_train.shape == (313, 142)
#assert Residuals_train.shape == (313, 161)

X = Residuals_train
y = age_train_ctrl
groups = site_train_ctrl
logo = LeaveOneGroupOut()
assert logo.get_n_splits(X, y, groups) == 10

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

r2_train, r2_test = list(), list()

for train, test in logo.split(X, y, groups):
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    r2_train.append(metrics.r2_score(y[train], model.predict(X_train)))
    r2_test.append(metrics.r2_score(y[test], model.predict(X_test)))

print("Train r2:%.2f" % np.mean(r2_train))
print("Test r2:%.2f" % np.mean(r2_test))
print(model.best_params_)
#Train r2:0.23
#Test r2:-0.62
#{'alpha': 1000.0}
#Train r2:0.23
#Test r2:-0.62
#{'alpha': 1000.0}
#Train r2:0.25
#Test r2:-0.63
#{'alpha': 100.0}
#Train r2:0.25
#Test r2:-0.63
#{'alpha': 100.0}

# validation on other cohort (BIOBD) also CbS and dummies for site

# for controls (BIOBD) target set
SiteSex_test_ctrl = pd.get_dummies(site_test_ctrl)
Sex_test_ctrl = pd.get_dummies(sex_test_ctrl)
SiteSex_test_ctrl['M'] = Sex_test_ctrl[0.0]
SiteSex_test_ctrl['F'] = Sex_test_ctrl[1.0]
assert SiteSex_test_ctrl.shape == (101, 9)
Residuals_test = np.array([X_te[:, j] - lr.fit(Sex_test_ctrl, X_te[:, j]).predict(Sex_test_ctrl) for j in range(X_te.shape[1])]).T
assert Residuals_test.shape == (101, 142)
#assert Residuals_test.shape == (101, 161)

## for UHR targets
SiteSex_test_uhr = pd.get_dummies(site_test_uhr)
Sex_test_uhr = pd.get_dummies(sex_test_uhr)
SiteSex_test_uhr['M'] = Sex_test_uhr[0.0]
SiteSex_test_uhr['F'] = Sex_test_uhr[1.0]
assert SiteSex_test_uhr.shape == (80, 4)
Residuals_uhr = np.array([X_uhr[:, j] - lr.fit(SiteSex_test_uhr, X_uhr[:, j]).predict(SiteSex_test_uhr) for j in range(X_uhr.shape[1])]).T
assert Residuals_uhr.shape == (80, 142) 
#assert Residuals_uhr.shape == (80, 161) 

# Fit the whole model on the control population

Residuals_train_s = scaler.fit_transform(Residuals_train)
model = lm.Ridge(alpha = 1000, random_state = 42)
model.fit(Residuals_train_s, age_train_ctrl)
age_train_ctrl_predicted = model.predict(Residuals_train_s)
print("Train r2:%.2f" % metrics.r2_score(age_train_ctrl, age_train_ctrl_predicted))
#Train r2:0.31
#Train r2:0.31
#Train r2:0.64
#Train r2:0.64

# Apply the model to the control BIOBD population

Residuals_test_s = scaler.transform(Residuals_test)
age_test_ctrl_predicted = model.predict(Residuals_test_s)
print("Test r2:%.2f" % metrics.r2_score(age_test_ctrl, age_test_ctrl_predicted))
#Test r2:0.15
#Test r2:0.13
#Test r2:0.04
#Test r2:0.06

# Apply the model to the uhr population

Residuals_uhr_s = scaler.transform(Residuals_uhr)
age_test_uhr_predicted = model.predict(Residuals_uhr_s)
print("Test r2:%.2f" % metrics.r2_score(age_test_uhr, age_test_uhr_predicted))
#Test r2:-0.34
#Test r2:-0.34
#Test r2:-0.46
#Test r2:-0.48



##### CV10 + Dummies for Site #####

#X_tr = X_train_ctrl
#X_te = X_test_ctrl
#X_uhr = X_test_uhr
X_tr = X_train_ctrl_bySite
X_te = X_test_ctrl_bySite
X_uhr = X_test_uhr_bySite
#X_tr = X_train_ctrl_DUM
#X_te = X_test_ctrl_DUM
#X_uhr = X_test_uhr_DUM
#X_tr = X_train_ctrl_bySite_DUM
#X_te = X_test_ctrl_bySite_DUM
#X_uhr = X_test_uhr_bySite_DUM


SiteSex_train_ctrl = pd.get_dummies(site_train_ctrl)
Sex_train_ctrl = pd.get_dummies(sex_train_ctrl)
SiteSex_train_ctrl['M'] = Sex_train_ctrl[0.0]
SiteSex_train_ctrl['F'] = Sex_train_ctrl[1.0]
assert SiteSex_train_ctrl.shape == (313, 12)
Residuals_train = np.array([X_tr[:, j] - lr.fit(SiteSex_train_ctrl, X_tr[:, j]).predict(SiteSex_train_ctrl) for j in range(X_tr.shape[1])]).T
assert Residuals_train.shape == (313, 142)
#assert Residuals_train.shape == (313, 161)

X = Residuals_train
y = age_train_ctrl

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

r2_train, r2_test = list(), list()

cv = KFold(n_splits=10, random_state = 42)
cv = [[train, test] for train, test in cv.split(X, y)]

for train, test in cv:
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    r2_train.append(metrics.r2_score(y[train], model.predict(X_train)))
    r2_test.append(metrics.r2_score(y[test], model.predict(X_test)))
    
print("Train r2:%.2f" % np.mean(r2_train))
print("Test r2:%.2f" % np.mean(r2_test))
print(model.best_params_)
#Train r2:0.23
#Test r2:-0.01
#{'alpha': 1000.0}
#Train r2:0.23
#Test r2:-0.01
#{'alpha': 1000.0}
#Train r2:0.24
#Test r2:0.00
#{'alpha': 1000.0}
#Train r2:0.24
#Test r2:0.00
#{'alpha': 1000.0}

# validation on other cohort (BIOBD) also dummies for site

# for controls (BIOBD) target set
SiteSex_test_ctrl = pd.get_dummies(site_test_ctrl)
Sex_test_ctrl = pd.get_dummies(sex_test_ctrl)
SiteSex_test_ctrl['M'] = Sex_test_ctrl[0.0]
SiteSex_test_ctrl['F'] = Sex_test_ctrl[1.0]
assert SiteSex_test_ctrl.shape == (101, 9)
Residuals_test = np.array([X_te[:, j] - lr.fit(Sex_test_ctrl, X_te[:, j]).predict(Sex_test_ctrl) for j in range(X_te.shape[1])]).T
assert Residuals_test.shape == (101, 142)
#assert Residuals_test.shape == (101, 161)

## for UHR targets
SiteSex_test_uhr = pd.get_dummies(site_test_uhr)
Sex_test_uhr = pd.get_dummies(sex_test_uhr)
SiteSex_test_uhr['M'] = Sex_test_uhr[0.0]
SiteSex_test_uhr['F'] = Sex_test_uhr[1.0]
assert SiteSex_test_uhr.shape == (80, 4)
Residuals_uhr = np.array([X_uhr[:, j] - lr.fit(SiteSex_test_uhr, X_uhr[:, j]).predict(SiteSex_test_uhr) for j in range(X_uhr.shape[1])]).T
assert Residuals_uhr.shape == (80, 142) 
#assert Residuals_uhr.shape == (80, 161) 

# Fit the whole model on the control population

Residuals_train_s = scaler.fit_transform(Residuals_train)
model = lm.Ridge(alpha = 1000, random_state = 42)
model.fit(Residuals_train_s, age_train_ctrl)
age_train_ctrl_predicted = model.predict(Residuals_train_s)
print("Train r2:%.2f" % metrics.r2_score(age_train_ctrl, age_train_ctrl_predicted))
#Train r2:0.31
#Train r2:0.31
#Train r2:0.41
#Train r2:0.41


# Apply the model to the control BIOBD population

Residuals_test_s = scaler.transform(Residuals_test)
age_test_ctrl_predicted = model.predict(Residuals_test_s)
print("Test r2:%.2f" % metrics.r2_score(age_test_ctrl, age_test_ctrl_predicted))
#Test r2:0.15
#Test r2:0.13
#Test r2:0.15
#Test r2:0.13


# Apply the model to the uhr population

Residuals_uhr_s = scaler.transform(Residuals_uhr)
age_test_uhr_predicted = model.predict(Residuals_uhr_s)
print("Test r2:%.2f" % metrics.r2_score(age_test_uhr, age_test_uhr_predicted))
#Test r2:-0.34
#Test r2:-0.34
#Test r2:-0.34
#Test r2:-0.34




###############################################################################
# NOT ACCOUNTING FOR SITE (CV10)
###############################################################################

# CV10

#X_tr = X_train_ctrl
#X_te = X_test_ctrl
#X_uhr = X_test_uhr
X_tr = X_train_ctrl_bySite
X_te = X_test_ctrl_bySite
X_uhr = X_test_uhr_bySite
#X_tr = X_train_ctrl_DUM
#X_te = X_test_ctrl_DUM
#X_uhr = X_test_uhr_DUM
#X_tr = X_train_ctrl_bySite_DUM
#X_te = X_test_ctrl_bySite_DUM
#X_uhr = X_test_uhr_bySite_DUM


Sex_train_ctrl = pd.get_dummies(sex_train_ctrl)
assert Sex_train_ctrl.shape == (313, 2)
Residuals_train = np.array([X_tr[:, j] - lr.fit(Sex_train_ctrl, X_tr[:, j]).predict(Sex_train_ctrl) for j in range(X_tr.shape[1])]).T
assert Residuals_train.shape == (313, 142) 
#assert Residuals_train.shape == (313, 161)

X = Residuals_train
y = age_train_ctrl

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

r2_train, r2_test = list(), list()

cv = KFold(n_splits=10, random_state = 42)
cv = [[train, test] for train, test in cv.split(X, y)]

for train, test in cv:
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    r2_train.append(metrics.r2_score(y[train], model.predict(X_train)))
    r2_test.append(metrics.r2_score(y[test], model.predict(X_test)))
    
print("Train r2:%.2f" % np.mean(r2_train))
print("Test r2:%.2f" % np.mean(r2_test))
print(model.best_params_)
#Train r2:0.45
#Test r2:0.15
#{'alpha': 100.0}
#Train r2:0.23
#Test r2:-0.01
#{'alpha': 1000.0}
#Train r2:0.46
#Test r2:0.16
#{'alpha': 100.0}
#Train r2:0.41
#Test r2:0.09
#{'alpha': 100.0}


# validation on other cohort (BIOBD) also CbS but not accounting for site (no dummies for site)

# for controls (BIOBD) target set (CbS)
Sex_test_ctrl = pd.get_dummies(sex_test_ctrl)
assert Sex_test_ctrl.shape == (101, 2)
Residuals_test = np.array([X_te[:, j] - lr.fit(Sex_test_ctrl, X_te[:, j]).predict(Sex_test_ctrl) for j in range(X_te.shape[1])]).T
assert Residuals_test.shape == (101, 142)
#assert Residuals_test.shape == (101, 161)

# for UHR targets (CbS)
Sex_test_uhr = pd.get_dummies(sex_test_uhr)
assert Sex_test_uhr.shape == (80, 2)
Residuals_uhr= np.array([X_uhr[:, j] - lr.fit(Sex_test_uhr, X_uhr[:, j]).predict(Sex_test_uhr) for j in range(X_uhr.shape[1])]).T
assert Residuals_uhr.shape == (80, 142) 
#assert Residuals_test.shape == (101, 161)

# Fit the whole model on the control population

# check alpha is as in model !
Residuals_train_s = scaler.fit_transform(Residuals_train)
model = lm.Ridge(alpha = 1000, random_state = 42)
model.fit(Residuals_train_s, age_train_ctrl)
age_train_ctrl_predicted = model.predict(Residuals_train_s)
print("Train r2:%.2f" % metrics.r2_score(age_train_ctrl, age_train_ctrl_predicted))
#Train r2:0.63
#Train r2:0.31
#Train r2:0.65
#Train r2:0.66


# Apply the model to the control BIOBD population

Residuals_test_s = scaler.transform(Residuals_test)
age_test_ctrl_predicted = model.predict(Residuals_test_s)
print("Test r2:%.2f" % metrics.r2_score(age_test_ctrl, age_test_ctrl_predicted))
#Test r2:0.09
#Test r2:0.13
#Test r2:0.08
#Test r2:0.07


# Apply the model to the uhr population

Residuals_uhr_s = scaler.transform(Residuals_uhr)
age_test_uhr_predicted = model.predict(Residuals_uhr_s)
print("Test r2:%.2f" % metrics.r2_score(age_test_uhr, age_test_uhr_predicted)) 
#Test r2:-0.48
#Test r2:-0.34
#Test r2:-0.44
#Test r2:-0.46













