#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:54:40 2019

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

PATH = '/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_all_data/data'

###############################################################################
#  TRAINING CONTROL DATA
###############################################################################

# load training control data (SchizConnect, VIP, Prague, BSNIP)

X_ctrl = np.load(os.path.join(PATH, "X_all_controls_SCZConn_Pra_BSNIP.npy"))
age_ctrl = np.load(os.path.join(PATH, "age_all_controls_SCZConn_Pra_BSNIP.npy"))
age_ctrl = age_ctrl.astype(np.float)
site_ctrl = np.load(os.path.join(PATH, "sites_all_controls_SCZConn_Pra_BSNIP.npy"))
sex_ctrl = np.load(os.path.join(PATH, "sex_all_controls_SCZConn_Pra_BSNIP.npy"))
assert X_ctrl.shape == (620, 142)
assert age_ctrl.shape == (620,)
assert site_ctrl.shape == (620,)
assert sex_ctrl.shape == (620,)

# centering by site
X_ctrl_bySite = np.zeros(X_ctrl.shape)
for s in set(site_ctrl):
    m = site_ctrl == s
    X_ctrl_bySite[m] = X_ctrl[m] - X_ctrl[m, :].mean(axis=0)

# selection of data <= 30 yo
age_ctrl_30 = age_ctrl[age_ctrl <= 30]
site_ctrl_30 = site_ctrl[age_ctrl <= 30]
sex_ctrl_30 = sex_ctrl[age_ctrl <= 30]
assert age_ctrl_30.shape == (313,)
assert site_ctrl_30.shape == (313,)
assert sex_ctrl_30.shape == (313,)

# Dummies for site
Site_ctrl_dum = pd.get_dummies(site_ctrl)
Site_ctrl_dum_30 = Site_ctrl_dum[age_ctrl <= 30]
assert Site_ctrl_dum_30.shape == (313, 10)

###############################################################################
# TARGET CONTROL DATA
###############################################################################

# load target control data (BIOBD)

X_ctrl_BIOBD = np.load(os.path.join(PATH, "X_controls_BIOBD.npy"))
age_ctrl_BIOBD = np.load(os.path.join(PATH, "age_controls_BIOBD.npy"))
age_ctrl_BIOBD = np.array([s.replace(',' , '.') for s in age_ctrl_BIOBD])
age_ctrl_BIOBD = age_ctrl_BIOBD.astype(np.float)
site_ctrl_BIOBD = np.load(os.path.join(PATH, "sites_controls_BIOBD.npy"))
sex_ctrl_BIOBD = np.load(os.path.join(PATH, "sex_controls_BIOBD.npy"))
assert X_ctrl_BIOBD.shape == (370, 142)
assert age_ctrl_BIOBD.shape == (370,)
assert site_ctrl_BIOBD.shape == (370,)
assert sex_ctrl_BIOBD.shape == (370,)

# centering by site
X_ctrl_BIOBD_bySite = np.zeros(X_ctrl_BIOBD.shape)
for s in set(site_ctrl_BIOBD):
    m = site_ctrl_BIOBD == s
    X_ctrl_BIOBD_bySite[m] = X_ctrl_BIOBD[m] - X_ctrl_BIOBD[m, :].mean(axis=0)

# selection of data <= 30 yo
age_ctrl_BIOBD_30 = age_ctrl_BIOBD[age_ctrl_BIOBD <= 30]
site_ctrl_BIOBD_30 = site_ctrl_BIOBD[age_ctrl_BIOBD <= 30]
sex_ctrl_BIOBD_30 = sex_ctrl_BIOBD[age_ctrl_BIOBD <= 30]
assert age_ctrl_BIOBD_30.shape == (101,)
assert site_ctrl_BIOBD_30.shape == (101,)
assert sex_ctrl_BIOBD_30.shape == (101,)

# Dummies for site
Site_ctrl_BIOBD_dum = pd.get_dummies(site_ctrl_BIOBD)
Site_ctrl_BIOBD_dum_30 = Site_ctrl_BIOBD_dum[age_ctrl_BIOBD <= 30]
assert Site_ctrl_BIOBD_dum_30.shape == (101, 8)


###############################################################################
# TARGET UHR DATA
###############################################################################

# load target UHR data (ICAAR-START)
X_ies = np.load(os.path.join(PATH, "X_icaar_eugei_start_ROIs.npy"))
y_ies_age = np.load(os.path.join(PATH, "age_icaar_eugei_start.npy"))
y_ies_status = np.load(os.path.join(PATH, "y_icaar_eugei_start_Clinical_Status.npy"))
groups_ies = np.load(os.path.join(PATH, "sites_icaar_eugei_start.npy"))
sex_ies = np.load(os.path.join(PATH, "sex_icaar_eugei_start.npy"))
assert X_ies.shape == (80, 142)
assert y_ies_age.shape == (80,)
assert y_ies_status.shape == (80,)
assert groups_ies.shape == (80,)
assert sex_ies.shape == (80,)

X_ies_bySite = np.zeros(X_ies.shape)
for s in set(groups_ies):
    m = groups_ies == s
    X_ies_bySite[m] = X_ies[m] - X_ies[m, :].mean(axis=0)
assert X_ies_bySite.shape == (80, 142)

# Dummies for site
Site_ies_dum = pd.get_dummies(groups_ies)
assert Site_ies_dum.shape == (80, 2)


###############################################################################


# Training data NOT CENTERED by site and <= 30 yo
X_ctrl_30 = X_ctrl[age_ctrl <= 30]
assert X_ctrl_30.shape == (313, 142)
X_ctrl_30 = np.concatenate((X_ctrl_30, Site_ctrl_dum_30), axis=1)
assert X_ctrl_30.shape == (313, 152)
s = (313, 10)
X_ctrl_30 = np.concatenate((X_ctrl_30, np.zeros(s)), axis=1)
assert X_ctrl_30.shape == (313, 162)

# Training data CENTERED by site and <= 30 yo
X_ctrl_30_bySite = X_ctrl_bySite[age_ctrl <= 30]
assert X_ctrl_30_bySite.shape == (313, 142)
X_ctrl_30_bySite = np.concatenate((X_ctrl_30_bySite, Site_ctrl_dum_30), axis=1)
assert X_ctrl_30_bySite.shape == (313, 152)
s = (313, 10)
X_ctrl_30_bySite = np.concatenate((X_ctrl_30_bySite, np.zeros(s)), axis=1)
assert X_ctrl_30.shape == (313, 162)

s = (101, 10)
r = (101, 2)

# Target control data NOT CENTERED by site and <= 30 yo
X_ctrl_BIOBD_30 = X_ctrl_BIOBD[age_ctrl_BIOBD <= 30]
assert X_ctrl_BIOBD_30.shape == (101, 142)
X_ctrl_BIOBD_30 = np.concatenate((X_ctrl_BIOBD_30, np.zeros(s), Site_ctrl_BIOBD_dum_30, np.zeros(r)), axis=1)
assert X_ctrl_BIOBD_30.shape == (101, 162)

# Target control data CENTERED by site and <= 30 yo
X_ctrl_BIOBD_30_bySite = X_ctrl_BIOBD_bySite[age_ctrl_BIOBD <= 30]
assert X_ctrl_BIOBD_30_bySite.shape == (101, 142)
X_ctrl_BIOBD_30_bySite = np.concatenate((X_ctrl_BIOBD_30_bySite, np.zeros(s), Site_ctrl_BIOBD_dum_30, np.zeros(r)), axis=1)
assert X_ctrl_BIOBD_30_bySite.shape == (101, 162)


s = (80, 10)
r = (80, 8)

# Target UHR data NOT CENTERED by site
X_ies = np.concatenate((X_ies, np.zeros(s), np.zeros(r), Site_ies_dum), axis=1)
assert X_ies.shape == (80, 162)

# Target UHR data CENTERED by site
X_ies_bySite = np.concatenate((X_ies_bySite, np.zeros(s), np.zeros(r), Site_ies_dum), axis=1)
assert X_ies_bySite.shape == (80, 162)



###############################################################################
# DISTRIBUTION OF BRAIN VOLUME ACCORDING TO AGE
###############################################################################

# plot distribution of brain volume according to age for training controls

ctrl = pd.read_csv(os.path.join(PATH,'norm_dataset_cat12_controls_SCHIZCONNECT_VIP_PRAGUE_BSNIP.tsv'),sep='\t')
del ctrl['Unnamed: 0']
cols = list(ctrl)
cols.remove('age')
cols.remove('sex_num')
cols.remove('site')
cols.remove('dx_num')
ctrl1 = ctrl[cols]
ctrl1['tgm'] = ctrl1.sum(axis=1)
df1 = ctrl1[['participant_id','tgm']]
df2 = ctrl[['participant_id','age']]
df = pd.merge(df1, df2, on="participant_id")

fig, ax = plt.subplots()
fig.set_size_inches(18.5,10.5)
sns.scatterplot(x='age', y='tgm', hue=None, style=None, size=None, data=df, ax=ax)
plt.title('Total grey matter atrophy across age (cross-sectional, N = 620)', fontsize=18)
plt.ylabel('Total grey matter', fontsize=16)
plt.xlabel('Age', fontsize=16)
# this justifies to look at subjects before 30 years old as different slope compared to after 30 yo


#############
## plot distribution of brain volume according to age for test controls (BIOBD)

#all_ctrl = pd.read_csv(os.path.join(PATH,'norm_dataset_cat12_all_controls_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD.tsv'),sep='\t')
#del ctrl['Unnamed: 0']
#cols = list(ctrl)
#cols.remove('age')
#cols.remove('sex_num')
#cols.remove('site')
#cols.remove('dx_num')
#ctrl1 = ctrl[cols]
#ctrl1['tgm'] = ctrl1.sum(axis=1)
#df1 = ctrl1[['participant_id','tgm']]
#df2 = ctrl[['participant_id','age']]
#df = pd.merge(df1, df2, on="participant_id")
#
#fig, ax = plt.subplots()
#fig.set_size_inches(18.5,10.5)
#sns.scatterplot(x='age', y='tgm', hue=None, style=None, size=None, data=df, ax=ax)
#plt.title('Total grey matter atrophy across age (cross-sectional, N = 620)', fontsize=18)
#plt.ylabel('Total grey matter', fontsize=16)
#plt.xlabel('Age', fontsize=16)





###############################################################################
# CENTERED BY SITE (CbS) + ACCOUNTING FOR SITE (LOGO or DUMMIES)
###############################################################################

lr = lm.LinearRegression()


##### CbS + LOGO #####

Sex_ctrl_30 = pd.get_dummies(sex_ctrl_30)
assert Sex_ctrl_30.shape == (313, 2)
Residuals_ctrl_30_bySite = np.array([X_ctrl_30_bySite[:, j] - lr.fit(Sex_ctrl_30, X_ctrl_30_bySite[:, j]).predict(Sex_ctrl_30) for j in range(X_ctrl_30_bySite.shape[1])]).T
assert Residuals_ctrl_30_bySite.shape == (313, 162)

X = Residuals_ctrl_30_bySite
y = age_ctrl_30
groups = site_ctrl_30
logo = LeaveOneGroupOut()
assert logo.get_n_splits(X, y, groups) == 10

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

y_test_pred = np.zeros(len(y))
for train, test in logo.split(X, y, groups):
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    y_test_pred[test] = model.predict(X_test_s)
    
print("Test r2:%.2f" % metrics.r2_score(y, y_test_pred)) # Test r2:-26.94
print(model.best_params_) # {'alpha': 100.0}

# validation on other cohort (BIOBD) also CbS but not accounting for site (no dummies for site)

# for controls (BIOBD) target set (CbS)
Sex_ctrl_BIOBD_30 = pd.get_dummies(sex_ctrl_BIOBD_30)
assert Sex_ctrl_BIOBD_30.shape == (101, 2)
Residuals_ctrl_BIOBD_30 = np.array([X_ctrl_BIOBD_30_bySite[:, j] - lr.fit(Sex_ctrl_BIOBD_30, X_ctrl_BIOBD_30_bySite[:, j]).predict(Sex_ctrl_BIOBD_30) for j in range(X_ctrl_BIOBD_30_bySite.shape[1])]).T
assert Residuals_ctrl_BIOBD_30.shape == (101, 162)

# for UHR targets (CbS)
Sex_ies = pd.get_dummies(sex_ies)
assert Sex_ies.shape == (80, 2)
Residuals_ies = np.array([X_ies_bySite[:, j] - lr.fit(Sex_ies, X_ies_bySite[:, j]).predict(Sex_ies) for j in range(X_ies_bySite.shape[1])]).T
assert Residuals_ies.shape == (80, 162) 

# Fit the whole model on the control population

Residuals_ctrl_30_bySite_s = scaler.fit_transform(Residuals_ctrl_30_bySite)
model = lm.Ridge(alpha = 100, random_state = 42)
model.fit(Residuals_ctrl_30_bySite_s, age_ctrl_30)
age_ctrl_30_predicted = model.predict(Residuals_ctrl_30_bySite_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_30, age_ctrl_30_predicted)) # Test r2:0.65

# Apply the model to the control BIOBD population

Residuals_ctrl_BIOBD_30_s = scaler.transform(Residuals_ctrl_BIOBD_30)
y_ctrl_BIOBD_30_age_predicted = model.predict(Residuals_ctrl_BIOBD_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_BIOBD_30, y_ctrl_BIOBD_30_age_predicted)) # Test r2:0.00

# Apply the model to the uhr population

Residuals_ies_s = scaler.transform(Residuals_ies)
y_ies_age_predicted = model.predict(Residuals_ies_s)
print("Test r2:%.2f" % metrics.r2_score(y_ies_age, y_ies_age_predicted)) # Test r2:-0.47


##### CbS + LOGO + Dummies for Site #####

SiteSex_ctrl_30 = pd.get_dummies(site_ctrl_30)
Sex_ctrl_30 = pd.get_dummies(sex_ctrl_30)
SiteSex_ctrl_30['M'] = Sex_ctrl_30[0.0]
SiteSex_ctrl_30['F'] = Sex_ctrl_30[1.0]
assert SiteSex_ctrl_30.shape == (313, 12)
Residuals_ctrl_30_bySite = np.array([X_ctrl_30_bySite[:, j] - lr.fit(SiteSex_ctrl_30, X_ctrl_30_bySite[:, j]).predict(SiteSex_ctrl_30) for j in range(X_ctrl_30_bySite.shape[1])]).T
assert Residuals_ctrl_30_bySite.shape == (313, 162)

X = Residuals_ctrl_30_bySite
y = age_ctrl_30
groups = site_ctrl_30
logo = LeaveOneGroupOut()
assert logo.get_n_splits(X, y, groups) == 10

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

y_test_pred = np.zeros(len(y))
for train, test in logo.split(X, y, groups):
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    y_test_pred[test] = model.predict(X_test_s)
    
print("Test r2:%.2f" % metrics.r2_score(y, y_test_pred)) # Test r2: -0.26
print(model.best_params_) # {'alpha': 1000.0}

# validation on other cohort (BIOBD) also CbS and dummies for site

# for controls (BIOBD) target set
SiteSex_ctrl_BIOBD_30 = pd.get_dummies(site_ctrl_BIOBD_30)
Sex_ctrl_BIOBD_30 = pd.get_dummies(sex_ctrl_BIOBD_30)
SiteSex_ctrl_BIOBD_30['M'] = Sex_ctrl_BIOBD_30[0.0]
SiteSex_ctrl_BIOBD_30['F'] = Sex_ctrl_BIOBD_30[1.0]
assert SiteSex_ctrl_BIOBD_30.shape == (101, 9)
Residuals_ctrl_BIOBD_30 = np.array([X_ctrl_BIOBD_30_bySite[:, j] - lr.fit(Sex_ctrl_BIOBD_30, X_ctrl_BIOBD_30_bySite[:, j]).predict(Sex_ctrl_BIOBD_30) for j in range(X_ctrl_BIOBD_30_bySite.shape[1])]).T
assert Residuals_ctrl_BIOBD_30.shape == (101, 162)

## for UHR targets
SiteSex_ies = pd.get_dummies(groups_ies)
Sex_ies = pd.get_dummies(sex_ies)
SiteSex_ies['M'] = Sex_ies[0.0]
SiteSex_ies['F'] = Sex_ies[1.0]
assert SiteSex_ies.shape == (80, 4)
Residuals_ies = np.array([X_ies_bySite[:, j] - lr.fit(Sex_ies, X_ies_bySite[:, j]).predict(Sex_ies) for j in range(X_ies_bySite.shape[1])]).T
assert Residuals_ies.shape == (80, 162) 

# Fit the whole model on the control population

Residuals_ctrl_30_bySite_s = scaler.fit_transform(Residuals_ctrl_30_bySite)
model = lm.Ridge(alpha = 1000, random_state = 42)
model.fit(Residuals_ctrl_30_bySite_s, age_ctrl_30)
age_ctrl_30_predicted = model.predict(Residuals_ctrl_30_bySite_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_30, age_ctrl_30_predicted)) # Test r2:0.39

# Apply the model to the control BIOBD population

Residuals_ctrl_BIOBD_30_s = scaler.transform(Residuals_ctrl_BIOBD_30)
y_ctrl_BIOBD_30_age_predicted = model.predict(Residuals_ctrl_BIOBD_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_BIOBD_30, y_ctrl_BIOBD_30_age_predicted)) # Test r2:0.08

# Apply the model to the uhr population

Residuals_ies_s = scaler.transform(Residuals_ies)
y_ies_age_predicted = model.predict(Residuals_ies_s)
print("Test r2:%.2f" % metrics.r2_score(y_ies_age, y_ies_age_predicted)) # Test r2:-0.32


##### CbS + CV10 + Dummies for Site #####

SiteSex_ctrl_30 = pd.get_dummies(site_ctrl_30)
Sex_ctrl_30 = pd.get_dummies(sex_ctrl_30)
SiteSex_ctrl_30['M'] = Sex_ctrl_30[0.0]
SiteSex_ctrl_30['F'] = Sex_ctrl_30[1.0]
assert SiteSex_ctrl_30.shape == (313, 12)
Residuals_ctrl_30_bySite = np.array([X_ctrl_30_bySite[:, j] - lr.fit(SiteSex_ctrl_30, X_ctrl_30_bySite[:, j]).predict(SiteSex_ctrl_30) for j in range(X_ctrl_30_bySite.shape[1])]).T
assert Residuals_ctrl_30_bySite.shape == (313, 162)

X = Residuals_ctrl_30_bySite
y = age_ctrl_30

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

cv = KFold(n_splits=10, random_state = 42)
cv = [[train, test] for train, test in cv.split(X, y)]

y_test_pred = np.zeros(len(y))
for train, test in cv:
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    y_test_pred[test] = model.predict(X_test_s)
    
print("Test r2:%.2f" % metrics.r2_score(y, y_test_pred)) # Test r2:0.11
print(model.best_params_) # {'alpha': 1000.0}

# validation on other cohort (BIOBD) also CbS and dummies for site

# for controls (BIOBD) target set
SiteSex_ctrl_BIOBD_30 = pd.get_dummies(site_ctrl_BIOBD_30)
Sex_ctrl_BIOBD_30 = pd.get_dummies(sex_ctrl_BIOBD_30)
SiteSex_ctrl_BIOBD_30['M'] = Sex_ctrl_BIOBD_30[0.0]
SiteSex_ctrl_BIOBD_30['F'] = Sex_ctrl_BIOBD_30[1.0]
assert SiteSex_ctrl_BIOBD_30.shape == (101, 9)
Residuals_ctrl_BIOBD_30 = np.array([X_ctrl_BIOBD_30_bySite[:, j] - lr.fit(Sex_ctrl_BIOBD_30, X_ctrl_BIOBD_30_bySite[:, j]).predict(Sex_ctrl_BIOBD_30) for j in range(X_ctrl_BIOBD_30_bySite.shape[1])]).T
assert Residuals_ctrl_BIOBD_30.shape == (101, 162)

## for UHR targets
SiteSex_ies = pd.get_dummies(groups_ies)
Sex_ies = pd.get_dummies(sex_ies)
SiteSex_ies['M'] = Sex_ies[0.0]
SiteSex_ies['F'] = Sex_ies[1.0]
assert SiteSex_ies.shape == (80, 4)
Residuals_ies = np.array([X_ies_bySite[:, j] - lr.fit(Sex_ies, X_ies_bySite[:, j]).predict(Sex_ies) for j in range(X_ies_bySite.shape[1])]).T
assert Residuals_ies.shape == (80, 162) 

# Fit the whole model on the control population

Residuals_ctrl_30_bySite_s = scaler.fit_transform(Residuals_ctrl_30_bySite)
model = lm.Ridge(alpha = 1000, random_state = 42)
model.fit(Residuals_ctrl_30_bySite_s, age_ctrl_30)
age_ctrl_30_predicted = model.predict(Residuals_ctrl_30_bySite_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_30, age_ctrl_30_predicted)) # Test r2:0.39

# Apply the model to the control BIOBD population

Residuals_ctrl_BIOBD_30_s = scaler.transform(Residuals_ctrl_BIOBD_30)
y_ctrl_BIOBD_30_age_predicted = model.predict(Residuals_ctrl_BIOBD_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_BIOBD_30, y_ctrl_BIOBD_30_age_predicted)) # Test r2:0.08

# Apply the model to the uhr population

Residuals_ies_s = scaler.transform(Residuals_ies)
y_ies_age_predicted = model.predict(Residuals_ies_s)
print("Test r2:%.2f" % metrics.r2_score(y_ies_age, y_ies_age_predicted)) # Test r2:-0.32


###############################################################################
# CENTERED BY SITE (CbS) + NOT ACCOUNTING FOR SITE (CV10)
###############################################################################

# CbS + CV10

Sex_ctrl_30 = pd.get_dummies(sex_ctrl_30)
assert Sex_ctrl_30.shape == (313, 2)
Residuals_ctrl_30_bySite = np.array([X_ctrl_30_bySite[:, j] - lr.fit(Sex_ctrl_30, X_ctrl_30_bySite[:, j]).predict(Sex_ctrl_30) for j in range(X_ctrl_30_bySite.shape[1])]).T
assert Residuals_ctrl_30_bySite.shape == (313, 162)

X = Residuals_ctrl_30_bySite
y = age_ctrl_30

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

cv = KFold(n_splits=10, random_state = 42)
cv = [[train, test] for train, test in cv.split(X, y)]

y_test_pred = np.zeros(len(y))
for train, test in cv:
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    y_test_pred[test] = model.predict(X_test_s)
    
print("Test r2:%.2f" % metrics.r2_score(y, y_test_pred)) # Test r2:0.12
print(model.best_params_) # {'alpha': 100.0}

# validation on other cohort (BIOBD) also CbS but not accounting for site (no dummies for site)

# for controls (BIOBD) target set (CbS)
Sex_ctrl_BIOBD_30 = pd.get_dummies(sex_ctrl_BIOBD_30)
assert Sex_ctrl_BIOBD_30.shape == (101, 2)
Residuals_ctrl_BIOBD_30 = np.array([X_ctrl_BIOBD_30_bySite[:, j] - lr.fit(Sex_ctrl_BIOBD_30, X_ctrl_BIOBD_30_bySite[:, j]).predict(Sex_ctrl_BIOBD_30) for j in range(X_ctrl_BIOBD_30_bySite.shape[1])]).T
assert Residuals_ctrl_BIOBD_30.shape == (101, 162)

# for UHR targets (CbS)
Sex_ies = pd.get_dummies(sex_ies)
assert Sex_ies.shape == (80, 2)
Residuals_ies = np.array([X_ies_bySite[:, j] - lr.fit(Sex_ies, X_ies_bySite[:, j]).predict(Sex_ies) for j in range(X_ies_bySite.shape[1])]).T
assert Residuals_ies.shape == (80, 162) 

# Fit the whole model on the control population

Residuals_ctrl_30_bySite_s = scaler.fit_transform(Residuals_ctrl_30_bySite)
model = lm.Ridge(alpha = 100, random_state = 42)
model.fit(Residuals_ctrl_30_bySite_s, age_ctrl_30)
age_ctrl_30_predicted = model.predict(Residuals_ctrl_30_bySite_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_30, age_ctrl_30_predicted)) # Test r2:0.65

# Apply the model to the control BIOBD population

Residuals_ctrl_BIOBD_30_s = scaler.transform(Residuals_ctrl_BIOBD_30)
y_ctrl_BIOBD_30_age_predicted = model.predict(Residuals_ctrl_BIOBD_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_BIOBD_30, y_ctrl_BIOBD_30_age_predicted)) # Test r2:0.09

# Apply the model to the uhr population

Residuals_ies_s = scaler.transform(Residuals_ies)
y_ies_age_predicted = model.predict(Residuals_ies_s)
print("Test r2:%.2f" % metrics.r2_score(y_ies_age, y_ies_age_predicted)) # Test r2:-0.47


###############################################################################
# NOT CENTERED BY SITE (nCbS) + ACCOUNTING FOR SITE (LOGO or DUMMIES)
###############################################################################

##### nCbS + LOGO #####

Sex_ctrl_30 = pd.get_dummies(sex_ctrl_30)
assert Sex_ctrl_30.shape == (313, 2)
Residuals_ctrl_30 = np.array([X_ctrl_30[:, j] - lr.fit(Sex_ctrl_30, X_ctrl_30[:, j]).predict(Sex_ctrl_30) for j in range(X_ctrl_30.shape[1])]).T
assert Residuals_ctrl_30.shape == (313, 162)

X = Residuals_ctrl_30
y = age_ctrl_30
groups = site_ctrl_30
logo = LeaveOneGroupOut()
assert logo.get_n_splits(X, y, groups) == 10

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

y_test_pred = np.zeros(len(y))
for train, test in logo.split(X, y, groups):
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    y_test_pred[test] = model.predict(X_test_s)
    
print("Test r2:%.2f" % metrics.r2_score(y, y_test_pred)) # Test r2: 0.08
print(model.best_params_) # {'alpha': 100.0}

# validation on other cohort (BIOBD) also CbS but not accounting for site (no dummies for site)

# for controls (BIOBD) target set (nCbS)
Sex_ctrl_BIOBD_30 = pd.get_dummies(sex_ctrl_BIOBD_30)
assert Sex_ctrl_BIOBD_30.shape == (101, 2)
Residuals_ctrl_BIOBD_30 = np.array([X_ctrl_BIOBD_30[:, j] - lr.fit(Sex_ctrl_BIOBD_30, X_ctrl_BIOBD_30[:, j]).predict(Sex_ctrl_BIOBD_30) for j in range(X_ctrl_BIOBD_30.shape[1])]).T
assert Residuals_ctrl_BIOBD_30.shape == (101, 162)

# for UHR targets (nCbS)
Sex_ies = pd.get_dummies(sex_ies)
assert Sex_ies.shape == (80, 2)
Residuals_ies = np.array([X_ies[:, j] - lr.fit(Sex_ies, X_ies[:, j]).predict(Sex_ies) for j in range(X_ies.shape[1])]).T
assert Residuals_ies.shape == (80, 162) 

# Fit the whole model on the control population

Residuals_ctrl_30_s = scaler.fit_transform(Residuals_ctrl_30)
model = lm.Ridge(alpha = 100, random_state = 42)
model.fit(Residuals_ctrl_30_s, age_ctrl_30)
age_ctrl_30_predicted = model.predict(Residuals_ctrl_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_30, age_ctrl_30_predicted)) # Test r2:0.63

# Apply the model to the control BIOBD population

Residuals_ctrl_BIOBD_30_s = scaler.transform(Residuals_ctrl_BIOBD_30)
y_ctrl_BIOBD_30_age_predicted = model.predict(Residuals_ctrl_BIOBD_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_BIOBD_30, y_ctrl_BIOBD_30_age_predicted)) # Test r2:0.08

# Apply the model to the uhr population

Residuals_ies_s = scaler.transform(Residuals_ies)
y_ies_age_predicted = model.predict(Residuals_ies_s)
print("Test r2:%.2f" % metrics.r2_score(y_ies_age, y_ies_age_predicted)) # Test r2:-0.44


##### nCbS + LOGO + Dummies for Site #####

SiteSex_ctrl_30 = pd.get_dummies(site_ctrl_30)
Sex_ctrl_30 = pd.get_dummies(sex_ctrl_30)
SiteSex_ctrl_30['M'] = Sex_ctrl_30[0.0]
SiteSex_ctrl_30['F'] = Sex_ctrl_30[1.0]
assert SiteSex_ctrl_30.shape == (313, 12)
Residuals_ctrl_30 = np.array([X_ctrl_30[:, j] - lr.fit(SiteSex_ctrl_30, X_ctrl_30[:, j]).predict(SiteSex_ctrl_30) for j in range(X_ctrl_30.shape[1])]).T
assert Residuals_ctrl_30.shape == (313, 162)

X = Residuals_ctrl_30
y = age_ctrl_30
groups = site_ctrl_30
logo = LeaveOneGroupOut()
assert logo.get_n_splits(X, y, groups) == 10

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

y_test_pred = np.zeros(len(y))
for train, test in logo.split(X, y, groups):
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    y_test_pred[test] = model.predict(X_test_s)
    
print("Test r2:%.2f" % metrics.r2_score(y, y_test_pred)) # Test r2: -0.24
print(model.best_params_) # {'alpha': 1000.0}

# validation on other cohort (BIOBD) also nCbS and accounting for site ( dummies for site)

# for controls (BIOBD) target set (nCbS)
SiteSex_ctrl_BIOBD_30 = pd.get_dummies(site_ctrl_BIOBD_30)
Sex_ctrl_BIOBD_30 = pd.get_dummies(sex_ctrl_BIOBD_30)
SiteSex_ctrl_BIOBD_30['M'] = Sex_ctrl_BIOBD_30[0.0]
SiteSex_ctrl_BIOBD_30['F'] = Sex_ctrl_BIOBD_30[1.0]
assert SiteSex_ctrl_BIOBD_30.shape == (101, 9)
Residuals_ctrl_BIOBD_30 = np.array([X_ctrl_BIOBD_30[:, j] - lr.fit(Sex_ctrl_BIOBD_30, X_ctrl_BIOBD_30[:, j]).predict(Sex_ctrl_BIOBD_30) for j in range(X_ctrl_BIOBD_30.shape[1])]).T
assert Residuals_ctrl_BIOBD_30.shape == (101, 162)

# for UHR targets (nCbS)
SiteSex_ies = pd.get_dummies(groups_ies)
Sex_ies = pd.get_dummies(sex_ies)
SiteSex_ies['M'] = Sex_ies[0.0]
SiteSex_ies['F'] = Sex_ies[1.0]
assert SiteSex_ies.shape == (80, 4)
Residuals_ies = np.array([X_ies[:, j] - lr.fit(Sex_ies, X_ies[:, j]).predict(Sex_ies) for j in range(X_ies.shape[1])]).T
assert Residuals_ies.shape == (80, 162) 

# Fit the whole model on the control population

Residuals_ctrl_30_s = scaler.fit_transform(Residuals_ctrl_30)
model = lm.Ridge(alpha = 1000, random_state = 42)
model.fit(Residuals_ctrl_30_s, age_ctrl_30)
age_ctrl_30_predicted = model.predict(Residuals_ctrl_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_30, age_ctrl_30_predicted)) # Test r2:0.39

# Apply the model to the control BIOBD population

Residuals_ctrl_BIOBD_30_s = scaler.transform(Residuals_ctrl_BIOBD_30)
y_ctrl_BIOBD_30_age_predicted = model.predict(Residuals_ctrl_BIOBD_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_BIOBD_30, y_ctrl_BIOBD_30_age_predicted)) # Test r2:0.14

# Apply the model to the uhr population

Residuals_ies_s = scaler.transform(Residuals_ies)
y_ies_age_predicted = model.predict(Residuals_ies_s)
print("Test r2:%.2f" % metrics.r2_score(y_ies_age, y_ies_age_predicted)) # Test r2:-0.32



##### nCbS + CV10 + Dummies for Site #####

SiteSex_ctrl_30 = pd.get_dummies(site_ctrl_30)
Sex_ctrl_30 = pd.get_dummies(sex_ctrl_30)
SiteSex_ctrl_30['M'] = Sex_ctrl_30[0.0]
SiteSex_ctrl_30['F'] = Sex_ctrl_30[1.0]
assert SiteSex_ctrl_30.shape == (313, 12)
Residuals_ctrl_30 = np.array([X_ctrl_30[:, j] - lr.fit(SiteSex_ctrl_30, X_ctrl_30[:, j]).predict(SiteSex_ctrl_30) for j in range(X_ctrl_30.shape[1])]).T
assert Residuals_ctrl_30.shape == (313, 162)

X = Residuals_ctrl_30
y = age_ctrl_30

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

cv = KFold(n_splits=10, random_state = 42)
cv = [[train, test] for train, test in cv.split(X, y)]

y_test_pred = np.zeros(len(y))
for train, test in cv:
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    y_test_pred[test] = model.predict(X_test_s)
    
print("Test r2:%.2f" % metrics.r2_score(y, y_test_pred)) # Test r2:0.12
print(model.best_params_) # {'alpha': 1000.0}

# validation on other cohort (BIOBD) also nCbS and accounting for site ( dummies for site)

# for controls (BIOBD) target set (nCbS)
SiteSex_ctrl_BIOBD_30 = pd.get_dummies(site_ctrl_BIOBD_30)
Sex_ctrl_BIOBD_30 = pd.get_dummies(sex_ctrl_BIOBD_30)
SiteSex_ctrl_BIOBD_30['M'] = Sex_ctrl_BIOBD_30[0.0]
SiteSex_ctrl_BIOBD_30['F'] = Sex_ctrl_BIOBD_30[1.0]
assert SiteSex_ctrl_BIOBD_30.shape == (101, 9)
Residuals_ctrl_BIOBD_30 = np.array([X_ctrl_BIOBD_30[:, j] - lr.fit(Sex_ctrl_BIOBD_30, X_ctrl_BIOBD_30[:, j]).predict(Sex_ctrl_BIOBD_30) for j in range(X_ctrl_BIOBD_30.shape[1])]).T
assert Residuals_ctrl_BIOBD_30.shape == (101, 162)

# for UHR targets (nCbS)
SiteSex_ies = pd.get_dummies(groups_ies)
Sex_ies = pd.get_dummies(sex_ies)
SiteSex_ies['M'] = Sex_ies[0.0]
SiteSex_ies['F'] = Sex_ies[1.0]
assert SiteSex_ies.shape == (80, 4)
Residuals_ies = np.array([X_ies[:, j] - lr.fit(Sex_ies, X_ies[:, j]).predict(Sex_ies) for j in range(X_ies.shape[1])]).T
assert Residuals_ies.shape == (80, 162) 

# Fit the whole model on the control population

Residuals_ctrl_30_s = scaler.fit_transform(Residuals_ctrl_30)
model = lm.Ridge(alpha = 1000, random_state = 42)
model.fit(Residuals_ctrl_30_s, age_ctrl_30)
age_ctrl_30_predicted = model.predict(Residuals_ctrl_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_30, age_ctrl_30_predicted)) # Test r2:0.39

# Apply the model to the control BIOBD population

Residuals_ctrl_BIOBD_30_s = scaler.transform(Residuals_ctrl_BIOBD_30)
y_ctrl_BIOBD_30_age_predicted = model.predict(Residuals_ctrl_BIOBD_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_BIOBD_30, y_ctrl_BIOBD_30_age_predicted)) # Test r2:0.14

# Apply the model to the uhr population

Residuals_ies_s = scaler.transform(Residuals_ies)
y_ies_age_predicted = model.predict(Residuals_ies_s)
print("Test r2:%.2f" % metrics.r2_score(y_ies_age, y_ies_age_predicted)) # Test r2:-0.32


###############################################################################
# NOT CENTERED BY SITE (nCbS) + NOT ACCOUNTING FOR SITE (CV10)
###############################################################################

########    nCbS + CV10   ########

Sex_ctrl_30 = pd.get_dummies(sex_ctrl_30)
assert Sex_ctrl_30.shape == (313, 2)
Residuals_ctrl_30 = np.array([X_ctrl_30[:, j] - lr.fit(Sex_ctrl_30, X_ctrl_30[:, j]).predict(Sex_ctrl_30) for j in range(X_ctrl_30.shape[1])]).T
assert Residuals_ctrl_30.shape == (313, 162)

X = Residuals_ctrl_30
y = age_ctrl_30

param_grid = {'alpha': 10. ** np.arange(-5, 5)}
model = GridSearchCV(lm.Ridge(max_iter=10000, tol = 0.0001, random_state = 42), param_grid, cv=10)
scaler = StandardScaler()

cv = KFold(n_splits=10, random_state = 42)
cv = [[train, test] for train, test in cv.split(X, y)]

y_test_pred = np.zeros(len(y))
for train, test in cv:
    X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model.fit(X_train_s, y_train)
    y_test_pred[test] = model.predict(X_test_s)
    
print("Test r2:%.2f" % metrics.r2_score(y, y_test_pred)) # Test r2: -20.87
print(model.best_params_) # {'alpha': 100.0}

# validation on other cohort (BIOBD) also nCbS and not accounting for site

# for controls (BIOBD) target set
Sex_ctrl_BIOBD_30 = pd.get_dummies(sex_ctrl_BIOBD_30)
assert Sex_ctrl_BIOBD_30.shape == (101, 2)
Residuals_ctrl_BIOBD_30 = np.array([X_ctrl_BIOBD_30[:, j] - lr.fit(Sex_ctrl_BIOBD_30, X_ctrl_BIOBD_30[:, j]).predict(Sex_ctrl_BIOBD_30) for j in range(X_ctrl_BIOBD_30.shape[1])]).T
assert Residuals_ctrl_BIOBD_30.shape == (101, 162)

# for UHR targets: regress out also the MADRS score, cannabis
Sex_ies = pd.get_dummies(sex_ies)
assert Sex_ies.shape == (80, 2)
Residuals_ies = np.array([X_ies[:, j] - lr.fit(Sex_ies, X_ies[:, j]).predict(Sex_ies) for j in range(X_ies.shape[1])]).T
assert Residuals_ies.shape == (80, 162) 

# Fit the whole model on the control population

Residuals_ctrl_30_s = scaler.fit_transform(Residuals_ctrl_30)
model = lm.Ridge(alpha = 100, random_state = 42)
model.fit(Residuals_ctrl_30_s, age_ctrl_30)
age_ctrl_30_predicted = model.predict(Residuals_ctrl_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_30, age_ctrl_30_predicted)) # Test r2:0.65

# Apply the model to the control BIOBD population

Residuals_ctrl_BIOBD_30_s = scaler.transform(Residuals_ctrl_BIOBD_30)
y_ctrl_BIOBD_30_age_predicted = model.predict(Residuals_ctrl_BIOBD_30_s)
print("Test r2:%.2f" % metrics.r2_score(age_ctrl_BIOBD_30, y_ctrl_BIOBD_30_age_predicted)) # Test r2:0.08

# Apply the model to the uhr population

Residuals_ies_s = scaler.transform(Residuals_ies)
y_ies_age_predicted = model.predict(Residuals_ies_s)
print("Test r2:%.2f" % metrics.r2_score(y_ies_age, y_ies_age_predicted)) # Test r2:-0.44





###############################################################################

# Plot the predicted age as a function of the real age

data1 = pd.DataFrame(y_test_pred)
data2 = pd.DataFrame(age_ctrl_30)
final_data = pd.concat([data1, data2], axis=1)
final_data.columns = ['predicted age','age']
lm = sns.lmplot(x='age',y='predicted age',data=final_data, fit_reg=True)
fig = lm.fig
fig.set_size_inches(18.5,10.5)
fig.suptitle("Variance explained (R²) = 38%", fontsize=18)
plt.ylabel('Predicted (Brain) Age', fontsize=16)
plt.xlabel('Real Age', fontsize=16)





# #############################################################################
# Results

brain_age_BIOBD_30 = pd.DataFrame(y_ctrl_BIOBD_30_age_predicted)
real_age_BIOBD_30 = pd.DataFrame(age_ctrl_BIOBD_30)
data_ctrl = pd.concat([real_age_BIOBD_30, brain_age_BIOBD_30], axis=1)
data_ctrl.columns = ['real age ctrl', 'predicted age ctrl']
data_ctrl['age_diff_ctrl'] = data_ctrl['predicted age ctrl'] - data_ctrl['real age ctrl']
assert data_ctrl['real age ctrl'].mean() == 24.41769844316832
assert data_ctrl['real age ctrl'].median() == 24.0

brain_age = pd.DataFrame(y_ies_age_predicted)
real_age = pd.DataFrame(y_ies_age)
status = pd.DataFrame(y_ies_status)
data_final = pd.concat([real_age, brain_age, status], axis=1)
data_final.columns = ['real age', 'predicted age', 'clinical status']
data_final['age_diff'] = data_final['predicted age'] - data_final['real age']
assert data_final['real age'].mean() == 21.5
assert data_final['real age'].median() == 21.0


age_diff_ctrl = data_ctrl.age_diff_ctrl
age_diff_C = data_final[data_final['clinical status'] == 1].age_diff
age_diff_NC = data_final[data_final['clinical status'] == 0].age_diff

age_diff_C.mean() # 3.0039052100409456
age_diff_NC.mean() # 1.1038159352218984
age_diff_ctrl.mean() # -1.1726023776949919

# rejecting outliers
    
def reject_outliers(x, mad):
    return x[abs(x - x.median()) < 3 * mad]

mad_age_diff_C = 1.4826 * np.median(np.abs(age_diff_C - age_diff_C.median()))
mad_age_diff_NC = 1.4826 * np.median(np.abs(age_diff_NC - age_diff_NC.median()))
mad_age_diff_ctrl = 1.4826 * np.median(np.abs(age_diff_ctrl - age_diff_ctrl.median()))

age_diff_C_no_out = reject_outliers(age_diff_C, mad_age_diff_C)
age_diff_NC_no_out = reject_outliers(age_diff_NC, mad_age_diff_NC)
age_diff_ctrl_no_out = reject_outliers(age_diff_ctrl, mad_age_diff_ctrl)

age_diff_C_no_out.mean() # 3.2398208035161207
age_diff_NC_no_out.mean() # 1.1038159352218984
age_diff_ctrl_no_out.mean() # -1.1726023776949919

############# BOOTSTRAPPING POUR VOIR LA VARIATION DE LA DIFFERENCE D'AGE (calcul de l'intervalle de confiance) #####
    
from scipy.stats import mannwhitneyu

# pour les contrôles
X_ctrl = age_diff_ctrl_no_out
nboot = 1000
variation_median_ctrl = []
for boot in range(nboot):
    boot = np.random.choice(X_ctrl, size=len(X_ctrl), replace=True)
    variation_median_ctrl += [np.median(boot)]

# pour les non-converteurs
X_NC = age_diff_NC_no_out
nboot = 1000
variation_median_NC = []
for boot in range(nboot):
    boot = np.random.choice(X_NC, size=len(X_NC), replace=True)
    variation_median_NC += [np.median(boot)]

# pour les converteurs
X_C = age_diff_C_no_out
nboot = 1000
variation_median_C = []
for boot in range(nboot):
    boot = np.random.choice(X_C, size=len(X_C), replace=True)
    variation_median_C += [np.median(boot)]
    
variation_median_ctrl = pd.DataFrame(variation_median_ctrl)  
variation_median_NC = pd.DataFrame(variation_median_NC)
variation_median_C = pd.DataFrame(variation_median_C)

# graphical representation

data_median = pd.concat([variation_median_ctrl, variation_median_NC, variation_median_C], axis=1)
data_median.columns = ['Controls = 101','Non-Converters = 53', 'Converters = 26'] 

import seaborn as sns
import matplotlib.pyplot as plt
from statannot.statannot import add_stat_annotation

dims = (8, 12)
fig, ax = plt.subplots(figsize=dims)
sns.set(style="whitegrid")
ax.set_title("Brain ageing of Controls, Non-Converters and Converters")
plt.ylabel('Brain age - Real Age')
ax = sns.boxplot(data=data_median)
add_stat_annotation(ax, data=data_median,
                    boxPairList=[("Controls = 101", "Non-Converters = 53"),
                                 ("Non-Converters = 53", "Converters = 26"), 
                                 ("Controls = 101", "Converters = 26")],
                    test='Mann-Whitney', textFormat='full', loc='inside', verbose=2)

############# PERMUTATION TEST (calcul de p non-paramétrique) POUR C vs NC

    
C = age_diff_C_no_out
NC = age_diff_NC_no_out
C_array = np.array(C)
NC_array = np.array(NC)
C_array = C_array[:,np.newaxis] # 25
NC_array = NC_array[:,np.newaxis] # 49

T_array = np.ones(26)
NT_array = np.zeros(53)
T_array = T_array[:,np.newaxis] # 25
NT_array = NT_array[:,np.newaxis] # 49
array_T = np.c_[C_array,T_array]
array_NT = np.c_[NC_array,NT_array]
data_array = np.r_[array_T,array_NT]
columns = ['brain_ageing','conversion']
df_brain_ageing = pd.DataFrame(data_array, columns=columns)
df_brain_ageing.conversion = df_brain_ageing.conversion.map({1.0:'yes', 0.0:'no'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = mannwhitneyu(df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'yes'],df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'no'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p
    
for i in range(1,nperm):
    brain_ageing = df_brain_ageing.brain_ageing.values
    brain_ageing = brain_ageing[:,np.newaxis]
    perm = np.random.permutation(df_brain_ageing.conversion)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[brain_ageing,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = mannwhitneyu(df_permuted.brain_ageing[df_permuted.conversion == 'yes'],df_permuted.brain_ageing[df_permuted.conversion == 'no'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.0021997800219978004 ~ 0.002

# Plot
from matplotlib.pyplot import figure
# Re-weight to obtain distribution
figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
weights = np.ones(perms.shape[0]) / perms.shape[0]
plt.hist([perms[perms >= perms[0]], perms], histtype='stepfilled', bins=100, label=["U > U obs", "U < U obs (p-value)"], weights = [weights[perms >= perms[0]], weights])
plt.xlabel("Statistic distribution under the null hypothesis ")
plt.title("Non-parametric p for the difference between non-converters and converters: \n Significant")
plt.axvline(x=perms[0], color='red', linewidth=1, label='observed statistic: p = 0.002')
_ = plt.legend(loc='upper left')

############# PERMUTATION TEST (calcul de p non-paramétrique) pour ctrl vs NC
    
C = age_diff_ctrl_no_out
NC = age_diff_NC_no_out
C_array = np.array(C)
NC_array = np.array(NC)
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]

T_array = np.ones(101)
NT_array = np.zeros(53)
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
array_T = np.c_[C_array,T_array]
array_NT = np.c_[NC_array,NT_array]
data_array = np.r_[array_T,array_NT]
columns = ['brain_ageing','conversion']
df_brain_ageing = pd.DataFrame(data_array, columns=columns)
df_brain_ageing.conversion = df_brain_ageing.conversion.map({1.0:'yes', 0.0:'no'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = mannwhitneyu(df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'yes'],df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'no'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p
    
for i in range(1,nperm):
    brain_ageing = df_brain_ageing.brain_ageing.values
    brain_ageing = brain_ageing[:,np.newaxis]
    perm = np.random.permutation(df_brain_ageing.conversion)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[brain_ageing,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = mannwhitneyu(df_permuted.brain_ageing[df_permuted.conversion == 'yes'],df_permuted.brain_ageing[df_permuted.conversion == 'no'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.00019998000199980003 ~ 0.0002

# Plot
from matplotlib.pyplot import figure
# Re-weight to obtain distribution
figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
weights = np.ones(perms.shape[0]) / perms.shape[0]
plt.hist([perms[perms >= perms[0]], perms], histtype='stepfilled', bins=100, label=["U > U obs", "U < U obs (p-value)"], weights = [weights[perms >= perms[0]], weights])
plt.xlabel("Statistic distribution under the null hypothesis ")
plt.title("Non-parametric p for the difference between controls and non-converters: \n Significant")
plt.axvline(x=perms[0], color='red', linewidth=1, label='observed statistic: p = 0.0002')
_ = plt.legend(loc='upper left')

############# PERMUTATION TEST (calcul de p non-paramétrique) pour ctrl vs C
    
C = age_diff_C_no_out
NC = age_diff_ctrl_no_out
C_array = np.array(C)
NC_array = np.array(NC)
C_array = C_array[:,np.newaxis] # 25
NC_array = NC_array[:,np.newaxis] # 49

T_array = np.ones(26)
NT_array = np.zeros(101)
T_array = T_array[:,np.newaxis] # 25
NT_array = NT_array[:,np.newaxis] # 49
array_T = np.c_[C_array,T_array]
array_NT = np.c_[NC_array,NT_array]
data_array = np.r_[array_T,array_NT]
columns = ['brain_ageing','conversion']
df_brain_ageing = pd.DataFrame(data_array, columns=columns)
df_brain_ageing.conversion = df_brain_ageing.conversion.map({1.0:'yes', 0.0:'no'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = mannwhitneyu(df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'yes'],df_brain_ageing.brain_ageing[df_brain_ageing.conversion == 'no'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p
    
for i in range(1,nperm):
    brain_ageing = df_brain_ageing.brain_ageing.values
    brain_ageing = brain_ageing[:,np.newaxis]
    perm = np.random.permutation(df_brain_ageing.conversion)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[brain_ageing,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = mannwhitneyu(df_permuted.brain_ageing[df_permuted.conversion == 'yes'],df_permuted.brain_ageing[df_permuted.conversion == 'no'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.00019998000199980003 ~ 0.0002

# Plot
from matplotlib.pyplot import figure
# Re-weight to obtain distribution
figure(num=None, figsize=(8,6), dpi=80, facecolor='w', edgecolor='k')
weights = np.ones(perms.shape[0]) / perms.shape[0]
plt.hist([perms[perms >= perms[0]], perms], histtype='stepfilled', bins=100, label=["U > U obs", "U < U obs (p-value)"], weights = [weights[perms >= perms[0]], weights])
plt.xlabel("Statistic distribution under the null hypothesis")
plt.title("Non-parametric p for the difference between controls and converters: \n Significant")
plt.axvline(x=perms[0], color='red', linewidth=1, label='observed statistic: p = 0.0002')
_ = plt.legend(loc='upper left')




################### plot distribution of brain volume according to age


ctrl = pd.read_csv(os.path.join(PATH,'norm_dataset_cat12_controls_BIOBD.tsv'),sep='\t')
del ctrl['Unnamed: 0']
ctrl.Age = [s.replace(',' , '.') for s in ctrl.Age]
ctrl.Age = ctrl.Age.astype(np.float)
ctrl = ctrl[(ctrl.Age <= 30)]
cols = list(ctrl)
cols.remove('Age')
cols.remove('sex_num')
cols.remove('siteID')
cols.remove('DX')
cols.remove('participant_id')
ctrl1 = ctrl[cols]
ctrl1['tgm'] = ctrl1.sum(axis=1)
ctrl_part = ctrl.participant_id.to_frame()
ctrl_tgm = ctrl1.tgm.to_frame()
df1 = pd.merge(ctrl_part, ctrl_tgm, right_index=True, left_index=True)
df1.columns = ['participant_id','total grey matter']
df2 = ctrl[['participant_id','Age']]
df2.columns = ['participant_id','age']
df = pd.merge(df1, df2, on="participant_id")
df['status'] = 'Controls'


lm1 = sns.lmplot(x='age',y='total grey matter',data=df, fit_reg=True, size=8)
fig = lm1.fig
fig.suptitle("Brain atrophy across age", fontsize=14)

# placer sur ce graphique les sujets converteurs (puis les non-converteurs)

UHR = pd.read_csv(os.path.join(PATH, "norm_dataset_cat12_ICAAR_EUGEI_START.tsv"), sep='\t')
UHRC = UHR[(UHR.clinical_status == 'UHR-C') & (UHR.irm == 'M0')]
UHRC.shape # (27, 161)
cols = list(UHRC)
for i in ['age',
 'sex',
 'clinical_status',
 'medication',
 'cannabis_last_month',
 'tobacco_last_month',
 'BPRS',
 'PANSS_total',
 'PANSS_positive',
 'PANSS_negative',
 'PANSS_psychopatho',
 'PANSS_desorganisation',
 'SANS',
 'SAPS',
 'MADRS',
 'SOFAS',
 'NSS',
 'irm']:
    cols.remove(i)
UHRC1 = UHRC[cols]
UHRC1['tgm'] = UHRC1.sum(axis=1)
UHRC1 = UHRC1[['participant_id','tgm']]
UHRC1.columns = ['participant_id','total grey matter']
UHRC2 = UHRC[['participant_id','age']]
UHRC = pd.merge(UHRC1, UHRC2, on="participant_id")
UHRC['status'] = 'UHR Converters'

UHR = pd.read_csv(os.path.join(PATH, "norm_dataset_cat12_ICAAR_EUGEI_START.tsv"), sep='\t')
UHRNC = UHR[(UHR.clinical_status == 'UHR-NC') & (UHR.irm == 'M0')]
UHRNC.shape # (53, 161)
cols = list(UHRNC)
for i in ['age',
 'sex',
 'clinical_status',
 'medication',
 'cannabis_last_month',
 'tobacco_last_month',
 'BPRS',
 'PANSS_total',
 'PANSS_positive',
 'PANSS_negative',
 'PANSS_psychopatho',
 'PANSS_desorganisation',
 'SANS',
 'SAPS',
 'MADRS',
 'SOFAS',
 'NSS',
 'irm']:
    cols.remove(i)
UHRNC1 = UHRNC[cols]
UHRNC1['tgm'] = UHRNC1.sum(axis=1)
UHRNC1 = UHRNC1[['participant_id','tgm']]
UHRNC1.columns = ['participant_id','total grey matter']
UHRNC2 = UHRNC[['participant_id','age']]
UHRNC = pd.merge(UHRNC1, UHRNC2, on="participant_id")
UHRNC['status'] = 'UHR Non-Converters'

final_data = pd.concat([df,UHRNC,UHRC])
final_data.columns
lm2 = sns.lmplot(x='age',y='total grey matter',hue='status',hue_order=['Controls','UHR Converters','UHR Non-Converters'],data=final_data,fit_reg=True,size=6)
fig2 = lm2.fig
fig2.suptitle("Brain atrophy across age", fontsize=14)

final_data = pd.concat([UHRNC,UHRC])
lm2 = sns.lmplot(x='age',y='total grey matter',hue='status',hue_order=['UHR Non-Converters','UHR Converters'],data=final_data,fit_reg=True,size=6)
fig2 = fig2 = lm2.fig
fig2.suptitle("Brain atrophy across age", fontsize=14)


