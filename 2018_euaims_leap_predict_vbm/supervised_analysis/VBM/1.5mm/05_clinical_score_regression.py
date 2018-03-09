#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:47:41 2018

@author: ad247405
"""

import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel as nib
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
import array_utils
import nilearn
from nilearn import plotting
from nilearn import image
import array_utils
import pandas as pd
from statsmodels.formula.api import ols

BASE_PATH = '/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm'
INPUT_CSV= os.path.join(BASE_PATH,"population.csv")
DATA_PATH = "/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/data"


pop = pd.read_csv(INPUT_CSV)
age = pop["age"].values
sex = pop["sex_num"].values

categories = pop["schedule_enrol"].values
site = pop["site"].values

X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
ados_tot = np.load(os.path.join(DATA_PATH,"ados_tot.npy"))
ados_sa = np.load(os.path.join(DATA_PATH,"ados_sa.npy"))
ados_rrb = np.load(os.path.join(DATA_PATH,"ados_rrb.npy"))
srs_t = np.load(os.path.join(DATA_PATH,"srs_t.npy"))
srs_self_t = np.load(os.path.join(DATA_PATH,"srs_self_t.npy"))


# ADOS TOTAL SCORE PREDICTION
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
#site1 = np.zeros((577,1))
#site1[site==1]=1
#site2 = np.zeros((577,1))
#site2[site==2]=1
#site3 = np.zeros((577,1))
#site3[site==3]=1
#site4 = np.zeros((577,1))
#site4[site==4]=1
#site5 = np.zeros((577,1))
#site5[site==5]=1
#site6 = np.zeros((577,1))
#site6[site==6]=1
#X = np.hstack((X,site1,site2,site3,site4,site5,site6))

y = np.load(os.path.join(DATA_PATH,"y.npy"))
ados_tot = np.load(os.path.join(DATA_PATH,"ados_tot.npy"))

df = pd.DataFrame()
df["ados"] = ados_tot
ados_nan = np.logical_not(df.isnull().values.ravel())


X = X[ados_nan,:]
y = y[ados_nan]
ados_tot = ados_tot[ados_nan]
lr = linear_model.Ridge(alpha = 0.5)

pred = sklearn.cross_validation.cross_val_predict(lr,X[:,3:] ,ados_tot, cv=10)
slope, intercept, r_value, p_value, std_err = stats.linregress(ados_tot, pred)
plt.plot(ados_tot, pred, 'o', label='original data')
plt.plot(ados_tot, intercept + slope*ados_tot, 'r', label='fitted line')
plt.xlabel(" Observed ADOS total score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.title("R = %f and  p = %f"%(float(r_value),float(p_value)))
plt.show()
###############################################################################


# ADOS TOTAL SCORE PREDICTION by site
###############################################################################
for i in range(1,7):
    X = np.load(os.path.join(DATA_PATH,"X.npy"))
    y = np.load(os.path.join(DATA_PATH,"y.npy"))
    ados_tot = np.load(os.path.join(DATA_PATH,"ados_tot.npy"))
    site = pop["site"].values


    X = X[site==i,:]
    y = y[site==i]
    ados_tot = ados_tot[site==i]

    df = pd.DataFrame()
    df["ados"] = ados_tot
    ados_nan = np.logical_not(df.isnull().values.ravel())


    X = X[ados_nan,:]
    y = y[ados_nan]
    ados_tot = ados_tot[ados_nan]
    lr = linear_model.Ridge(alpha = 0.5)

    pred = sklearn.cross_validation.cross_val_predict(lr,X ,ados_tot, cv=5)
    slope, intercept, r_value, p_value, std_err = stats.linregress(ados_tot, pred)
    plt.plot(ados_tot, pred, 'o', label='original data')
    plt.plot(ados_tot, intercept + slope*ados_tot, 'r', label='fitted line')
    plt.xlabel(" Observed ADOS total score")
    plt.ylabel("Predicted score using MRI-based features")
    plt.legend()
    plt.title(" Site %s R = %f and  p = %f"%(i,float(r_value),float(p_value)))
    plt.show()
###############################################################################



#LOSO
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
ados_tot = np.load(os.path.join(DATA_PATH,"ados_tot.npy"))
site = pop["site"].values

df = pd.DataFrame()
df["ados"] = ados_tot
ados_nan = np.logical_not(df.isnull().values.ravel())

X = X[ados_nan,:]
y = y[ados_nan]
site = site[ados_nan]
ados_tot = ados_tot[ados_nan]

NFOLDS_OUTER = 5
cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
cv_outer[0][0] = np.transpose(np.where(site != 1)).ravel()
cv_outer[0][1] = np.transpose(np.where(site == 1)).ravel()

cv_outer[1][0] = np.transpose(np.where(site != 2)).ravel()
cv_outer[1][1] = np.transpose(np.where(site == 2)).ravel()

cv_outer[2][0] = np.transpose(np.where(site != 3)).ravel()
cv_outer[2][1] = np.transpose(np.where(site == 3)).ravel()

cv_outer[3][0] = np.transpose(np.where(site != 4)).ravel()
cv_outer[3][1] = np.transpose(np.where(site == 4)).ravel()

#cv_outer[4][0] = np.transpose(np.where(site != 5)).ravel()
#cv_outer[4][1] = np.transpose(np.where(site == 5)).ravel()

cv_outer[4][0] = np.transpose(np.where(site != 6)).ravel()
cv_outer[4][1] = np.transpose(np.where(site == 6)).ravel() #



lr = linear_model.Ridge(alpha = 0.5)

pred = sklearn.cross_validation.cross_val_predict(lr,X[:,3:] ,ados_tot, cv=cv_outer)
slope, intercept, r_value, p_value, std_err = stats.linregress(ados_tot, pred)
plt.plot(ados_tot, pred, 'o', label='original data')
plt.plot(ados_tot, intercept + slope*ados_tot, 'r', label='fitted line')
plt.xlabel(" Observed ADOS total score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.title("R = %f and  p = %f"%(float(r_value),float(p_value)))
plt.show()
###############################################################################



#
#import statsmodels.api as sm
#X = np.load(os.path.join(DATA_PATH,"X.npy"))
#
#intercept = np.ones((577))
#age = pop["age"].values
#sex= pop["sex_num"].values
#site1 = np.zeros((577))
#site1[site==1]=1
#site2 = np.zeros((577))
#site2[site==2]=1
#site3 = np.zeros((577))
#site3[site==3]=1
#site4 = np.zeros((577))
#site4[site==4]=1
#site5 = np.zeros((577))
#site5[site==5]=1
#site6 = np.zeros((577))
#site6[site==6]=1
#
#cov = np.vstack((intercept,age,sex,site1,site2,site3,site4,site5,site6)).T
#
#model = sm.OLS(X,cov)
#fitted = model.fit()
#res = fitted.resid
#X = res
#









#ADOS sore prediction on children only
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
ados_tot = np.load(os.path.join(DATA_PATH,"ados_tot.npy"))
sex = pop["sex_num"].values
age = pop["age"].values

X = X[categories==1,:]
y = y[categories==1]
age = age[categories==1]
ados_tot = ados_tot[categories==1]

df = pd.DataFrame()
df["ados"] = ados_tot
ados_nan = np.logical_not(df.isnull().values.ravel())


X = X[ados_nan,:]
y = y[ados_nan]
ados_tot = ados_tot[ados_nan]
lr = linear_model.Ridge(alpha = 0.5)

pred = sklearn.cross_validation.cross_val_predict(lr,X ,ados_tot, cv=10)
slope, intercept, r_value, p_value, std_err = stats.linregress(ados_tot, pred)
plt.plot(ados_tot, pred, 'o', label='original data')
plt.plot(ados_tot, intercept + slope*ados_tot, 'r', label='fitted line')
plt.xlabel(" Observed ADOS total score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.title("R = %f and  p = %f"%(float(r_value),float(p_value)))
plt.show()
###############################################################################



#ADOS sore prediction on male only
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
ados_tot = np.load(os.path.join(DATA_PATH,"ados_tot.npy"))

X = X[sex==1,:]
y = y[sex==1]
ados_tot = ados_tot[sex==1]

df = pd.DataFrame()
df["ados"] = ados_tot
ados_nan = np.logical_not(df.isnull().values.ravel())


X = X[ados_nan,:]
y = y[ados_nan]
ados_tot = ados_tot[ados_nan]
lr = linear_model.Ridge(alpha = 0.5)

pred = sklearn.cross_validation.cross_val_predict(lr,X ,ados_tot, cv=10)
slope, intercept, r_value, p_value, std_err = stats.linregress(ados_tot, pred)
plt.plot(ados_tot, pred, 'o', label='original data')
plt.plot(ados_tot, intercept + slope*ados_tot, 'r', label='fitted line')
plt.xlabel(" Observed ADOS total score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.title("R = %f and  p = %f"%(float(r_value),float(p_value)))
plt.show()
###############################################################################






# ADOS  SOCIAL AFFECT PREDICTION
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
ados_sa = np.load(os.path.join(DATA_PATH,"ados_sa.npy"))
df = pd.DataFrame()
df["ados"] = ados_sa
ados_nan = np.logical_not(df.isnull().values.ravel())


X = X[ados_nan,:]
y = y[ados_nan]
ados_sa = ados_sa[ados_nan]
lr = linear_model.Ridge(alpha = 0.5)

pred = sklearn.cross_validation.cross_val_predict(lr,X ,ados_sa, cv=10)
slope, intercept, r_value, p_value, std_err = stats.linregress(ados_sa, pred)
plt.plot(ados_sa, pred, 'o', label='original data')
plt.plot(ados_sa, intercept + slope*ados_sa, 'r', label='fitted line')
plt.xlabel("ados_sa score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.title("R = %f and  p = %f"%(float(r_value),float(p_value)))
plt.show()

# ADOS  Restrictive and repetitive behiavior PREDICTION
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
ados_rrb = np.load(os.path.join(DATA_PATH,"ados_rrb.npy"))
df = pd.DataFrame()
df["ados"] = ados_rrb
ados_nan = np.logical_not(df.isnull().values.ravel())


X = X[ados_nan,:]
y = y[ados_nan]
ados_rrb = ados_rrb[ados_nan]
lr = linear_model.Ridge(alpha = 0.5)

pred = sklearn.cross_validation.cross_val_predict(lr,X ,ados_rrb, cv=10)
slope, intercept, r_value, p_value, std_err = stats.linregress(ados_rrb, pred)
plt.plot(ados_rrb, pred, 'o', label='original data')
plt.plot(ados_rrb, intercept + slope*ados_rrb, 'r', label='fitted line')
plt.xlabel("ados_rrb score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.title("R = %f and  p = %f"%(float(r_value),float(p_value)))
plt.show()


# SRS Total score Parent report PREDICTION
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
srs_t = np.load(os.path.join(DATA_PATH,"srs_t.npy"))
df = pd.DataFrame()
df["srs_t"] = srs_t
srs_t_nan = np.logical_not(df.isnull().values.ravel())


X = X[srs_t_nan,:]
y = y[srs_t_nan]
srs_t = srs_t[srs_t_nan]
lr = linear_model.Ridge(alpha = 0.5)

pred = sklearn.cross_validation.cross_val_predict(lr,X ,srs_t, cv=5)
slope, intercept, r_value, p_value, std_err = stats.linregress(srs_t, pred)
plt.plot(srs_t, pred, 'o', label='original data')
plt.plot(srs_t, intercept + slope*srs_t, 'r', label='fitted line')
plt.xlabel("srs_t score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.title("R = %f and  p = %f"%(float(r_value),float(p_value)))
plt.show()


# SRS Total score self report PREDICTION
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
srs_self_t = np.load(os.path.join(DATA_PATH,"srs_self_t.npy"))
df = pd.DataFrame()
df["srs_self_t"] = srs_self_t
srs_self_t_nan = np.logical_not(df.isnull().values.ravel())


X = X[srs_self_t_nan,:]
y = y[srs_self_t_nan]
srs_self_t = srs_self_t[srs_self_t_nan]
lr = linear_model.Ridge(alpha = 0.5)

pred = sklearn.cross_validation.cross_val_predict(lr,X ,srs_self_t, cv=5)
slope, intercept, r_value, p_value, std_err = stats.linregress(srs_self_t, pred)
plt.plot(srs_self_t, pred, 'o', label='original data')
plt.plot(srs_self_t, intercept + slope*srs_self_t, 'r', label='fitted line')
plt.xlabel("srs_self_t score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.title("R = %f and  p = %f"%(float(r_value),float(p_value)))
plt.show()