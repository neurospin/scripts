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

BASE_PATH = '/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/3mm'
INPUT_CSV= os.path.join(BASE_PATH,"population.csv")
DATA_PATH = "/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/3mm/data"

X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
ados_tot = np.load(os.path.join(DATA_PATH,"ados_tot.npy"))
ados_sa = np.load(os.path.join(DATA_PATH,"ados_sa.npy"))
ados_rrb = np.load(os.path.join(DATA_PATH,"ados_rrb.npy"))
srs_t = np.load(os.path.join(DATA_PATH,"srs_t.npy"))
srs_self_t = np.load(os.path.join(DATA_PATH,"srs_self_t.npy"))

pop = pd.read_csv(INPUT_CSV)
age = pop["age"].values
categories = pop["schedule_enrol"].values

# ADOS TOTAL SCORE PREDICTION
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
ados_tot = np.load(os.path.join(DATA_PATH,"ados_tot.npy"))

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
plt.title("R = %f and  p = %f"%(float(r_value),float(p_value)))
plt.show()
###############################################################################


#ADOS sore prediction on children only
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))
ados_tot = np.load(os.path.join(DATA_PATH,"ados_tot.npy"))

X = X[categories==3,:]
y = y[categories==3]
ados_tot = ados_tot[categories==3]

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