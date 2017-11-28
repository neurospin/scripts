#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:42:00 2017

@author: ad247405
"""


import os
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

BASE_PATH = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start"
DATA_PATH = os.path.join(BASE_PATH,"5cv")
INPUT_CSV = os.path.join(BASE_PATH,"linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start_dcv.xlsx")

INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/data_ASD_SCZ_only/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/data_ASD_SCZ_only/MASCtot.npy'
INPUT_DATA_DX = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/data_ASD_SCZ_only/y.npy'

X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)
DX = np.load(INPUT_DATA_DX)


X = X[np.logical_not(np.isnan(y)).ravel(),:]
DX = DX[np.logical_not(np.isnan(y))]
y = y[np.logical_not(np.isnan(y))]



results = pd.read_excel(INPUT_CSV)
argmax_cv = pd.read_excel(INPUT_CSV,sheetname = 2)


argmax_cv = argmax_cv[argmax_cv["key_refit"] == "Ridge_dcv"]
argmax_cv = argmax_cv[argmax_cv["key_refit"] == "enettv_dcv-reduced"]
argmax_cv = argmax_cv[argmax_cv["key_refit"] == "enet_dcv-lasso-reduced"]
argmax_cv = argmax_cv[argmax_cv["key_refit"] == "enettv_dcv-lasso-reduced"]

argmax_cv = argmax_cv.reset_index()
true = list()
pred = list()
for i in range(0,5):
    cv = "cv0" + str(i)
    best_param = argmax_cv["key"][i]
    param_path = os.path.join(DATA_PATH,cv,"refit",best_param)
    y_true = np.load(os.path.join(param_path,"y_true.npz"))["arr_0"].ravel()
    true.append(y_true)
    y_pred = np.load(os.path.join(param_path,"y_pred.npz"))["arr_0"].ravel()
    pred.append(y_pred)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    print(r_squared)
    plt.figure()
    plt.plot(y_true, y_pred, 'o', label='original data')
    plt.plot(y_true, intercept + slope*y_true, 'r', label='fitted line')



true = np.concatenate(true)
pred = np.concatenate(pred)
slope, intercept, r_value, p_value, std_err = stats.linregress(true, pred)
print(r_value)
r_squared = r2_score(true, pred)
print(r_squared)

plt.grid()
plt.plot(true[DX==1], pred[DX==1], 'o',label = "ASD")
plt.plot(true[DX==3], pred[DX==3], 'o',label = "SCZ")
plt.plot(true, intercept + slope*true, 'r',color = "black")
plt.xlabel("MASC score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.title("R2 = %.02f and p = %.01e" % (r_value,p_value),fontsize=12)

plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/enettv.png")


plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/enet.png")



plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/ridge.png")



