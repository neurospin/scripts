"""
Created on Mar 22 16:42:02 2017

@author: ad247405
"""


import os
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import pandas as pd
import numpy as np


BASE_PATH = "/neurospin/brainomics/2016_AUSZ/results/VBM/linear_regression"
DATA_PATH = os.path.join(BASE_PATH,"model_selectionCV")
INPUT_CSV = os.path.join(BASE_PATH,"results_dCV_no0.1.xlsx")

results = pd.read_excel(INPUT_CSV)
argmax_cv = pd.read_excel(INPUT_CSV,sheetname = 2)
true = list()
pred = list()
for i in range(0,5):
    cv = "cv0" + str(i)
    best_param = argmax_cv["param_key"][i]
    param_path = os.path.join(DATA_PATH,cv,"refit",best_param)
    y_true = np.load(os.path.join(param_path,"y_true.npz"))["arr_0"].ravel()
    true.append(y_true)
    y_pred = np.load(os.path.join(param_path,"y_pred.npz"))["arr_0"].ravel()
    pred.append(y_pred)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    print(r_value)
    plt.figure()
    plt.plot(y_true, y_pred, 'o', label='original data')
    plt.plot(y_true, intercept + slope*y_true, 'r', label='fitted line')

    
true = np.concatenate(true)
pred = np.concatenate(pred)
slope, intercept, r_value, p_value, std_err = stats.linregress(true, pred)

plt.plot(true, pred, 'o', label='original data')
plt.plot(true, intercept + slope*true, 'r', label='fitted line')
plt.xlabel("True")
plt.ylabel("Predicted")
plt.legend()
plt.show()
