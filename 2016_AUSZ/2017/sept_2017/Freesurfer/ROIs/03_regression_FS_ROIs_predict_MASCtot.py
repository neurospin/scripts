
import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
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
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils


WD = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/\
ROIs_analysis/results/linear_regression'

INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/ROIs_analysis/data/X_patients.npy'
INPUT_DATA_MASC = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/ROIs_analysis/data/MASCtot_patients.npy'
INPUT_DATA_DX = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/ROIs_analysis/data/DX_patients.npy'



n_folds = 5

X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_MASC)
DX = np.load(INPUT_DATA_DX)

#Remove nan lines
X = X[np.logical_not(np.isnan(y)).ravel(),:]
DX = DX[np.logical_not(np.isnan(y))]
y = y[np.logical_not(np.isnan(y))]
assert X.shape == (80, 49)

#lr = linear_model.LinearRegression(fit_intercept=False)
lr = linear_model.LinearRegression(fit_intercept=True)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
pred = sklearn.cross_validation.cross_val_predict(lr,X[:,2:] , y, cv=n_folds)
slope, intercept, r_value, p_value, std_err = stats.linregress(y, pred)

features = np.load("/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/ROIs_analysis/data/features.npy")

#Obain coef map
lr = linear_model.LinearRegression()
lr.fit(X[:,2:],y)
beta = lr.coef_


plt.figure()
plt.grid()
plt.title("R2 = %.02f and p = %.01e" % (r_value,p_value),fontsize=12)
plt.plot(y[DX==1], pred[DX==1], 'o',label = "ASD")
plt.plot(y[DX==2], pred[DX==2], 'o',label = "SCZ-ASD")
plt.plot(y[DX==3], pred[DX==3], 'o',label = "SCZ")
plt.plot(y, intercept + slope*y, 'r',color = "black")
plt.xlabel("MAASC score")
plt.ylabel("Predicted score using ROIs volume")
plt.legend(loc = "bottom left")
plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/ROIs_analysis/results/linear_regression_all_ROIs/plot.png")
