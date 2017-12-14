#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:38:23 2017

@author: ad247405
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import nibabel as nib
import json
from nilearn import plotting
from nilearn import image
from scipy.stats.stats import pearsonr
import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
################
# Input/Output #
################

components = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/vbm_pcatv_all+VIP_scz/results/0/struct_pca_0.1_0.1_0.1/components.npz")["arr_0"]


assert components.shape == (125959, 10)



y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
X = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/X.npy")
assert X.shape == (606, 125961)
X = X[:,2:]
U, d = transform(V=components , X = X , n_components=components.shape[1], in_place=False)
assert U.shape == (606, 10)
U_scz = U[y==1,:]
U_con = U[y==0,:]


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_all+vip/boxplots"
for i in range(10):
    plt.figure()
    df = pd.DataFrame()
    df["score"] = U[:,i]
    df["dx"] = y
    T,pvalue = scipy.stats.ttest_ind(U[y==0,i],U[y==1,i])
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    ax = sns.violinplot(x="dx", y="score", hue="dx", data=df,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=2)
    plt.ylabel("Score on component %r"%i)
    plt.title(("T : %s and pvalue = %r"%(np.around(T,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))



#X_prague = np.load("/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/VBM/data/X.npy")
#y_prague = np.load("/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/VBM/data/y.npy")
#X_prague = X_prague[:,2:]
#U_prague, d = transform(V=components,X = X_prague, n_components=components.shape[1],in_place=False)
#assert U_prague.shape == (133, 10)
#
#for i in range(10):
#    plt.figure()
#    df = pd.DataFrame()
#    df["score"] = U_prague[:,i]
#    df["dx"] = y_prague
#    T,pvalue = scipy.stats.ttest_ind(U_prague[y_prague==0,i],U_prague[y_prague==1,i])
#    sns.set_style("whitegrid")
#    sns.set(font_scale=1.3)
#    ax = sns.violinplot(x="dx", y="score", hue="dx", data=df,linewidth = 3)
#    plt.tight_layout()
#    plt.legend(loc='lower center',ncol=2)
#    plt.ylabel("Score on component %r"%i)
#    plt.title(("T : %s and pvalue = %r"%(np.around(T,decimals=3),pvalue)))

###############################################################################
def transform(V, X, n_components, in_place=False):
    """ Project a (new) dataset onto the components.
    Return the projected data and the associated d.
    We have to recompute U and d because the argument may not have the same
    number of lines.
    The argument must have the same number of columns than the datset used
    to fit the estimator.
    """
    Xk = X
    if not in_place:
        Xk = Xk.copy()
    n, p = Xk.shape
    if p != V.shape[0]:
        raise ValueError(
                    "The argument must have the same number of columns "
                    "than the datset used to fit the estimator.")
    U = np.zeros((n, n_components))
    d = np.zeros((n_components, ))
    for k in range(n_components):
        # Project on component j
        vk = V[:, k].reshape(-1, 1)
        uk = np.dot(X, vk)
        uk /= np.linalg.norm(uk)
        U[:, k] = uk[:, 0]
        dk = np.dot(uk.T, np.dot(Xk, vk))
        d[k] = dk
        # Residualize
        Xk -= dk * np.dot(uk, vk.T)
    return U, d
