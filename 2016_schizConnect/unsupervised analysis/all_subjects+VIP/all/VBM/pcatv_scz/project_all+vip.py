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
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing

WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data"

U_all = np.load(os.path.join(WD,"U_all.npy"))
U_all_scz = np.load(os.path.join(WD,"U_all_scz.npy"))
U_all_con = np.load(os.path.join(WD,"U_all_con.npy"))

y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
X_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/X.npy")
population = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")
age = population["age"].values

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/results/projection_all+vip/boxplot"
for i in range(10):
    plt.figure()
    df = pd.DataFrame()
    df["score"] = U_all[:,i]
    df["dx"] = y_all
    T,pvalue = scipy.stats.ttest_ind(U_all_con[:,i],U_all_scz[:,i])
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    ax = sns.violinplot(x="dx", y="score", hue="dx", data=df,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=2)
    plt.ylabel("Score on component %r"%i)
    plt.title(("T : %s and pvalue = %r"%(np.around(T,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%s"%((i+1))))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/results/projection_all+vip/age"
for i in range(10):
    plt.figure()
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    x0 = age[y_all==0]
    y0 = U_all[y_all==0,i]
    fit = np.polyfit(x0, y0, deg=1)
    plt.plot(x0, fit[0] * x0 + fit[1],label = "CTL",color = "blue")
    plt.scatter(x0,y0,color = "blue")
    x1 = age[y_all==1]
    y1 = U_all[y_all==1,i]
    fit1 = np.polyfit(x1, y1, deg=1)
    plt.plot(x1, fit1[0] * x1 + fit1[1],label = "SCZ",color = "r")
    plt.scatter(x1,y1,color = "r")
    plt.tight_layout()
    plt.legend()
    plt.xlabel("age")
    plt.ylabel("Score on component %s"%(i+1))
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))

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
