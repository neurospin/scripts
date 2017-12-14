#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:42:06 2017

@author: ad247405
"""

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

BASE_PATH = "/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all"
MASK_PATH = "/neurospin/brainomics/2017_memento/analysis/FS/data/mask.npy"
COMP_PATH = os.path.join(BASE_PATH,"results_3triscotte","0","struct_pca_0.01_0.5_0.1","components.npz")
INPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/population.csv"


pop_memento = pd.read_csv(INPUT_CSV)
assert pop_memento.shape == (2164, 27)
GROUP_MAP = {0: "No", 1: 'AD', 2: 'Vasc', 3:"Mixte", 4:"DFT",\
             5:"Parkinson", 6:"Corps de Lewy", 7:"Autres"}
pop_memento['DX'] = pop_memento ["etiol"].map(GROUP_MAP)

COMP_PATH = os.path.join(BASE_PATH,"results_3triscotte","0","struct_pca_0.01_0.5_0.1","components.npz")
PROJ_PATH = os.path.join(BASE_PATH,"results_3triscotte","0","struct_pca_0.01_0.5_0.1","X_test_transform.npz")

components = np.load(COMP_PATH)["arr_0"]
projections = np.load(PROJ_PATH)["arr_0"]
assert components.shape == (299879, 10)
assert projections.shape == (2164, 10)


################################################################################
#Project Memento subejcts
U_memento = np.load(PROJ_PATH)["arr_0"]

for i in range(10):
    pop_memento["comp%s"%i] = U_memento[:,i]


#Too few subjects of other demence than AD
pop_memento = pop_memento[pop_memento["DX"]!="Vasc"]
pop_memento = pop_memento[pop_memento["DX"]!="DFT"]
pop_memento = pop_memento[pop_memento["DX"]!="Corps de Lewy"]
pop_memento = pop_memento[pop_memento["DX"]!="Mixte"]
pop_memento = pop_memento[pop_memento["DX"]!="Autres"]


output = "/neurospin/brainomics/2017_memento/analysis/FS/results/PC_score_correlation/struct_pca_0.01_0.5_0.1/memento"
for i in range(10):
    T,p = scipy.stats.ttest_ind(pop_memento[pop_memento["etiol"]==0]["comp%r"%i],\
                                pop_memento[pop_memento["etiol"]==1]["comp%r"%i])
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.figure()
    ax = sns.violinplot(x="DX", y="comp%r"%i, data=pop_memento,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=1)
    plt.title("ANOVA : T = %f" %T + " and p = %e"%p)
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))




output = "/neurospin/brainomics/2017_memento/analysis/FS/results/PC_score_correlation/struct_pca_0.01_0.5_0.1/memento/mmse"
for i in range(10):
    x = pop_memento['mmssctot'][np.isnan(pop_memento['mmssctot'])==False]
    y = U_memento[:,i][np.array(np.isnan(pop_memento['mmssctot'])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    mci_status = pop_memento["DX"][np.isnan(pop_memento['mmssctot'])==False]
    plt.figure()
    plt.plot(x[mci_status=="No"],y[np.array(mci_status=="No")],'o',label = "No conversion")
    plt.plot(x[mci_status=="AD"],y[np.array(mci_status=="AD")],'o',label = "AD")
    plt.xlabel("mmssctot score")
    plt.ylabel("Score on component %r"%i)
    plt.legend(loc = "bottom left")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))
################################################################################
################################################################################
#Project ADNI subjects
INPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/data/adni/population.csv"
pop_adni  = pd.read_csv(INPUT_CSV)

X_adni = np.load("/neurospin/brainomics/2017_memento/analysis/FS/data/adni/X.npy")
y_adni = np.load("/neurospin/brainomics/2017_memento/analysis/FS/data/adni/y.npy").ravel()
assert X_adni.shape == (706, 299879)

X_memento = np.load("/neurospin/brainomics/2017_memento/analysis/FS/data/X.npy")
scaler = preprocessing.StandardScaler().fit(X_memento)
X_adni = scaler.transform(X_adni)
U_adni, d = transform(V=components , X = X_adni , n_components=components.shape[1], in_place=False)

assert U_adni.shape == (706, 10)


#Create dataframe with column of interest
for i in range(10):
    pop_adni["comp%s"%i] = U_adni[:,i]

output = "/neurospin/brainomics/2017_memento/analysis/FS/results/PC_score_correlation/struct_pca_0.01_0.5_0.1/adni"
for i in range(10):
    T,p= scipy.stats.f_oneway(pop_adni[pop_adni["DX"]=="CTL"]["comp%r"%i],pop_adni[pop_adni["DX"]=="MCInc"]["comp%r"%i],\
                         pop_adni[pop_adni["DX"]=="MCIc"]["comp%r"%i],pop_adni[pop_adni["DX"]=="AD"]["comp%r"%i] )
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.figure()
    ax = sns.violinplot(x="DX", y="comp%r"%i, data=pop_adni,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=1)
    plt.title("ANOVA : T = %f" %T + " and p = %r"%p)
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

################################################################################
#Correlate ADNI projections with clinical scores

#MMSE
pop_adni['MMSE Total Score.sc'].mean()


T,p= scipy.stats.f_oneway(pop_adni[pop_adni["DX"]=="CTL"]['MMSE Total Score.sc'],\
                          pop_adni[pop_adni["DX"]=="MCInc"]['MMSE Total Score.sc'],\
                     pop_adni[pop_adni["DX"]=="MCIc"]['MMSE Total Score.sc'],\
pop_adni[pop_adni["DX"]=="AD"]['MMSE Total Score.sc'] )

sns.set_style("whitegrid")
sns.set(font_scale=1.3)
plt.figure()
ax = sns.violinplot(x="DX", y='MMSE Total Score.sc', data=pop_adni,linewidth = 3)
plt.tight_layout()
plt.legend(loc='lower center',ncol=1)
plt.title("ADNI - ANOVA : T = %f" %T + " and p = %r"%p)
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/adni_mmse.png")

output = "/neurospin/brainomics/2017_memento/analysis/FS/results/PC_score_correlation/struct_pca_0.01_0.5_0.1/adni/mmse"
for i in range(10):
    x = pop_adni['MMSE Total Score.sc'][np.isnan(pop_adni['MMSE Total Score.sc'])==False]
    y = U_adni[:,i][np.array(np.isnan(pop_adni['MMSE Total Score.sc'])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    mci_status = pop_adni["DX"][np.isnan(pop_adni['MMSE Total Score.sc'])==False]
    plt.figure()
    plt.plot(x[mci_status=="CTL"],y[np.array(mci_status=="CTL")],'o',label = "CTL")
    plt.plot(x[mci_status== "MCInc"],y[np.array(mci_status== "MCInc")],'o',label = "MCInc")
    plt.plot(x[mci_status== "MCIc"],y[np.array(mci_status== "MCIc")],'o',label = "MCIc")
    plt.plot(x[mci_status=="AD"],y[np.array(mci_status=="AD")],'o',label = "AD")
    plt.xlabel("mmssctot score")
    plt.ylabel("Score on component %r"%i)
    plt.legend(loc = "bottom left")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

output = "/neurospin/brainomics/2017_memento/analysis/FS/results/PC_score_correlation/struct_pca_0.01_0.5_0.1/adni/adas11"
for i in range(10):
    x = pop_adni['ADAS11.sc'][np.isnan(pop_adni['ADAS11.sc'])==False]
    y = U_adni[:,i][np.array(np.isnan(pop_adni['ADAS11.sc'])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    mci_status = pop_adni["DX"][np.isnan(pop_adni['ADAS11.sc'])==False]
    plt.figure()
    plt.plot(x[mci_status=="CTL"],y[np.array(mci_status=="CTL")],'o',label = "CTL")
    plt.plot(x[mci_status== "MCInc"],y[np.array(mci_status== "MCInc")],'o',label = "MCInc")
    plt.plot(x[mci_status== "MCIc"],y[np.array(mci_status== "MCIc")],'o',label = "MCIc")
    plt.plot(x[mci_status=="AD"],y[np.array(mci_status=="AD")],'o',label = "AD")
    plt.xlabel("ADAS11.sc score")
    plt.ylabel("Score on component %r"%i)
    plt.legend(loc = "bottom left")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

################################################################################

#Effect of sex
for i in range(10):
    x = pop_adni['ADAS11.sc'][np.isnan(pop_adni['ADAS11.sc'])==False]
    y = U_adni[:,i][np.array(np.isnan(pop_adni['ADAS11.sc'])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sex = pop_adni["Sex"][np.isnan(pop_adni['ADAS11.sc'])==False]
    plt.figure()
    plt.plot(x[sex=="female"],y[np.array(sex=="female")],'o',label = "female")
    plt.plot(x[sex== "male"],y[np.array(sex== "male")],'o',label = "male")
    plt.xlabel("ADAS11.sc score")
    plt.ylabel("Score on component %r"%i)
    plt.legend(loc = "bottom left")








################################################################################
#Project SCZCO controls subjects
INPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/data/sczCo/population.csv"
pop_sczCo  = pd.read_csv(INPUT_CSV)

X_sczCo = np.load("/neurospin/brainomics/2017_memento/analysis/FS/data/sczCo/X.npy")
y_sczCo = np.load("/neurospin/brainomics/2017_memento/analysis/FS/data/sczCo/y.npy").ravel()
assert X_sczCo.shape == (314, 299879)

X_memento = np.load("/neurospin/brainomics/2017_memento/analysis/FS/data/X.npy")
scaler = preprocessing.StandardScaler().fit(X_memento)
X_sczCo = scaler.transform(X_sczCo)
U_sczCo, d = transform(V=components , X = X_sczCo , n_components=components.shape[1], in_place=False)

assert U_sczCo.shape == (314, 10)


#Create dataframe with column of interest
for i in range(10):
    pop_sczCo["comp%s"%i] = U_sczCo[:,i]

################################################################################











################################################################################
def transform(V, X, n_components, in_place=False):
    """ Project a (new) dataset onto the components.
    Return the projected data and the associated d.
    We have to recompute U and d because the argument may not have the same
    number of lines.
    The argument must have the same number of columns than the datset used
    to fit the estimator.
    """
    Xk = check_arrays(X)
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
