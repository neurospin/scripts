#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:31:21 2017

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

BASE_PATH = "/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all"
MASK_PATH = "/neurospin/brainomics/2017_memento/analysis/FS/data/mask.npy"
COMP_PATH = os.path.join(BASE_PATH,"results_3triscotte","0","struct_pca_0.01_0.5_0.1","components.npz")
INPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/population.csv"



pop = pd.read_csv(INPUT_CSV)
assert  pop.shape == (2164, 27)
GROUP_MAP = {0: "non MCI", 1: 'aMCI pur', 2: 'aMCI multidomain', 3:"naMCI pur", 4:"naMCI multidomain"}
pop['mci_name'] = pop["mci"].map(GROUP_MAP)


# Standard PCA
COMP_PATH = os.path.join(BASE_PATH,"results_3triscotte","0","struct_pca_0.01_0.5_0.1","components.npz")
PROJ_PATH = os.path.join(BASE_PATH,"results_3triscotte","0","struct_pca_0.01_0.5_0.1","X_test_transform.npz")

components = np.load(COMP_PATH)["arr_0"]
projections = np.load(PROJ_PATH)["arr_0"]
assert components.shape == (299879, 10)
assert projections.shape == (2164, 10)



#Create dataframe with column of interest
for i in range(10):
    pop["comp%s"%i] = projections[:,i]

#plot
sns.set_style("whitegrid")
sns.set(font_scale=1.3)
ax = sns.violinplot(x="mci_name", y="mmssctot", data=pop,linewidth = 3)
plt.tight_layout()
plt.legend(loc='lower center',ncol=1)
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/memento_mmse.png")

output = "/neurospin/brainomics/2017_memento/analysis/FS/results/PC_score_correlation/struct_pca_0.01_0.5_0.1/memento/mci_subgroups"
for i in range(10):
    T,p= scipy.stats.f_oneway(pop[pop["mci_name"]=="aMCI pur"]["comp%r"%i],pop[pop["mci_name"]=='naMCI pur']["comp%r"%i],\
                         pop[pop["mci_name"]=='naMCI multidomain']["comp%r"%i],\
                         pop[pop["mci_name"]=='aMCI multidomain']["comp%r"%i],pop[pop["mci_name"]=='non MCI']["comp%r"%i] )

    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.figure()
    ax = sns.violinplot(x="mci_name", y="comp%r"%i, data=pop,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=1)
    plt.title("T = %r" %T + " and p = %r"%p)
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))


for i in range(10):
    T,p = scipy.stats.ttest_ind(pop[pop["etiol"]==0]["comp%r"%i],pop[pop["etiol"]==1]["comp%r"%i])
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.figure()
    ax = sns.violinplot(x="etiol", y="comp%r"%i, data=pop,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=1)
    plt.title("T = %r" %T + " and p = %r"%p)

scipy.stats.ttest_ind(pop[pop["etiol"]==0]["comp1"],pop[pop["etiol"]==1]["comp1"])


##################################################################################

#assert pop.shape ==  (2164, 38)
##Too few subjects of other demence than AD
#pop= pop[pop["etiol"]!= 2]
#pop = pop[pop["etiol"]!= 3]
#pop = pop[pop["etiol"]!= 4]
#pop = pop[pop["etiol"]!= 5]
#pop = pop[pop["etiol"]!= 6]
#pop = pop[pop["etiol"]!= 7]
#assert pop.shape == (2128, 38)
#
#pop["DX"] ==
#
##if mmse =30 , considered as MCI stable
#assert sum(pop[pop["mmssctot"]==30]["etiol"]==0) == 427
#
#
#
#pop["dx"] =
