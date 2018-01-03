#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:08:33 2017

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
from nibabel import gifti
from sklearn.cluster import KMeans


##############################################################################
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
U_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_all.npy")
U_all_con = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_all_con.npy")
U_all_scz = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_all_scz.npy")

mod = KMeans(n_clusters=3)
mod.fit(U_all_scz[:,:])
labels_all_scz = mod.labels_

df = pd.DataFrame()
df["labels"] = np.zeros((U_all.shape[0]))
df["labels"][y_all==0] = "controls"
df["labels"][y_all==1] = labels_all_scz


for i in range(10):
    df["U%s"%i] = U_all[:,i]



for i in range(10):
    plt.figure()
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    ax = sns.violinplot(x="labels", y="U%s"%i, hue="labels", data=df,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=5)
##############################################################################


##############################################################################
y_vip = np.load("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/y.npy")
U_vip = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_vip.npy")
U_vip_scz = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_vip_scz.npy")

mod = KMeans(n_clusters=3)
mod.fit(U_vip_scz[:,:])
labels_vip_scz = mod.labels_

df = pd.DataFrame()
df["labels"] = np.zeros((U_vip.shape[0]))
df["labels"][y_vip==0] = "controls"
df["labels"][y_vip==1] = labels_vip_scz


for i in range(10):
    df["U%s"%i] = U_vip[:,i]

for i in range(10):
    plt.figure()
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    ax = sns.violinplot(x="labels", y="U%s"%i, hue="labels", data=df,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=5)
##############################################################################



##############################################################################
y_cobre = np.load("/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM/data/y.npy")
U_cobre = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_cobre.npy")
U_cobre_scz = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_cobre_scz.npy")

mod = KMeans(n_clusters=3)
mod.fit(U_cobre_scz[:,:])
labels_cobre_scz = mod.labels_

df = pd.DataFrame()
df["labels"] = np.zeros((U_cobre.shape[0]))
df["labels"][y_cobre==0] = "controls"
df["labels"][y_cobre==1] = labels_cobre_scz


for i in range(10):
    df["U%s"%i] = U_cobre[:,i]

for i in range(10):
    plt.figure()
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    ax = sns.violinplot(x="labels", y="U%s"%i, hue="labels", data=df,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=5)
##############################################################################


##############################################################################
y_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/y.npy")
U_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_nudast.npy")
U_nudast_scz = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_nudast_scz.npy")

mod = KMeans(n_clusters=3)
mod.fit(U_nudast_scz[:,:])
labels_nudast_scz = mod.labels_

df = pd.DataFrame()
df["labels"] = np.zeros((U_nudast.shape[0]))
df["labels"][y_nudast==0] = "controls"
df["labels"][y_nudast==1] = labels_nudast_scz


for i in range(10):
    df["U%s"%i] = U_nudast[:,i]

for i in range(10):
    plt.figure()
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    ax = sns.violinplot(x="labels", y="U%s"%i, hue="labels", data=df,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=5)
##############################################################################
