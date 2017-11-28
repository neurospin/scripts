#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:05:57 2017

@author: ad247405
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from brainomics import plot_utilities
import parsimony.utils.check_arrays as check_arrays
import scipy.stats


X= np.load("/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/X.npy")
y=np.load("/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/y.npy").ravel()
X_patients = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/X_patients.npy")
comp = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/model_selectionCV/all/all/struct_pca_0.1_0.5_0.1/components.npz")['arr_0']
projections = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/model_selectionCV/all/all/struct_pca_0.1_0.5_0.1/X_test_transform.npz")['arr_0']

U_controls, d = transform(V=comp, X=X[y==0], n_components=comp.shape[1], in_place=False)
U_patients, d = transform(V=comp, X=X[y==1], n_components=comp.shape[1], in_place=False)
U_all, d = transform(V=comp, X=X, n_components=comp.shape[1], in_place=False)

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


BASE_PATH = "/neurospin/brainomics/2014_pca_struct/adni"
INPUT_CSV = os.path.join(BASE_PATH,"/neurospin/brainomics/2014_pca_struct/adni/population_with_converters_time.csv")
# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop = pop[pop["DX.num"]==1]
pop["PC1"] = projections[:,0]
pop["PC2"] = projections[:,1]
pop["PC3"] = projections[:,2]
pop["y"] = pop["DX.num"]
age = (pop["Age"]).values

INPUT_CSV = os.path.join(BASE_PATH,"/neurospin/brainomics/2014_pca_struct/adni/population.csv")
pop = pd.read_csv(INPUT_CSV)
pop = pop[pop["DX.num"]==1]
mmse = pop["MMSE Total Score.sc"].values
mmse12 = pop["MMSE Total Score.m12"].values
mmse24 = pop["MMSE Total Score.m24"].values
adas = pop["ADAS11.sc"].values


INPUT_CSV = os.path.join(BASE_PATH,"/neurospin/brainomics/2014_pca_struct/adni/population.csv")
pop = pd.read_csv(INPUT_CSV)
mmse = pop["MMSE Total Score.sc"].values
mmse12 = pop["MMSE Total Score.m12"].values
mmse24 = pop["MMSE Total Score.m24"].values
adas = pop["ADAS11.sc"].values


pearsonr(projections[:,0],adas)
pearsonr(projections[:,1],adas)
pearsonr(projections[:,2],adas)

pearsonr(projections[:,0],mmse)
pearsonr(projections[:,1],mmse)
pearsonr(projections[:,2],mmse)


pearsonr(U_all[:,0],mmse)
pearsonr(U_all[:,1],mmse)
pearsonr(U_all[:,2],mmse)

pearsonr(U_all[:,0],adas)
pearsonr(U_all[:,1],adas)
pearsonr(U_all[:,2],adas)

#pearsonr(U_patients[:,0],mmse)
#pearsonr(U_patients[:,1],mmse)
#pearsonr(U_patients[:,2],mmse)

plt.plot(mmse[y==0],U_all[y==0,0],'o',color='g',label = "controls")
plt.plot(mmse[y==1],U_all[y==1,0],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 1")
plt.xlabel("MMSE score")
plt.title("corr: 0.28, p = 2.3e-08")
plt.legend()

plt.plot(mmse[y==0],U_all[y==0,1],'o',color='g',label = "controls")
plt.plot(mmse[y==1],U_all[y==1,1],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 2")
plt.xlabel("MMSE score")
plt.title("corr: -0.21, p = 4.9e-05")
plt.legend()

plt.plot(mmse[y==0],U_all[y==0,2],'o',color='g',label = "controls")
plt.plot(mmse[y==1],U_all[y==1,2],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 3")
plt.xlabel("MMSE score")
plt.title("corr: 0.28, p = 6.7e-08")
plt.legend()

plt.plot(adas[y==0],U_all[y==0,0],'o',color='g',label = "controls")
plt.plot(adas[y==1],U_all[y==1,0],'o',color='r',label = "MCI patients")
plt.ylabel("Score Component 1")
plt.xlabel("ADAS score")
plt.title("Pearson r: - 0.34, p = 4.2e-11")
plt.legend()
plt.savefig("/neurospin/brainomics/2016_pca_struct/adni/clinical_analysis/comp1_correlation.pdf")

#Error on map in PCATV paper, coulors are inverted, so I inverted the sign here as well for the
#correlation to make sens
plt.plot(adas[y==0],-U_all[y==0,1],'o',color='g',label = "controls")
plt.plot(adas[y==1],-U_all[y==1,1],'o',color='r',label = "MCI patients")
plt.ylabel("Score Component 2")
plt.xlabel("ADAS score")
plt.title("Pearson r: - 0.26, p = 3.6e-07")
plt.legend()
plt.savefig("/neurospin/brainomics/2016_pca_struct/adni/clinical_analysis/comp2_correlation.pdf")

plt.plot(adas[y==0],U_all[y==0,2],'o',color='g',label = "controls")
plt.plot(adas[y==1],U_all[y==1,2],'o',color='r',label = "MCI patients")
plt.ylabel("Score Component 3")
plt.xlabel("ADAS score")
plt.title("Pearson r: - 0.35, p = 4.5e-12")
plt.legend()
plt.savefig("/neurospin/brainomics/2016_pca_struct/adni/clinical_analysis/comp3_correlation.pdf")

###############################

import scipy
from scipy import ndimage
comp = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/model_selectionCV/all/all/struct_pca_0.1_0.5_0.1/components.npz")['arr_0']
thresh_norm_ratio = 0.99
map_arr, thres = array_utils.arr_threshold_from_norm2_ratio(comp[:,1], thresh_norm_ratio)

clust_pos = map_arr
clust_pos[map_arr<0.1] = 0

clust_neg = map_arr
clust_neg[map_arr>-0.001] = 0

U_all, d = transform(V=clust_pos.reshape(317379,1), X=X, n_components=1, in_place=False)

plt.plot(adas[y==0],-U_all[y==0],'o',color='g',label = "controls")
plt.plot(adas[y==1],-U_all[y==1],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 2, precentral part")
plt.xlabel("ADAS score")
plt.title("corr: 0.13, p = 0.01")
plt.legend()
# -1 sign because of color error in pcatv paper
pearsonr(-U_all.ravel(),adas)


U_all, d = transform(V=clust_neg.reshape(317379,1), X=X, n_components=1, in_place=False)

plt.plot(adas[y==0],-U_all[y==0],'o',color='g',label = "controls")
plt.plot(adas[y==1],-U_all[y==1],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 2,temporal pole")
plt.xlabel("ADAS score")
plt.title("corr: -0.48, p = 4e-22")
plt.legend()
pearsonr(-U_all.ravel(),adas)

##################################################################################"

import scipy
from scipy import ndimage
comp = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/model_selectionCV/all/all/struct_pca_0.1_0.5_0.1/components.npz")['arr_0']
thresh_norm_ratio = 0.99
map_arr, thres = array_utils.arr_threshold_from_norm2_ratio(comp[:,2], thresh_norm_ratio)

clust_pos = map_arr
clust_pos[map_arr<0.001] = 0

clust_neg = map_arr
clust_neg[map_arr>-0.001] = 0

U_all, d = transform(V=clust_pos.reshape(317379,1), X=X, n_components=1, in_place=False)

plt.plot(adas[y==0],U_all[y==0],'o',color='g',label = "controls")
plt.plot(adas[y==1],U_all[y==1],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 2, precentral part")
plt.xlabel("ADAS score")
plt.title("corr: -0.37, p = 1.9e-13")
plt.legend()

pearsonr(U_all.ravel(),adas)


U_all, d = transform(V=clust_neg.reshape(317379,1), X=X, n_components=1, in_place=False)

plt.plot(adas[y==0],-U_all[y==0],'o',color='g',label = "controls")
plt.plot(adas[y==1],-U_all[y==1],'o',color='r',label = "MCI patients")
plt.ylabel("Score Comp 2,temporal pole")
plt.xlabel("ADAS score")
plt.title("corr: 0.14, p = 5.6e-3")
plt.legend()
pearsonr(U_all.ravel(),adas)


tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = clust_pos[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_clust_pos_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = clust_pos[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_clust_pos_right.gii"), data=tex)

tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = clust_neg[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_clust_neg_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = clust_neg[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_clust_neg_right.gii"), data=tex)

tex = np.zeros(mask_left__left_mesh.shape)
tex[mask_left__left_mesh] = map_arr[mask_left__beta]
print ("left", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_map_arr_left.gii"), data=tex)
# right
tex = np.zeros(mask_right__right_mesh.shape)
tex[mask_right__right_mesh] = map_arr[mask_right__beta]
print ("right", np.sum(tex != 0), tex.max(), tex.min())
mesh_utils.save_texture(filename=os.path.join(OUTPUT, "tex_map_arr_right.gii"), data=tex)
