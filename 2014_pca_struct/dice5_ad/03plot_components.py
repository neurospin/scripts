# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:29:58 2016

@author: ad247405
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from brainomics import plot_utilities
from parsimony.utils import plot_map2d



#FIGURES OF  PAPER 


#################################################################################################################################################
#reconstruction with Standard PCA
components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_0.1_1e-6/data_100_100_0/results/0/pca_0.0_0.0_0.0/components.npz")
components=components['arr_0']
for k in range(3):
    fig=plot_map2d(components[:,k].reshape(100,100))
  


#reconstruction with Sparse PCA
components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_0.1_1e-6/data_100_100_0/results/0/sparse_pca_0.0_0.0_1.0/components.npz")
components=components['arr_0']
for k in range(3):
    fig=plots.map2d(components[:,k].reshape(100,100))
  


#reconstruction with ElasticNet PCA
components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_0.1_1e-6/data_100_100_0/results/0/struct_pca_0.01_1e-05_0.5/components.npz")
components=components['arr_0']
for k in range(3):
    fig=plots.map2d(components[:,k].reshape(100,100))
  

#Reconstruction with PCA-TV
components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_0.1_1e-6/data_100_100_0/results/0/struct_pca_0.01_0.5_0.5/components.npz")
components=components['arr_0']
for k in range(3):
    fig=plots.map2d(components[:,k].reshape(100,100))
  

##Ground truth
INPUT_RESULTS_DIR= os.path.join("/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_0.1_1e-6/data_100_100_0")
#Load masks of Betas star
mask=np.zeros((100,100,3))
mask[:,:,0]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_0.npy")).reshape(100,100)
mask[:,:,1]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_1.npy")).reshape(100,100)
mask[:,:,2]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_2.npy")).reshape(100,100)


true=np.zeros((100,100,3))  
for k in range(0,3):
    beta3d= np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/data_0.1/data_100_100_0/beta3d.npy")
    beta3d=beta3d.reshape(100,100)
    beta3d[mask[:,:,k]==False]=0
    true[:,:,k]=beta3d
    true[:,:,k]=true[:,:,k]-true[:,:,k].mean()
    true[:,:,k]=true[:,:,k]/true[:,:,k].std()
    fig=plot_map2d(true[:,:,k])


