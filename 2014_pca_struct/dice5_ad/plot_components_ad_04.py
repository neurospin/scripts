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


################
# Input/Output #
################

INPUT_DIR = "/neurospin/brainomics/2014_pca_struct/dice5_ad/results"
INPUT_RESULTS_FILE = os.path.join(INPUT_DIR, "consolidated_results.csv")

OUTPUT_DIR = INPUT_DIR
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "summary.csv")

##############
# Parameters #
##############

SNRS = [0.1,0.5, 1.0]
N_COMP = 3
###################
# Plot components #
###################
b=beta3d[:,:].reshape(100,100)



b=components[:,0]+components[:,1]+components[:,2]
fig=plot_map2d(b.reshape(100,100),title="# component %d"%k)
k=scipy.stats.kurtosis(b,axis=None)

  
#Snr =0.5
beta3d = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/data/data_100_100_0.5/beta3d.npy")
plot_map2d(beta3d[:,:].reshape(100,100))


#reconstruction with Standard PCA
components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/results/data_100_100_0.5/results/0/pca_0.0_0.0_0.0/components.npz")
components=components['arr_0']
for k in range(3):
    fig=plot_map2d(components[:,k].reshape(100,100),title="# component %d"%k)
  

#Reconstruction with PCA-TVn (Good parameters obtain with DIce and Frobenius )
components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/results/data_100_100_0.5/results/0/struct_pca_0.01_0.5_0.5/components.npz")
components=components['arr_0']
for k in range(3):
    fig=plot_map2d(components[:,k].reshape(100,100),title="# component %d"%k)
  
  
  
#Snr =1.0
beta3d = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/data/data_100_100_1.0/beta3d.npy")
plot_map2d(beta3d)

#reconstruction with Standard PCA

components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/results/data_100_100_1.0/results/0/pca_0.0_0.0_0.0/components.npz")
components=components['arr_0']
for k in range(3):
    fig=plot_map2d(components[:,k].reshape(100,100),title="# component %d"%k)
    
    

components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/results/data_100_100_1.0/results/0/struct_pca_0.01_0.5_0.5/components.npz")
components=components['arr_0']
for k in range(3):
    fig=plot_map2d(components[:,k].reshape(100,100),title="# component %d"%k)
    



#Snr =0.1
beta3d = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/data/data_100_100_0.1/beta3d.npy")
plot_map2d(beta3d)

#reconstruction with Standard PCA
components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/results/data_100_100_0.1/results/0/pca_0.0_0.0_0.0/components.npz")
components=components['arr_0']
for k in range(3):
    fig=plot_map2d(components[:,k].reshape(100,100),title="# component %d"%k)
  

#Reconstruction with PCA-TVn (Good parameters obtain with DIce and Frobenius )
components = np.load("/neurospin/brainomics/2014_pca_struct/dice5_ad/results/data_100_100_0.1/results/0/struct_pca_0.01_0.5_0.5/components.npz")
components=components['arr_0']
for k in range(3):
    fig=plot_map2d(components[:,k].reshape(100,100),title="# component %d"%k)
  




