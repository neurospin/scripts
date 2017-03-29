# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:29:46 2017

@author: ad247405
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from brainomics import plot_utilities
from parsimony.utils import plots

BASE_DIR = "/neurospin/brainomics/2016_pca_struct/dice/local_minima_experiment"
INPUT_RESULTS_DIR_FORMAT = os.path.join(BASE_DIR,"data_100_100_{set}")
 
comp = np.zeros((10000,3,50))
for set in range(0,50):
    input_dir = INPUT_RESULTS_DIR_FORMAT.format(set=set)
    comp[:,:,set] = np.load(os.path.join(input_dir,"results","0","struct_pca_0.01_0.5_0.5","components.npz"))['arr_0']
    #fig=plots.map2d(comp[:,1,set].reshape(100,100))
    comp = identify_comp(comp)
    

mean_dice_comp0, dice_comp0 = dice_bar(comp[:,0,:])
mean_dice_comp1, dice_comp1 = dice_bar(comp[:,1,:])
mean_dice_comp2, dice_comp2 = dice_bar(comp[:,2,:])

data = dice_comp0 
plt.boxplot(data)

x0 =  np.random.normal(0, 0.04, size = len(dice_comp0)) 
x1 =  np.random.normal(1, 0.04, size = len(dice_comp1))
x2 =  np.random.normal(2, 0.04, size = len(dice_comp2)) 
plt.plot(x0,dice_comp0,'o')
plt.plot(x1,dice_comp1,'o')
plt.plot(x2,dice_comp2,'o')


#
#
#components = comp[2,:,:]
#for k in range(3):
#    fig=plots.map2d(components[:,k].reshape(100,100))
#  
  
def dice_bar(thresh_comp):
    """Given an array of thresholded component of size n_voxels x n_folds,
    compute the average DICE coefficient.
    """
    n_voxels, n_folds = thresh_comp.shape
    # Paire-wise DICE coefficient (there is the same number than
    # pair-wise correlations)
    n_corr = int(n_folds * (n_folds - 1) / 2)
    thresh_comp_n0 = thresh_comp != 0
    # Index of lines (folds) to use
    ij = [[i, j] for i in range(n_folds) for j in range(i + 1, n_folds)]
    num =([2 * (np.sum(thresh_comp_n0[:,idx[0]] & thresh_comp_n0[:,idx[1]]))
    for idx in ij])

    denom = [(np.sum(thresh_comp_n0[:,idx[0]]) + \
              np.sum(thresh_comp_n0[:,idx[1]]))
             for idx in ij]
    dices = np.array([float(num[i]) / denom[i] for i in range(n_corr)])
    return dices.mean(), dices  
    
    
def identify_comp(comp):
    for i in range(1,50):
        if np.abs(np.corrcoef(comp[:,0,0],comp[:,0,i])[0,1]) <  np.abs(np.corrcoef(comp[:,0,0],comp[:,1,i])[0,1]):
                
            print ("components inverted")
            print (i)
            temp_comp1 = np.copy(comp[:,1,i])
            comp[:,1,i] = comp[:,0,i]
            comp[:,0,i] = temp_comp1
            
        if np.abs(np.corrcoef(comp[:,1,0],comp[:,1,i])[0,1]) <  np.abs(np.corrcoef(comp[:,1,0],comp[:,2,i])[0,1]):
                
            print ("components inverted" )
            print (i)
            temp_comp2 = np.copy(comp[:,2,i])
            comp[:,2,i] = comp[:,1,i]
            comp[:,1,i] = temp_comp2
    return comp 

  
  
