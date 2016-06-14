# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:13:18 2016

@author: ad247405
"""
import numpy as np
from sklearn import metrics 
import os

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from brainomics import plot_utilities
from parsimony.utils import plot_map2d


#Compare reconstruction of loading vector when varying EPS 


BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5_ad_validation"
DATA_DIR = os.path.join(BASE_DIR,"data_0.1/data_100_100_1") 

for EPS in range(2,8):
    RESULTS_DIR = os.path.join(BASE_DIR,"results_0.1_1e-%r","data_100_100_1") % (EPS)
    BETA_DIR = os.path.join(RESULTS_DIR,"results/0/struct_pca_0.01_0.5_0.5")
    
    #Load masks of Betas star
    mask=np.zeros((100,100,3))
    mask[:,:,0]= np.load(os.path.join(RESULTS_DIR,"mask_0.npy")).reshape(100,100)
    mask[:,:,1]= np.load(os.path.join(RESULTS_DIR,"mask_1.npy")).reshape(100,100)
    mask[:,:,2]= np.load(os.path.join(RESULTS_DIR,"mask_2.npy")).reshape(100,100)
    
    true=np.zeros((100,100,3))
    for k in range(0,3):
        beta3d = np.load(os.path.join(DATA_DIR,"beta3d.npy")).reshape(100,100)
        beta3d[mask[:,:,k]==False]=0
        true[:,:,k]=beta3d
        true[:,:,k]=true[:,:,k]-true[:,:,k].mean()
        true[:,:,k]=true[:,:,k]/true[:,:,k].std()
    
    
    #Load result of minimization
    beta_path=os.path.join(BETA_DIR,"components.npz") 
    components = np.load(beta_path)
    components=components['arr_0'].reshape(100,100,3)

 
    for k in range(0,3):
        components[:,:,k]=components[:,:,k]-components[:,:,k].mean()
        components[:,:,k]=components[:,:,k]/components[:,:,k].std()
    true=np.abs(true)
    components=np.abs(components)


    for k in range(3):
        plot = plt.subplot(3, 3, 3+1+k)
        title = "comp #{i}".format(i=k)
        plot_map2d(components[:,:,k],plot, title=title)
    f = plt.gcf()  
    plt.show()
    filename= os.path.join(BASE_DIR,"eps_components_0.1/eps_1e-%r.svg") %(EPS)
    f.savefig(filename)
    plt.clf()
    

    
    
    mean_mse=0  
    for k in range(0,2):
        data=np.zeros((10000,2))    
        data[:,0] = true[:,:,k].reshape(10000)
        R=np.zeros((2))
        for i in range(0,2):
            data[:,1] = components[:,:,i].reshape(10000)
            R[i]=np.abs(np.corrcoef(np.abs(data.T))[0,1])
        m=mse(true[:,:,k],components[:,:,np.argmax(R)])       
        mean_mse=mean_mse+m       
    m=mse(true[:,:,2],components[:,:,2])  
    mean_mse=mean_mse+m  
    mean_mse=mean_mse / float(3)  
    print mean_mse





#mse
def mse(imageA, imageB):
    err = np.sum(((imageA) - (imageB)) ** 2)
    err /= (imageA.shape[0] * imageA.shape[1])
    return err
        