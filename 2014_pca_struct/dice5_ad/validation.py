# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:27:31 2016

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


#Search for best reconstruction using two metrics: Frobenius and EVR
def find_best_params(metric,pca):
    if metric == 'frob':
        arg=pca.frobenius_test.argmin()
    if metric == 'evr':    
        arg=pca.evr_test_mean.argmax()
        
    model=pca['model']
    global_pen=pca['global_pen']
    tv=pca['tv_ratio']
    l1=pca['l1_ratio']
    params=np.array((model[arg],global_pen[arg],tv[arg],l1[arg]))
    return params


def compute_mse(data,model,metric,beta_star_path,beta_path):
    
    pca=data[data.model == model]
    p = find_best_params(metric,pca)
    #print p
    # Load data and Center scale it
    param_path=os.path.join(beta_path,"results/0/%s_%s_%s_%s/components.npz") % (model,p[1],p[2],p[3])
    components = np.load(param_path)
    components=components['arr_0'].reshape(100,100,3)
    
    true=np.zeros((100,100,3))  
    for k in range(0,3):
        beta3d = np.load(beta_star_path)
        beta3d=beta3d.reshape(100,100)
        beta3d[mask[:,:,k]==False]=0
        true[:,:,k]=beta3d
        true[:,:,k]=true[:,:,k]-true[:,:,k].mean()
        true[:,:,k]=true[:,:,k]/true[:,:,k].std()

        components[:,:,k]=components[:,:,k]-components[:,:,k].mean()
        components[:,:,k]=components[:,:,k]/components[:,:,k].std()
        
    
#   #Plot command
#    for k in range(3):
##
#        plot_map2d(components[:,:,k],title=" #%d"%k)
#        plot_map2d(true[:,:,k],title="# TRUE BETA - component %d"%k)
##       
    #Take absolute value
    true=np.abs(true)
    components=np.abs(components)
 
    #identify components correlation with ground truth
   
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
    return mean_mse

#mse
def mse(imageA, imageB):
    err = np.sum(((imageA) - (imageB)) ** 2)
    err /= (imageA.shape[0] * imageA.shape[1])
    return err
    
  
BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5_ad_validation"
MSE_results=np.zeros((50,3))
for i in range(50):
    
    
    INPUT_RESULTS_DIR= os.path.join(BASE_DIR,"results_0.1_1e-3/data_100_100_%r") % (i) 
    INPUT_DATA_DIR= os.path.join(BASE_DIR,"data_0.1/data_100_100_%r") % (i) 
    INPUT_RESULTS_FILE = os.path.join(INPUT_RESULTS_DIR, "results.csv")
    INPUT_BETA_FILE = os.path.join(INPUT_DATA_DIR, "beta3d.npy") 
    #Load masks of Betas star
    mask=np.zeros((100,100,3))
    mask[:,:,0]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_0.npy")).reshape(100,100)
    mask[:,:,1]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_1.npy")).reshape(100,100)
    mask[:,:,2]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_2.npy")).reshape(100,100)
    
    #Load result of minimization
    data = pd.read_csv(INPUT_RESULTS_FILE)
    data['evr_test_mean'] = (data.evr_test_0 + data.evr_test_1 + data.evr_test_2) / float(3)  

    MSE_results[i,0] = compute_mse(data,model="pca",metric="frob",beta_star_path= INPUT_BETA_FILE,beta_path= INPUT_RESULTS_DIR)
    MSE_results[i,1] = compute_mse(data,model="sparse_pca",metric="frob",beta_star_path= INPUT_BETA_FILE, beta_path= INPUT_RESULTS_DIR)
    MSE_results[i,2] = compute_mse(data,model="struct_pca",metric="frob",beta_star_path= INPUT_BETA_FILE, beta_path= INPUT_RESULTS_DIR)
    print i


print MSE_results

print MSE_results[:,:].mean(axis=0)
print MSE_results[:,:].std(axis=0)


plt.figure()
plt.ylabel("MSE")
plt.grid(True)
plt.title(" SSE based on 50 simulations - SNR=0.25")
labels=['Standard PCA', 'Sparse PCA', 'PCA-TV']
plt.boxplot(MSE_results)
plt.xticks([1, 2, 3], labels)
plt.legend()
plt.show()


import scipy.stats
tval, pval = scipy.stats.ttest_rel(MSE_results [:,1],MSE_results [:,2], axis=0)
print tval, pval

tval, pval = scipy.stats.ttest_rel(MSE_results [:,0],MSE_results [:,2], axis=0)
print tval, pval


#########################################################################
#########################################################################
#Boxplots 


m= np.zeros((50,6))
m[:,0:2]= m3[:,0:2]
m[:,2]= m3[:,2]
m[:,3]= m5[:,2]
m[:,4]=m6[:,2]

fig, ax1 = plt.subplots(figsize=(10, 5
))
plt.ylabel("MSE")
plt.grid(True)
plt.title(" SSE based on 50 simulations - SNR=0.1")
labels=['Standard PCA', 'Sparse PCA', 'PCA-TV eps=1e-3', 'PCA-TV eps=1e-5', 'PCA-TV eps=1e-6']
plt.boxplot(m)
#xtickNames = plt.setp(ax1, xticklabels=np.repeat(randomDists, 2))
#plt.setp(xtickNames, rotation=45, fontsize=8)
plt.xticks([1, 2, 3,4,5], labels)
plt.legend()
plt.show()

