
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


###############################################################################
# SCZ ONLY
############################################################################### 

INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/population.csv"
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/pcatv_scz/5_folds_NMoprhCH_scz"
INPUT_RESULTS = os.path.join(BASE_PATH,"results","0")


# Compute clinical Scores

pop = pd.read_csv(INPUT_POPULATION)
age =  pop[pop.dx_num ==1].age.values


scores = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.5_0.1","X_test_transform.npz"))['arr_0']


for i in range(scores.shape[1]):        
    corr,p = pearsonr(scores[:,i],age)
    if p < 0.05:
        print ("Significant correlation between age and score on component %s" % (i))
        plt.figure()
        plt.plot(scores[:,i],age,'o')
        plt.xlabel('Score on component %s' %(i))
        plt.ylabel('age')
        plt.title("Pearson's correlation = %.02f, p = %.01e" % (corr,p),fontsize=12)
############################################################################### 
############################################################################### 
# PLOT ALL CORRELATIONW WITH AGE
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(scores[:,i],age)
    axs[i].plot(scores[:,i],age,'o', markersize = 4)
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)  
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('age')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/pcatv_scz/5_folds_NMoprhCH_scz/correlation_Age.pdf")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/pcatv_scz/5_folds_NMoprhCH_scz/correlation_Age.png")
############################################################################### 
    

###############################################################################
# CONTROLS ONLY
############################################################################### 

INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/population.csv"
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/pcatv_controls/5_folds_NMoprhCH_controls"
INPUT_RESULTS = os.path.join(BASE_PATH,"results","0")


# Compute clinical Scores

pop = pd.read_csv(INPUT_POPULATION)

age =  pop[pop.dx_num ==0].age.values


scores = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.8_0.5","X_test_transform.npz"))['arr_0']


for i in range(scores.shape[1]):
      corr,p = pearsonr(scores[:,i],age)
      if p < 0.05:
        print ("Significant correlation between age and score on component %s" % (i))
        plt.figure()
        plt.plot(scores[:,i],age,'o')
        plt.xlabel('Score on component %s' %(i))
        plt.ylabel('age')
        plt.title("Pearson's correlation = %.02f, p = %.01e" % (corr,p),fontsize=12)
        
############################################################################### 
# PLOT ALL CORRELATIONW WITH AGE
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(scores[:,i],age)
    axs[i].plot(scores[:,i],age,'o', markersize = 4)
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)  
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('age')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/pcatv_controls/5_folds_NMoprhCH_controls/correlation_Age.pdf")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/pcatv_controls/5_folds_NMoprhCH_controls/correlation_Age.png")
############################################################################### 
    
    

        
        
############################################################################### 
############################################################################### 


###############################################################################
# CONTROLS + SCZ
############################################################################### 

INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/population.csv"
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/pcatv_all/5_folds_NMoprhCH_all"
INPUT_RESULTS = os.path.join(BASE_PATH,"results","0")
INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/data/y.npy' 


y = np.load(INPUT_DATA_y)
pop = pd.read_csv(INPUT_POPULATION)
age =  pop.age.values


scores = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.8_0.5","X_test_transform.npz"))['arr_0']


for i in range(scores.shape[1]):
      corr,p = pearsonr(scores[:,i],age)
      if p < 0.05:
        print ("Significant correlation between age and score on component %s" % (i))
        plt.figure()
        plt.plot(scores[y==0,i],age[y==0],'o')
        plt.plot(scores[y==1,i],age[y==1],'o')
        plt.xlabel('Score on component %s' %(i))
        plt.ylabel('age')
        plt.title("Pearson's correlation = %.02f, p = %.01e" % (corr,p),fontsize=12)
############################################################################### 
############################################################################### 

############################################################################### 
############################################################################### 
# PLOT ALL CORRELATIONW WITH AGE
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(scores[:,i],age)
    axs[i].plot(scores[y==0,i],age[y==0],'o', markersize = 4)
    axs[i].plot(scores[y==1,i],age[y==1],'o', markersize = 4)
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)  
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('age')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/pcatv_all/correlation_Age.pdf")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/pcatv_all/correlation_Age.png")
############################################################################### 
 ############################################################################### 
  