# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:27:31 2016

@author: ad247405

Compute metrics to assess reconstruction error: Frobenius norm and Mean Squared Error
and to assess Stability across resampling : Mean Dice index

Then, assess statistical significance of these score. For the reconstruction error, 
we use a Two sample related two test: TV vs sparse and TV vs ElasticNet

For the stability( Dice index), we use a one sample permutation test on the differences
of dice obtained in two methods

"""
       
        
import numpy as np
from sklearn import metrics 
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from brainomics import plot_utilities
from parsimony.utils import plot_map2d
import scipy.stats




BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/oldies"

#Load components of all 50 datasets 
##############################################################################
MSE_results = np.zeros((50,4))
frob= np.zeros((50,4))
dice= np.zeros((50,4))
components_pca= np.zeros((10000,3,50))
components_sparse= np.zeros((10000,3,50))
components_enet= np.zeros((10000,3,50))
components_tv= np.zeros((10000,3, 50))

for i in range(50):
       
    INPUT_RESULTS_DIR= os.path.join(BASE_DIR,"results_0.1_1e-6_5folds/data_100_100_%r") % (i) 
    INPUT_DATA_DIR= os.path.join(BASE_DIR,"data_0.1/data_100_100_%r") % (i) 
    INPUT_RESULTS_FILE = os.path.join(INPUT_RESULTS_DIR, "results.csv")
    INPUT_BETA_FILE = os.path.join(INPUT_DATA_DIR, "beta3d.npy") 
    #Load masks of Betas star
    mask=np.zeros((100,100,3))
    mask[:,:,0]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_0.npy")).reshape(100,100)
    mask[:,:,1]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_1.npy")).reshape(100,100)
    mask[:,:,2]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_2.npy")).reshape(100,100)
    
    #Load csv file to extract frobenius norm
    data = pd.read_csv(INPUT_RESULTS_FILE)
    frob[i,0] = data[data["params"]=="(u'pca', 0.0, 0.0, 0.0)"].frobenius_test
    frob[i,1] = data[data["params"]=="(u'sparse_pca', 0.0, 0.0, 1.0)"].frobenius_test
    frob[i,2] = data[data["params"]=="(u'struct_pca', 0.01, 1e-05, 0.5)"].frobenius_test
    frob[i,3] = data[data["params"]=="(u'struct_pca', 0.01, 0.5, 0.5)"].frobenius_test
    
    #compute MSE
    MSE_results[i,0] = compute_mse(data,model="(u'pca', 0.0, 0.0, 0.0)",type="pca",p=np.array((0.0,0.0,0.0)),beta_star_path= INPUT_BETA_FILE, beta_path= INPUT_RESULTS_DIR)
    MSE_results[i,1] = compute_mse(data,model="(u'sparse_pca', 0.0, 0.0, 1.0)",type="sparse_pca",p=np.array((0.0,0.0,1.0)),beta_star_path= INPUT_BETA_FILE, beta_path= INPUT_RESULTS_DIR)
    MSE_results[i,2] = compute_mse(data,model="(u'struct_pca', 0.01, 1e-05, 0.5)",type="struct_pca",p=np.array((0.01,1e-05,0.5)),beta_star_path= INPUT_BETA_FILE,beta_path= INPUT_RESULTS_DIR)
    MSE_results[i,3] = compute_mse(data,model="(u'struct_pca', 0.01, 0.5, 0.5)",type="struct_pca",p=np.array((0.01,0.5,0.5)),beta_star_path= INPUT_BETA_FILE,beta_path= INPUT_RESULTS_DIR)
    
    #Load components
    #standard
    comp_path=os.path.join(INPUT_RESULTS_DIR,"results/0/","pca_0.0_0.0_0.0","thresh_components.npz")
    comp = np.load(comp_path)
    components_pca[:,:,i]=comp['arr_0']
   
    #Sparse
    comp_path=os.path.join(INPUT_RESULTS_DIR,"results/0/","sparse_pca_0.0_0.0_1.0","thresh_components.npz")
    comp = np.load(comp_path)
    components_sparse[:,:,i] =comp['arr_0']   
    #Enet
    comp_path=os.path.join(INPUT_RESULTS_DIR,"results/0/","struct_pca_0.01_1e-05_0.5","thresh_components.npz")
    comp = np.load(comp_path)
    components_enet[:,:,i] =comp['arr_0']  
    #TV
    comp_path=os.path.join(INPUT_RESULTS_DIR,"results/0/","struct_pca_0.01_0.5_0.5","thresh_components.npz")
    comp = np.load(comp_path)
    components_tv[:,:,i]=comp['arr_0']

    print i


print MSE_results
print MSE_results[:,:].mean(axis=0)
print MSE_results[:,:].std(axis=0)
print frob[:,:].mean(axis=0)

# Boxplot of MSE scores across methods
###############################################################################
plt.figure()
plt.ylabel("MSE")
plt.grid(True)
plt.title(" MSE based on 50 simulations")
labels=['Standard PCA', 'Sparse PCA', 'Enet PCA','PCA-TV']
plt.boxplot(MSE_results)
#plt.boxplot(frob)
plt.xticks([1, 2, 3,4], labels)
plt.legend()
plt.show()

plt.figure()
plt.ylabel("Frobenius norm")
plt.grid(True)
plt.title(" Frobenius norm based on 50 simulations")
labels=['Standard PCA', 'Sparse PCA', 'Enet PCA','PCA-TV']
plt.boxplot(frob)
#plt.boxplot(frob)
plt.xticks([1, 2, 3,4], labels)
plt.legend()
plt.show()


#Test significance of MSE  (two samples related t test)
##################################################################################
# TV vs sparse
tval, pval = scipy.stats.ttest_rel(MSE_results [:,1],MSE_results [:,3], axis=0)
print ("MSE stats for TV vs sparse: T = %r , pvalue = %r ") %(tval, pval)
#TV vs Enet
tval, pval = scipy.stats.ttest_rel(MSE_results [:,2],MSE_results [:,3], axis=0)
print ("MSE stats for TV vs Enet: T = %r , pvalue = %r ") %(tval, pval)

#Test significance of Frobenius norm  (two samples related t test)
# TV vs sparse
tval, pval = scipy.stats.ttest_rel(frob [:,1],frob [:,3], axis=0)
print ("Frobenius stats for TV vs sparse: T = %r , pvalue = %r ") %(tval, pval)

#TV vs Enet
tval, pval = scipy.stats.ttest_rel(frob [:,2],frob [:,3], axis=0)
print ("Frobenius stats for TV vs Enet: T = %r , pvalue = %r ") %(tval, pval)
 ##################################################################################


#Compute Mean DIce Index across the 50 sets
##################################################################################
components_tv = identify_comp(components_tv)
components_sparse = identify_comp(components_sparse)
components_pca = identify_comp(components_pca)
components_enet = identify_comp(components_enet)
# Compute dice coefficients
print (dice_bar(components_pca[:,0,:])[0] + dice_bar(components_pca[:,1,:])[0] + dice_bar(components_pca[:,2,:])[0] )/3
print (dice_bar(components_sparse[:,0,:])[0]+ dice_bar(components_sparse[:,1,:])[0] + dice_bar(components_sparse[:,2,:])[0] )/3 
print (dice_bar(components_enet[:,0,:])[0]+ dice_bar(components_enet[:,1,:])[0] + dice_bar(components_enet[:,2,:])[0] )/3
print (dice_bar(components_tv[:,0,:])[0]+ dice_bar(components_tv[:,1,:])[0] + dice_bar(components_tv[:,2,:])[0]  )/3
###############################################################################  
  
  
#Statistical test of Dice index 
###############################################################################  
dices_pca = return_dices_pair(components_pca)
dices_sparse = return_dices_pair(components_sparse)
dices_enet = return_dices_pair(components_enet)
dices_tv = return_dices_pair(components_tv)

# I want to test whether this list of pairwise diff is different from zero? 
#(i.e TV leads to different results than sparse?).
#We cannot do a one-sample t-test since samples are not independant!
#Use of permuations
diff_sparse = dices_tv - dices_sparse
diff_enet = dices_tv - dices_enet

pval = one_sample_permutation_test(y=diff_sparse,nperms = 1000)
print ("Dice index stats for TV vs sparse: pvalue = %r ") %(pval)
pval = one_sample_permutation_test(y=diff_enet,nperms = 1000)
print ("Dice index stats for TV vs Enet: pvalue = %r ") %(pval)
###############################################################################







#functions
###############################################################################


def compute_mse(data,model,type,p,beta_star_path,beta_path):
    
    # Load data and Center scale it
    param_path=os.path.join(beta_path,"results/0/%s_%s_%s_%s/components.npz") % (type,p[0],p[1],p[2])
    components = np.load(param_path)
    components=components['arr_0'].reshape(100,100,3)
    
    #Load ground truth
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
    return mean_mse

#mse
def mse(imageA, imageB):
    err = np.sum(((imageA) - (imageB)) ** 2)
    err /= (imageA.shape[0] * imageA.shape[1])
    return err
    
    
#Solve non_identifiability of components 
def identify_comp(comp):
    for i in range(1,50):
        if np.abs(np.corrcoef(comp[:,0,0],comp[:,0,i])[0,1]) <  np.abs(np.corrcoef(comp[:,0,0],comp[:,1,i])[0,1]):
                
            print "components inverted" 
            print i
            temp_comp1 = np.copy(comp[:,1,i])
            comp[:,1,i] = comp[:,0,i]
            comp[:,0,i] = temp_comp1
            
        if np.abs(np.corrcoef(comp[:,1,0],comp[:,1,i])[0,1]) <  np.abs(np.corrcoef(comp[:,1,0],comp[:,2,i])[0,1]):
                
            print "components inverted" 
            print i
            temp_comp2 = np.copy(comp[:,2,i])
            comp[:,2,i] = comp[:,1,i]
            comp[:,1,i] = temp_comp2
    return comp 


#Corrected pvalue using Tmax with 1000 permutation
def one_sample_permutation_test(y,nperms):
    T,p =scipy.stats.ttest_1samp(y,0.0)       
    max_t = list()
    
    for i in xrange(nperms): 
            r=np.random.choice((-1,1),y.shape)
            y_p=r*abs(y)
            Tperm,pp =scipy.stats.ttest_1samp(y_p,0.0)    
            Tperm= np.abs(Tperm)
            max_t.append(Tperm)           
    max_t = np.array(max_t)
    pvalue = np.sum(max_t>=np.abs(T)) / float(nperms)
    return pvalue

def return_dices_pair(comp):
    m, dices_0 = dice_bar(comp[:,0,:])
    m, dices_1 = dice_bar(comp[:,1,:])
    m, dices_2 = dice_bar(comp[:,2,:])
    dices_mean = (dices_0 + dices_1 + dices_2) / 3    
    return dices_mean
    
    

def dice_bar(thresh_comp):
    """Given an array of thresholded component of size n_voxels x n_folds,
    compute the average DICE coefficient.
    """
    n_voxels, n_folds = thresh_comp.shape
    # Paire-wise DICE coefficient (there is the same number than
    # pair-wise correlations)
    n_corr = n_folds * (n_folds - 1) / 2
    thresh_comp_n0 = thresh_comp != 0
    # Index of lines (folds) to use
    ij = [[i, j] for i in xrange(n_folds) for j in xrange(i + 1, n_folds)]
    num =([2 * (np.sum(thresh_comp_n0[:,idx[0]] & thresh_comp_n0[:,idx[1]]))
    for idx in ij])

    denom = [(np.sum(thresh_comp_n0[:,idx[0]]) + \
              np.sum(thresh_comp_n0[:,idx[1]]))
             for idx in ij]
    dices = np.array([float(num[i]) / denom[i] for i in range(n_corr)])
    return dices.mean(), dices


#########################################################################
#########################################################################
#Boxplots 
#
#
#m= np.zeros((50,6))
#m[:,0:2]= m3[:,0:2]
#m[:,2]= m3[:,2]
#m[:,3]= m5[:,2]
#m[:,4]=m6[:,2]
#
#fig, ax1 = plt.subplots(figsize=(10, 5
#))
#plt.ylabel("MSE")
#plt.grid(True)
#plt.title(" SSE based on 50 simulations - SNR=0.1")
#labels=['Standard PCA', 'Sparse PCA', 'PCA-TV eps=1e-3', 'PCA-TV eps=1e-5', 'PCA-TV eps=1e-6']
#plt.boxplot(m)
##xtickNames = plt.setp(ax1, xticklabels=np.repeat(randomDists, 2))
##plt.setp(xtickNames, rotation=45, fontsize=8)
#plt.xticks([1, 2, 3,4,5], labels)
#plt.legend()
#plt.show()
#
#
#
#####Plot stability index
#BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5_ad_validation"
#
#stability_index = np.zeros((50,3,3))
#
#for i in range(50):
#    INPUT_RESULTS_DIR= os.path.join(BASE_DIR,"results_0.1_1e-6_5folds/data_100_100_%r") % (i) 
#    INPUT_RESULTS_FILE = os.path.join(INPUT_RESULTS_DIR, "results.csv")  
#    data = pd.read_csv(INPUT_RESULTS_FILE)
#    stability_index[i,0,:] = data.correlation_mean 
#    stability_index[i,1,:] = data.kappa_mean 
#    stability_index[i,2,:] = data.dice_bar_mean 
#    
#plt.figure()
#plt.ylabel("Dice index across folds")
#plt.grid(True)
#plt.title(" Dice index  across folds based on 50 simulations - SNR=0.1")
#labels=['PCA-TV','Sparse PCA', 'Standard PCA']
#plt.boxplot(stability_index[:,2,:])
#plt.xticks([1, 2, 3], labels)
#plt.legend()
#plt.show()    
#    
#stability_index[:,0,0].mean()  
#stability_index[:,0,0].std()  
#
#
#
#import scipy.stats
#tval, pval = scipy.stats.ttest_rel(stability_index[:,0,0],stability_index [:,0,2], axis=0)
#print tval, pval
