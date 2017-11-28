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
from parsimony.utils import plots
import scipy.stats
import array_utils

import scipy.stats

BASE_DIR = "/neurospin/brainomics/2016_pca_struct/dice"
BASE_DIR_GN = "/neurospin/brainomics/2016_pca_struct/dice/2017"
#Load components of all 50 datasets
##############################################################################
MSE_results = np.zeros((50,6))
frob_test= np.zeros((50,6))
frob_train= np.zeros((50,6))
dice= np.zeros((50,6))
components_pca= np.zeros((10000,10,50))
components_sparse= np.zeros((10000,10,50))
components_enet= np.zeros((10000,10,50))
components_tv= np.zeros((10000,10, 50))
components_gn= np.zeros((10000,10, 50))
components_je= np.zeros((10000,10, 50))

for i in range(50):

    INPUT_RESULTS_DIR= os.path.join(BASE_DIR,"results_10comp/data_100_100_%r") % (i)
    INPUT_RESULTS_DIR_GN= os.path.join(BASE_DIR_GN,"results/data_100_100_%r") % (i)
    INPUT_RESULTS_DIR_JE= os.path.join(BASE_DIR_GN,"results_Jenatton/data_100_100_%r") % (i)
    INPUT_DATA_DIR= os.path.join(BASE_DIR,"data_0.1/data_100_100_%r") % (i)
    INPUT_RESULTS_FILE = os.path.join(INPUT_RESULTS_DIR, "results_dCV_5folds.xlsx")
    INPUT_RESULTS_FILE_GN = os.path.join(INPUT_RESULTS_DIR_GN, "results_dCV_5folds.xlsx")
    INPUT_RESULTS_FILE_JE = os.path.join(INPUT_RESULTS_DIR_JE, "results_dCV_5folds.xlsx")
    INPUT_BETA_FILE = os.path.join(INPUT_DATA_DIR, "beta3d.npy")
    #Load masks of Betas star
    mask=np.zeros((100,100,3))
    mask[:,:,0]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_0.npy")).reshape(100,100)
    mask[:,:,1]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_1.npy")).reshape(100,100)
    mask[:,:,2]= np.load(os.path.join(INPUT_RESULTS_DIR,"mask_2.npy")).reshape(100,100)

    #Load csv file to extract frobenius norm
    data_all = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 0)
    score_cv_enettv= pd.read_excel(INPUT_RESULTS_FILE,sheetname = 5)
    score_cv_enet = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 6)
    score_cv_sparse = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 7)
    score_cv_gn = pd.read_excel(INPUT_RESULTS_FILE_GN,sheetname = 3)
    score_cv_je = pd.read_excel(INPUT_RESULTS_FILE_JE,sheetname = 3)
    frob_test[i,0] = data_all[data_all["param_key"]=="pca_0.0_0.0_0.0"].frobenius_test
    frob_test[i,1] = score_cv_sparse.frobenius_test
    frob_test[i,2] = score_cv_enet.frobenius_test
    frob_test[i,3] = score_cv_gn.frobenius_test
    frob_test[i,4] = score_cv_je.frobenius_test
    frob_test[i,5] = score_cv_enettv.frobenius_test


    frob_train[i,0] = data_all[data_all["param_key"]=="pca_0.0_0.0_0.0"].frobenius_train
    frob_train[i,1] = score_cv_sparse.frobenius_train
    frob_train[i,2] = score_cv_enet.frobenius_train
    frob_train[i,3] = score_cv_gn.frobenius_train
    frob_train[i,4] = score_cv_je.frobenius_train
    frob_train[i,5] = score_cv_enettv.frobenius_train



    scores_argmax_enet = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 3)
    best_enet_param = scores_argmax_enet.param_key[0]

    scores_argmax_enettv = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 2)
    best_enettv_param = scores_argmax_enettv.param_key[0]

    scores_argmax_sparse = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 4)
    best_sparse_param = scores_argmax_sparse.param_key[0]

    scores_argmax_gn = pd.read_excel(INPUT_RESULTS_FILE_GN,sheetname = 2)
    best_gn_param = scores_argmax_gn.param_key[0]

    scores_argmax_je = pd.read_excel(INPUT_RESULTS_FILE_JE,sheetname = 2)
    best_je_param = scores_argmax_je.param_key[0]

    #compute MSE
    MSE_results[i,0] = compute_mse(p = best_enet_param ,beta_star_path= INPUT_BETA_FILE, beta_path= INPUT_RESULTS_DIR)
    MSE_results[i,1] = compute_mse(p = best_sparse_param,beta_star_path= INPUT_BETA_FILE, beta_path= INPUT_RESULTS_DIR)
    MSE_results[i,2] = compute_mse(p = best_enet_param,beta_star_path= INPUT_BETA_FILE,beta_path= INPUT_RESULTS_DIR)
    MSE_results[i,3] = compute_mse(p = best_gn_param,beta_star_path= INPUT_BETA_FILE,beta_path= INPUT_RESULTS_DIR_GN)
    MSE_results[i,4] = compute_mse(p = best_je_param,beta_star_path= INPUT_BETA_FILE,beta_path= INPUT_RESULTS_DIR_JE)
    MSE_results[i,5] = compute_mse(p = best_enettv_param,beta_star_path= INPUT_BETA_FILE,beta_path= INPUT_RESULTS_DIR)

    #extract components
    pca_param_path=os.path.join(INPUT_RESULTS_DIR,"results/cv01/all/%s/components.npz") % ("pca_0.0_0.0_0.0")
    components_pca[:,:,i] = np.load(pca_param_path)['arr_0']

    sparse_param_path=os.path.join(INPUT_RESULTS_DIR,"results/cv01/all/%s/components.npz") % (best_sparse_param)
    print(best_sparse_param)
    components_sparse[:,:,i] = np.load(sparse_param_path)['arr_0']

    enet_param_path=os.path.join(INPUT_RESULTS_DIR,"results/cv01/all/%s/components.npz") % (best_enet_param)
    print(best_enet_param)
    components_enet[:,:,i] = np.load(enet_param_path)['arr_0']

    gn_param_path=os.path.join(INPUT_RESULTS_DIR_GN,"results/cv01/all/%s/components.npz") % (best_gn_param)
    print(best_gn_param)
    components_gn[:,:,i] = np.load(gn_param_path)['arr_0']

    je_param_path=os.path.join(INPUT_RESULTS_DIR_JE,"results/cv01/all/%s/V.npy") % (best_je_param)
    print(best_je_param)
    components_je[:,:,i] = np.load(je_param_path)

    enettv_param_path=os.path.join(INPUT_RESULTS_DIR,"results/cv01/all/%s/components.npz") % (best_enettv_param)
    print(best_enettv_param)
    components_tv[:,:,i] = np.load(enettv_param_path)['arr_0']


    print (i)


print( MSE_results)
print (MSE_results[:,:].mean(axis=0))
print (MSE_results[:,:].std(axis=0))
print (frob_train[:,:].mean(axis=0))
print (frob_test[:,:].mean(axis=0))

# Boxplot of MSE scores across methods
###############################################################################
plt.figure()
plt.ylabel("MSE")
plt.grid(True)
plt.title(" MSE based on 50 simulations")
labels=['Standard PCA', 'Sparse PCA', 'Enet PCA',"PCA-GraphNet", "SSPCA -Jenatton" 'PCA-TV']
plt.boxplot(MSE_results)
#plt.boxplot(frob)
plt.xticks([1, 2, 3,4,5,6], labels)
plt.legend()
plt.show()

plt.figure()
plt.ylabel("Frobenius norm")
plt.grid(True)
plt.title(" Frobenius norm based on 50 simulations")
labels=['Standard PCA', 'Sparse PCA', 'Enet PCA',"PCA-GraphNet","SSPCA -Jenatton",'PCA-TV']
plt.boxplot(frob_test)
#plt.boxplot(frob)
plt.xticks([1, 2, 3,4,5], labels)
plt.legend()
plt.show()


#PARAMETRIC TEST
#Test significance of MSE  (two samples related t test)
##################################################################################
# TV vs sparse
tval, pval = scipy.stats.ttest_rel(MSE_results [:,1],MSE_results [:,5], axis=0)
print (("MSE stats for TV vs sparse: T = %r , pvalue = %r ") %(tval, pval))
#TV vs Enet
tval, pval = scipy.stats.ttest_rel(MSE_results [:,2],MSE_results [:,5], axis=0)
print (("MSE stats for TV vs Enet: T = %r , pvalue = %r ") %(tval, pval))
#TV vs GraphNet
tval, pval = scipy.stats.ttest_rel(MSE_results [:,3],MSE_results [:,5], axis=0)
print (("MSE stats for TV vs GraphNet: T = %r , pvalue = %r ") %(tval, pval))

tval, pval = scipy.stats.ttest_rel(MSE_results [:,4],MSE_results [:,5], axis=0)
print (("MSE stats for TV vs Jenatton: T = %r , pvalue = %r ") %(tval, pval))


#Test significance of Frobenius norm  (two samples related t test)
# TV vs sparse
tval, pval = scipy.stats.ttest_rel(frob_test [:,1],frob_test [:,5], axis=0)
print (("Frobenius stats for TV vs sparse: T = %r , pvalue = %r ") %(tval, pval))

#TV vs Enet
tval, pval = scipy.stats.ttest_rel(frob_test [:,2],frob_test [:,5], axis=0)
print (("Frobenius stats for TV vs Enet: T = %r , pvalue = %r ") %(tval, pval))

#TV vs GraphNet
tval, pval = scipy.stats.ttest_rel(frob_test [:,3],frob_test [:,5], axis=0)
print (("Frobenius stats for TV vs GraphNet: T = %r , pvalue = %r ") %(tval, pval))


#TV vs Jenatton
tval, pval = scipy.stats.ttest_rel(frob_test [:,4],frob_test [:,5], axis=0)
print (("Frobenius stats for TV vs Jenatton: T = %r , pvalue = %r ") %(tval, pval))


#NON PARAMETRIC TEST
#Test significance of MSE  (two samples related t test)
##################################################################################
# TV vs sparse
tval, pval = scipy.stats.wilcoxon(MSE_results [:,1],MSE_results [:,5])
print (("MSE stats for TV vs sparse: T = %r , pvalue = %r ") %(tval, pval))
#TV vs Enet
tval, pval = scipy.stats.wilcoxon(MSE_results [:,2],MSE_results [:,5])
print (("MSE stats for TV vs Enet: T = %r , pvalue = %r ") %(tval, pval))
#TV vs GraphNet
tval, pval = scipy.stats.wilcoxon(MSE_results [:,3],MSE_results [:,5])
print (("MSE stats for TV vs GraphNet: T = %r , pvalue = %r ") %(tval, pval))

tval, pval = scipy.stats.wilcoxon(MSE_results [:,4],MSE_results [:,5])
print (("MSE stats for TV vs Jenatton: T = %r , pvalue = %r ") %(tval, pval))


#Test significance of Frobenius norm  (two samples related t test)
# TV vs sparse
tval, pval = scipy.stats.wilcoxon(frob_test [:,1],frob_test [:,5])
print (("Frobenius stats for TV vs sparse: T = %r , pvalue = %r ") %(tval, pval))

#TV vs Enet
tval, pval = scipy.stats.wilcoxon(frob_test [:,2],frob_test [:,5])
print (("Frobenius stats for TV vs Enet: T = %r , pvalue = %r ") %(tval, pval))

#TV vs GraphNet
tval, pval = scipy.stats.wilcoxon(frob_test [:,3],frob_test [:,5])
print (("Frobenius stats for TV vs GraphNet: T = %r , pvalue = %r ") %(tval, pval))


#TV vs Jenatton
tval, pval = scipy.stats.wilcoxon(frob_test [:,4],frob_test [:,5])
print (("Frobenius stats for TV vs Jenatton: T = %r , pvalue = %r ") %(tval, pval))





y = frob_test[:,4]-frob_test[:,1]
one_sample_permutation_test(y,nperms)
#Corrected pvalue using Tmax with 1000 permutation
def one_sample_permutation_test(y,nperms):
    nperms=1000
    tval, pval = scipy.stats.ttest_1samp(y,popmean=0)
    max_t = list()
    two_tailed = True

    for i in range(nperms):
            r=np.random.choice((-1,1),50)
            frob_permutated=r*abs(y)
            tvals_perm,_ = scipy.stats.ttest_1samp( frob_permutated,popmean=0)
            if two_tailed:
                tvals_perm = np.abs(tvals_perm)
            max_t.append(np.nanmax(tvals_perm))
            #print(i)

    max_t = np.array(max_t)
    tvals_perm = np.abs(tvals_perm) if two_tailed else  tvals_perm
    pvalues = ((np.sum(max_t> np.abs(tval))+1) / float(nperms))
    print (pvalues)
    return pvalues








 ##################################################################################
for i in range(10):
    for j in range(50):
        components_sparse[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_sparse[:,i,j], .99)
        components_enet[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_enet[:,i,j], .99)
        components_tv[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_tv[:,i,j], .99)
        components_gn[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_gn[:,i,j], .99)
        components_je[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_je[:,i,j], .99)


#Compute Mean DIce Index across the 50 sets
##################################################################################
components_tv = identify_comp(components_tv)
components_sparse = identify_comp(components_sparse)
components_pca = identify_comp(components_pca)
components_gn = identify_comp(components_gn)
components_je = identify_comp(components_je)
components_enet = identify_comp(components_enet)


# Compute dice coefficients
print ((dice_bar(components_pca[:,0,:])[0] + dice_bar(components_pca[:,1,:])[0] + dice_bar(components_pca[:,2,:])[0] )/3)
print ((dice_bar(components_sparse[:,0,:])[0]+ dice_bar(components_sparse[:,1,:])[0] + dice_bar(components_sparse[:,2,:])[0] )/3)
print ((dice_bar(components_enet[:,0,:])[0]+ dice_bar(components_enet[:,1,:])[0] + dice_bar(components_enet[:,2,:])[0] )/3)
print ((dice_bar(components_gn[:,0,:])[0]+ dice_bar(components_gn[:,1,:])[0] + dice_bar(components_gn[:,2,:])[0]  )/3)
print ((dice_bar(components_je[:,0,:])[0]+ dice_bar(components_je[:,1,:])[0] + dice_bar(components_je[:,2,:])[0]  )/3)
print ((dice_bar(components_tv[:,0,:])[0]+ dice_bar(components_tv[:,1,:])[0] + dice_bar(components_tv[:,2,:])[0]  )/3)

###############################################################################


#Statistical test of Dice index
###############################################################################
dices_pca = return_dices_pair(components_pca)
dices_sparse = return_dices_pair(components_sparse)
dices_enet = return_dices_pair(components_enet)
dices_gn = return_dices_pair(components_gn)
dices_tv = return_dices_pair(components_tv)

# I want to test whether this list of pairwise diff is different from zero?
#(i.e TV leads to different results than sparse?).
#We cannot do a one-sample t-test since samples are not independant!
#Use of permuations
diff_sparse = dices_tv - dices_sparse
diff_enet = dices_tv - dices_enet
diff_gn = dices_tv - dices_gn

pval = one_sample_permutation_test(y=diff_sparse,nperms = 1000)
print (("Dice index stats for TV vs sparse: pvalue = %a ") %(pval))
pval = one_sample_permutation_test(y=diff_enet,nperms = 1000)
print (("Dice index stats for TV vs Enet: pvalue = %r ") %(pval))
pval = one_sample_permutation_test(y=diff_gn,nperms = 1000)
print (("Dice index stats for TV vs GraphNet: pvalue = %r ") %(pval))
###############################################################################

###############################################################################







#functions
###############################################################################


def compute_mse(p,beta_star_path,beta_path):

    # Load data and Center scale it
    if (p[:8] == "Jenatton"):
        param_path=os.path.join(beta_path,"results/cv00/all/%s/V.npy") % (p)
        components = np.load(param_path)
        components=components.reshape(100,100,10)
    else:
        param_path=os.path.join(beta_path,"results/cv00/all/%s/components.npz") % (p)
        components = np.load(param_path)
        components=components['arr_0'].reshape(100,100,10)

    #Load ground truth
    true=np.zeros((100,100,10))
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


#Corrected pvalue using Tmax with 1000 permutation
def one_sample_permutation_test(y,nperms):
    T,p =scipy.stats.ttest_1samp(y,0.0)
    max_t = list()

    for i in range(nperms):
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
