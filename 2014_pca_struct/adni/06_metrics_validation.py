# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:19:45 2016

@author: ad247405
"""
import pandas as pd
import scipy
from scipy import stats

BASE_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs"
INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only/adni_5folds/"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"results")
INPUT_DATASET = os.path.join(INPUT_BASE_DIR,"X.npy")
INPUT_MASK = os.path.join(BASE_DIR,"mask.npy")                         
INPUT_RESULTS_FILE=os.path.join(INPUT_BASE_DIR,"results.csv")                          
INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR,"config_5folds.json")
OUTPUT_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only/adni_5folds"
OUTPUT_COMPONENTS = os.path.join(OUTPUT_DIR,"components.csv")


#Test of significance of Frobenius norm across folds
################################################################################
data = pd.read_csv(INPUT_RESULTS_FILE)
frob=np.zeros((4,5))
for i in range(1,6):
    frob[0,i-1]= data[data["params"]=="(u'pca', 0.0, 0.0, 0.0)"]["frob_test_fold%s"%(i)] 
    frob[1,i-1]= data[data["params"]=="(u'sparse_pca', 0.0, 0.0, 1.0)"]["frob_test_fold%s"%(i)] 
    frob[2,i-1]= data[data["params"]=="(u'struct_pca', 0.1, 1e-06, 0.1)"]["frob_test_fold%s"%(i)] 
    frob[3,i-1]= data[data["params"]=="(u'struct_pca', 0.1, 0.5, 0.1)"]["frob_test_fold%s"%(i)] 



#frob[0] is PCA-TV and frbo[1] is sparse, frob[2] is Enet
tval, pval = scipy.stats.ttest_rel(frob[3,:],frob[1,:])
print ("Frobenius stats for TV vs sparse: T = %r , pvalue = %r ") %(tval, pval)

tval, pval = scipy.stats.ttest_rel(frob[3,:],frob[2,:])
print ("Frobenius stats for TV vs Enet: T = %r , pvalue = %r ") %(tval, pval)

################################################################################
#Test Frob norm significance with permutation
diff_sparse = frob[1,:] - frob[3,:] 
diff_enet = frob[2,:] - frob[3,:]   
pval = one_sample_permutation_test(y=diff_sparse,nperms = 1000)
print ("Dice index stats for TV vs sparse: pvalue = %r ") %(pval)
pval = one_sample_permutation_test(y=diff_enet,nperms = 1000)
print ("Dice index stats for TV vs Enet: pvalue = %r ") %(pval)



#Statistical test of Dice index 
###############################################################################  
dices_pca = np.load('/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only/adni_5folds/results/dices_mean_pca.npy')
dices_sparse = np.load('/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only/adni_5folds/results/dices_mean_sparse_pca_1.npy')
dices_enet = np.load('/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only/adni_5folds/results/dices_mean_enet.npy')
dices_tv = np.load('/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only/adni_5folds/results/dices_mean_struct_pca.npy')

dices_sparse = np.load('/neurospin/brainomics/2014_pca_struct/adni/fs_3comp_patients_only/adni_5folds/results/dices_mean_sparse_pca_1.npy')

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




def return_dices_pair(comp):
    m, dices_0 = dice_bar(comp[:,0,:])
    m, dices_1 = dice_bar(comp[:,1,:])
    m, dices_2 = dice_bar(comp[:,2,:])
    dices_mean = (dices_0 + dices_1 + dices_2) / 3    
    return dices_mean
    
def one_sample_permutation_test(y,nperms):
    T,p =scipy.stats.ttest_1samp(y,0.0)     
    max_t = list()
    
    for i in xrange(nperms): 
            r=np.random.choice((-1,1),y.shape[0])
            y_p=r*abs(y)
            Tperm,pp =scipy.stats.ttest_1samp(y_p,0.0)    
            Tperm= np.abs(Tperm)
            max_t.append(Tperm)           
    max_t = np.array(max_t)
    pvalue = np.sum(max_t>=np.abs(T)) / float(nperms)
    return pvalue
    

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




