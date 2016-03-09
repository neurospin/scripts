# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:12:36 2016

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


################
# Input/Output #
################

BASE_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs"
INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs/adni_5folds/"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"results")
INPUT_DATASET = os.path.join(INPUT_BASE_DIR,"X.npy")
INPUT_MASK = os.path.join(BASE_DIR,"mask.npy")                         
INPUT_RESULTS_FILE=os.path.join(INPUT_BASE_DIR,"results.csv")                          
INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR,"config_5folds.json")
OUTPUT_DIR = "/neurospin/brainomics/2014_pca_struct/adni/fs_6/adni_5folds"
OUTPUT_COMPONENTS = os.path.join(OUTPUT_DIR,"components.csv")

##############
# Parameters #
##############

N_COMP = 3
INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'components.npz')
INPUT_PROJECTIONS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'X_train_transform.npz')

OUTPUT_COMPONENTS_FILE_FORMAT = os.path.join(OUTPUT_DIR,
                                             '{name}.nii')

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


####################################################################

# Load data
####################################################################

X=np.load(os.path.join(BASE_DIR,'X.npy'))
y=np.load(os.path.join(INPUT_BASE_DIR,'y.npy')).reshape(X.shape[0])
label=y 
  
  
#Define the parameter to load  
params=np.array(('struct_pca', '0.1', '0.5', '0.1')) 
params=np.array(('struct_pca', '0.1', '1e-06', '0.1')) 
params=np.array(('sparse_pca', '0.0', '0.0', '1.0')) 
params=np.array(('pca', '0.0', '0.0', '0.0')) 


components  =np.zeros((X.shape[1], 3))
fold=0 # First Fold is whole dataset
key = '_'.join([str(param)for param in params])
print "process", key
name=params[0]

# Load components and projections
####################################################################
components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=fold,key=key)  
components = np.load(components_filename)['arr_0']
projections = np.load(projections_filename)['arr_0']
assert projections.shape[1] == components.shape[1]



#Test correlation of projection with clinical DX
#####################################################################


BASE_PATH = "/neurospin/brainomics/2014_pca_struct/adni"
INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
# Read pop csv
pop = pd.read_csv(INPUT_CSV)

pop["PC1"] = projections[:,0]
pop["PC2"] = projections[:,1]
pop["PC3"] = projections[:,2]


from mulm.dataframe.descriptive_statistics import describe_df_basic
from mulm.dataframe.mulm_dataframe import MULM
from statsmodels.sandbox.stats.multicomp import multipletests
from patsy import dmatrices
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages

PCS = [1, 2, 3]
pdf = PdfPages(os.path.join(OUTPUT_DIR, "sparse_pca_clinic_associations.pdf"))
fig, axarr = plt.subplots(1, 3)
fig.set_figwidth(15)
for j, pc in enumerate(PCS):        
    model = 'y~PC%s+Age+Sex + Center' % (pc)
    y, X = dmatrices(model, data=pop, return_type='dataframe')
    mod = sm.OLS(y, X).fit()
    #test = mod.t_test([0, 1]+[0]*(X.shape[1]-2))
    test = mod.t_test([0,0,1,0,0])
    tval =  test.tvalue[0, 0]
    pval = test.pvalue
    x = pop["PC%i" % pc]
    axarr[j].scatter(y,x,alpha=1)
    axarr[j].set_xlabel('PC%i (T=%.3f, P=%.4g)' % (pc, tval, pval))
fig.tight_layout()
fig
pdf.savefig()  # saves the current figure into a pdf page
plt.close()
pdf.close()    
  




#################################
# Plots PC1 x PC3 color by clinic
#################################

pdf = PdfPages(os.path.join(OUTPUT_DIR, "pca_pc2vs3_clinic_color.pdf"))
y=np.load(os.path.join(INPUT_BASE_DIR,'y.npy')).reshape(X.shape[0])
fig = plt.figure()#, sharey=True)
#fig.set_figwidth(15)
print fig.get_figwidth()
#dt.PC1, dt.PC3
plt.scatter(pop.PC2, pop.PC3, c=pop["DX.num"], s=50)

plt.legend()
plt.colorbar()
plt.xlabel("PC2")
plt.ylabel("PC3")
#axarr[j].set_xticklabels([])
fig.tight_layout()
pdf.savefig()  # saves the current figure into a pdf page
plt.close()
pdf.close()


#Test of sgnificance of Frobenius norm across folds
################################################################################
data = pd.read_csv(INPUT_RESULTS_FILE)

frob=np.zeros((4,5))
frob[:,0]= data.frob_test_fold1
frob[:,1]= data.frob_test_fold2
frob[:,2]= data.frob_test_fold3
frob[:,3]= data.frob_test_fold4
frob[:,4]= data.frob_test_fold5

diff_sparse= frob[3,:] - frob[0,:]
diff_enet= frob[3,:] - frob[1,:]

import scipy.stats
tval, pval=scipy.stats.ttest_1samp(diff_sparse,0.0)
scipy.stats.ttest_1samp(diff_enet,0.0)

scipy.stats.ttest_rel(frob[3,:],frob[1,:])



