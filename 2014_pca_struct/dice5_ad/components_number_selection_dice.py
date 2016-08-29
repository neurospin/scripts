# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:29:19 2016

@author: ad247405
"""




import numpy as np
import math
import os

import matplotlib.pyplot as plt


pca = np.zeros((10,2))

for i in range(0,2):
    comp_pca = np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_%s/results/0/pca_0.0_0.0_0.0/components.npz'%i)['arr_0']
    X = np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_%s/data.std.npy'%i)
      
    X_transform_pca =np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_%s/results/0/pca_0.0_0.0_0.0/X_test_transform.npz'%i)['arr_0']
    X_predict_pca =np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_%s/results/0/pca_0.0_0.0_0.0/X_test_predict.npz'%i)['arr_0']
       
    pca[i,0] =  np.linalg.norm(X, 'fro')  

    for j in range(1,11):
        X_predict_pca = np.dot(X_transform_pca[:,:j], comp_pca.T[:j,:])
        f_pca = np.linalg.norm(X - X_predict_pca, 'fro')
        pca[i,j] = f_pca
#        
#       
#    
print pca.mean(axis=0)  
   

pca_plot,= plt.plot(np.arange(0,11),pca.mean(axis=0),'r',label = "standard pca")









sparse_plot, = plt.plot(np.arange(0,11),sparse.mean(axis=0) ,'b',label = "sparse pca")
enet_plot, = plt.plot(np.arange(0,11),enet.mean(axis=0) ,'g',label = "Enet pca")
tv_plot, = plt.plot(np.arange(0,11),tv.mean(axis=0) ,'y',label = "pca-tv")
plt.xlabel("# of components")
plt.ylabel("Frobenius norm")
plt.legend(handles=([pca_plot,sparse_plot,enet_plot,tv_plot]))


import numpy as np
import sklearn.decomposition
from sklearn import metrics 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from parsimony.utils import plot_map2d


  
  


X = np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_0/data.std.npy')

comp_pca = np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_0/results/2/pca_0.0_0.0_0.0/components.npz')['arr_0']
X_transform_pca =np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_0/results/2/pca_0.0_0.0_0.0/X_test_transform.npz')['arr_0']
    
comp_sparse = np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_0/results/2/sparse_pca_0.0_0.0_1.0/components.npz')['arr_0']
X_transform_sparse =np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_0/results/2/sparse_pca_0.0_0.0_1.0/X_test_transform.npz')['arr_0']

comp_enet = np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_0/results/2/struct_pca_0.01_1e-05_0.5/components.npz')['arr_0']
X_transform_enet =np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_0/results/2/struct_pca_0.01_1e-05_0.5/X_test_transform.npz')['arr_0']

comp_tv = np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_0/results/2/struct_pca_0.01_0.5_0.5/components.npz')['arr_0']
X_transform_tv =np.load('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_10comp/data_100_100_0/results/2/struct_pca_0.01_0.5_0.5/X_test_transform.npz')['arr_0']
############################################################################### 


X=X[250:]



pca = np.zeros((1,11))
sparse = np.zeros((1,11))
enet = np.zeros((1,11))
tv = np.zeros((1,11))

for j in range(1,11):
        X_predict_pca = np.dot(X_transform_pca[:,:j], comp_pca.T[:j,:])
        X_predict_sparse = np.dot(X_transform_sparse[:,:j], comp_sparse.T[:j,:])
        X_predict_enet = predict(X,comp_enet[:,:j])
        X_predict_tv = predict(X,comp_tv[:,:j])
        
        f_pca =( 1 - (((np.linalg.norm(X - X_predict_pca, 'fro'))))  / (np.linalg.norm(X, 'fro'))) * 2
        f_sparse =( 1 - (((np.linalg.norm(X - X_predict_sparse, 'fro'))))  / (np.linalg.norm(X - X.mean(), 'fro'))) * 2
        f_enet =( 1 - (((np.linalg.norm(X - X_predict_enet, 'fro'))))  / (np.linalg.norm(X, 'fro'))) * 2
        f_tv =( 1 - (((np.linalg.norm(X - X_predict_tv, 'fro'))))  / (np.linalg.norm(X, 'fro'))) * 2

        pca[0,j] = f_pca
        sparse[0,j] = f_sparse
        enet[0,j] = f_enet
        tv[0,j] = f_tv

evr_pca=np.zeros(10)
evr_sparse=np.zeros(10)
evr_enet=np.zeros(10)
evr_tv=np.zeros(10)
for i in range(1,11):
    evr_pca[i-1] = pca[0,i] - pca[0,i-1]
    evr_sparse[i-1] = sparse[0,i] - sparse[0,i-1]
    evr_enet[i-1] = enet[0,i] - enet[0,i-1]
    evr_tv[i-1] = tv[0,i] - tv[0,i-1]





   
pca_plot,= plt.plot(np.arange(1,11),evr_pca*100,'r', label = "standard pca")

sparse_plot= plt.plot(np.arange(1,11),np.abs(evr_sparse)*100,'b-o',markersize=5,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(1,11),np.abs(evr_enet)*100,'g-^',markersize=5,label = "ElasticNet")
tv_plot= plt.plot(np.arange(1,11),np.abs(evr_tv)*100,'r-s',markersize=5,label = "PCA-TV")
plt.xlabel("Component Number")
plt.ylabel("Test Data Explained Variance (%)")
plt.axis([0,10,0,0.5])
plt.legend(loc= 'upper right')

plt.savefig('/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/Figures paper/explained_variance_dice.pdf',format='pdf')



import parsimony.utils.check_arrays as check_arrays
def predict(X,V):
    """ Return the approximated matrix for a given matrix.
    We have to recompute U and d because the argument may not have the same
    number of lines.
    The argument must have the same number of columns than the datset used
    to fit the estimator.
    """
    Xk = check_arrays(X)
    n, p = Xk.shape
    if p != V.shape[0]:
        raise ValueError("The argument must have the same number of "
                         "columns than the datset used to fit the "
                         "estimator.")
    Ut, dt = transform(Xk,V)
    Xt = np.zeros(Xk.shape)
    for k in range(V.shape[1]):
        vk = V[:, k].reshape(-1, 1)
        uk = Ut[:, k].reshape(-1, 1)
        Xt += compute_rank1_approx(dt[k], uk, vk)
    return Xt


def transform(X,V):
    """ Project a (new) dataset onto the components.
    Return the projected data and the associated d.
    We have to recompute U and d because the argument may not have the same
    number of lines.
    The argument must have the same number of columns than the datset used
    to fit the estimator.
    """
    
    Xk = check_arrays(X)
    Xk = Xk.copy()
    n, p = Xk.shape
    if p != V.shape[0]:
        raise ValueError("The argument must have the same number of "
                         "columns than the datset used to fit the "
                         "estimator.")
    U = np.zeros((n,V.shape[1]))
    d = np.zeros((V.shape[1], ))
    for k in range(V.shape[1]):
        # Project on component j
        vk = V[:, k].reshape(-1, 1)
        uk = np.dot(X, vk)
        uk /= np.linalg.norm(uk)
        U[:, k] = uk[:, 0]
        dk = compute_d(X, uk, vk)
        d[k] = dk
        # Residualize
        Xk -= dk * np.dot(uk, vk.T)
    return U, d
    
def compute_d(X, u, v):
    norm_v2 = np.linalg.norm(v)**2
    d = np.dot(u.T, np.dot(X, v)) / norm_v2
    return d    

def compute_rank1_approx(d, u, v):
        """Compute rank 1 approximation given by d, u, v.
           X_approx = d.u.v^t
        """
        X_approx = d * np.dot(u, v.T)
        return X_approx
    