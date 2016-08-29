# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:29:19 2016

@author: ad247405
"""
import numpy as np
import sklearn.decomposition
from sklearn import metrics 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from parsimony.utils import plot_map2d
import json

  # Compute the Explained variance for each folds, and average it for the graph
 ############################################################################# 

config_filenane = '/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds/config_5folds.json'
config = json.load(open(config_filenane))

evr_sparse=np.zeros((5,3))
evr_enet=np.zeros((5,3))
evr_tv=np.zeros((5,3))

X = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds/T_hallu_only.npy')

for cv in range(1,6):
    X = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds/T_hallu_only.npy')
    test_samples =  config['resample'][cv][1] 
    X = X[test_samples,:]
 
    comp_sparse = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds/results/%r/sparse_pca_0.0_0.0_5.0/components.npz'%(cv))['arr_0']
    X_transform_sparse =np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds/results/%r/sparse_pca_0.0_0.0_5.0/X_test_transform.npz'%(cv))['arr_0']

    comp_enet = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds/results/%r/struct_pca_0.1_1e-06_0.5/components.npz'%(cv))['arr_0']
    X_transform_enet =np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds/results/%r/struct_pca_0.1_1e-06_0.5/X_test_transform.npz'%(cv))['arr_0']

    comp_tv = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds/results/%r/struct_pca_0.1_0.8_0.5/components.npz'%(cv))['arr_0']
    X_transform_tv =np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds/results/%r/struct_pca_0.1_0.8_0.5/X_test_transform.npz'%(cv))['arr_0']
   
    sparse = np.zeros((1,4))
    enet = np.zeros((1,4))
    tv = np.zeros((1,4))

    for j in range(1,4):
           
            X_predict_sparse = np.dot(X_transform_sparse[:,:j], comp_sparse.T[:j,:])
            X_predict_enet = predict(X,comp_enet[:,:j])
            X_predict_tv = predict(X,comp_tv[:,:j])
            
          
            f_sparse =( 1 - (((np.linalg.norm(X - X_predict_sparse, 'fro'))))  / (np.linalg.norm(X, 'fro'))) * 2
            f_enet =( 1 - (((np.linalg.norm(X - X_predict_enet, 'fro'))))  / (np.linalg.norm(X, 'fro'))) * 2
            f_tv =( 1 - (((np.linalg.norm(X - X_predict_tv, 'fro'))))  / (np.linalg.norm(X, 'fro'))) * 2
    
       
            sparse[0,j] = f_sparse
            enet[0,j] = f_enet
            tv[0,j] = f_tv
    

    for i in range(1,4):
        evr_sparse[cv-1,i-1] = sparse[0,i] - sparse[0,i-1]
        evr_enet[cv-1,i-1] = enet[0,i] - enet[0,i-1]
        evr_tv[cv-1,i-1] = tv[0,i] - tv[0,i-1]



sparse_plot= plt.plot(np.arange(1,4),np.abs(evr_sparse.mean(axis=0))*100,'b-o',markersize=5,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(1,4),np.abs(evr_enet.mean(axis=0))*100,'g-^',markersize=5,label = "ElasticNet")
tv_plot= plt.plot(np.arange(1,4),np.abs(evr_tv.mean(axis=0))*100,'r-s',markersize=5,label = "PCA-TV")
plt.xlabel("Component Number")
plt.ylabel("Test Data Explained Variance (%)")
plt.axis([0,10,0,15])
plt.legend(loc= 'upper right')






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
    