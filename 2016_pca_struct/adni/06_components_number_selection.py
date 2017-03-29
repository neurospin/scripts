# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:04:56 2017

@author: ad247405
"""

import numpy as np
import matplotlib.pyplot as plt
import json

config_filename = '/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/config_dCV.json'
config = json.load(open(config_filename))

evr_pca = np.zeros((5,10))
evr_sparse=np.zeros((5,10))
evr_enet=np.zeros((5,10))
evr_tv=np.zeros((5,10))


for cv in range(0,5):
    fold = "cv0%r" %(cv)
    X = np.load("/neurospin/brainomics/2016_pca_struct/adni/data/X.npy")
    fold = fold+'/all'
    test_samples =  config['resample'][fold][1]

    X = X[test_samples,:]
    
    comp_pca = np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/pca_0.0_0.0_0.0/components.npz')['arr_0']
    X_transform_pca =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/pca_0.0_0.0_0.0/X_test_transform.npz')['arr_0']

    comp_sparse = np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/sparse_pca_0.0_0.0_1.0/components.npz')['arr_0']
    X_transform_sparse =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/sparse_pca_0.0_0.0_1.0/X_test_transform.npz')['arr_0']

    comp_enet = np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/struct_pca_0.1_1e-06_0.01/components.npz')['arr_0']
    X_transform_enet =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/struct_pca_0.1_1e-06_0.01/X_test_transform.npz')['arr_0']

    comp_tv = np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/struct_pca_0.1_0.5_0.1/components.npz')['arr_0']
    X_transform_tv =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/struct_pca_0.1_0.5_0.1/X_test_transform.npz')['arr_0']
 
    
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
            f_sparse =( 1 - (((np.linalg.norm(X - X_predict_sparse, 'fro'))))  / (np.linalg.norm(X, 'fro'))) * 2
            f_enet =( 1 - (((np.linalg.norm(X - X_predict_enet, 'fro'))))  / (np.linalg.norm(X, 'fro'))) * 2
            f_tv =( 1 - (((np.linalg.norm(X - X_predict_tv, 'fro'))))  / (np.linalg.norm(X, 'fro'))) * 2
    
            pca[0,j] = f_pca
            sparse[0,j] = f_sparse
            enet[0,j] = f_enet
            tv[0,j] = f_tv
    

    for i in range(1,11):
        evr_pca[cv-1,i-1] = pca[0,i] - pca[0,i-1]
        evr_sparse[cv-1,i-1] = sparse[0,i] - sparse[0,i-1]
        evr_enet[cv-1,i-1] = enet[0,i] - enet[0,i-1]
        evr_tv[cv-1,i-1] = tv[0,i] - tv[0,i-1]


pca_plot= plt.plot(np.arange(1,11),np.abs(evr_pca.mean(axis=0))*100,'y-o',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(1,11),np.abs(evr_sparse.mean(axis=0))*100,'b-o',markersize=5,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(1,11),np.abs(evr_enet.mean(axis=0))*100,'g-^',markersize=5,label = "ElasticNet")
tv_plot= plt.plot(np.arange(1,11),np.abs(evr_tv.mean(axis=0))*100,'r-s',markersize=5,label = "PCA-TV")
plt.xlabel("Component Number")
plt.ylabel("Test Data Explained Variance (%)")
plt.axis([0,10,0,15])
plt.legend(loc= 'upper right')

###############################################################################    

plt.savefig('/neurospin/brainomics/2016_pca_struct/adni/explained_variance_adni.pdf',format='pdf')


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
    