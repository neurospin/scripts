# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:02:45 2016

@author: ad247405
"""


#sklearn convergence time computation

###############################################################################

snapshot =  AlgorithmSnapshot(output_prefix='/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/sparse_s/').save
#mod = sklearn.decomposition.MiniBatchSparsePCA(n_components=3,verbose=10,alpha=1,n_iter=10000,callback=snapshot)   
cd "git/scripts/2014_pca_struct/"

import sklearn_modified
from sklearn_modified import decomposition

mod = sklearn_modified.decomposition.SparsePCA(n_components=3,alpha=10.0,tol=1e-20,verbose=100,callback=snapshot) 
mod.fit(X)
############################################################################### 

class AlgorithmSnapshot:
 
    def __init__(self, output_prefix, saving_period=1):
        self.output_prefix = output_prefix
        self.saving_period = saving_period
        self.cpt = 0

    def save(self, locals):
        self.cpt += 1
        if (self.cpt % self.saving_period) != 0:
            return
        snapshot = dict(dict_ = locals["dictionary"],time_iter = locals['dt_iter'],code = locals['code'],time=locals["dt"],iter=locals["ii"],error= locals['errors'])       
        cpt_str = str(self.cpt)
        output_filename = self.output_prefix + 'ite:%s.npz' % (cpt_str)
        #print "AlgorithmSnapshot.save_conesta: save in ", output_filename
        np.savez_compressed(output_filename, **snapshot)
 

       
        
cpt=0
frob = np.zeros((146)) 
dict_ = np.zeros((146,165,3))
code = np.zeros((146,63966,3))
time = np.zeros((146))
for i in range (1,147,1):
    if i!= 0:
        a = np.load('/neurospin/brainomics/2014_pca_struct/fmri/fmri_time/sparse/ite:%s.npz' %(i))
        dict_[cpt,:,:] = a['dict_'].T
        code[cpt,:,:] = a['code']
        time[cpt] = a['time']
        frob[cpt] = np.linalg.norm(X - np.dot(dict_[cpt,:,:],code[cpt,:,:].T),ord='fro') 
        cpt = cpt+1
        

       
eps = np.zeros((146))
eps = frob -  frob[-1] 


import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
from scipy.interpolate import Rbf
plt.plot(time,np.abs(eps))
plt.xlabel('Time in second')
plt.ylabel('precision')
plt.yscale('log')
plt.title('fMRI dataset - MiniBatch Sparse PCA')        



ius = Rbf(np.abs(eps),(time),smooth=0.1) 
x= np.linspace(0,np.abs(eps).max())
y = ius(x)
plt.plot(np.abs(eps),(time),'o',x,y)

print "Total time necessary in order to achieve a precision of 1e-2;", ius(1e-02) 
print "Total time necessary in order to achieve a precision of 1e-3;", ius(1e-03) 
print "Total time necessary in order to achieve a precision of 1e-4;", ius(1e-04) 
print "Total time necessary in order to achieve a precision of 1e-5;", ius(1e-05) 
print "Total time necessary in order to achieve a precision of 1e-6;", ius(1e-06) 
 