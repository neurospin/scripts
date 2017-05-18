# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:05:45 2016

@author: ad247405
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from parsimony.datasets.regression import dice5
import pca_tv
import dice5_data
from parsimony.utils import plots
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.algorithms.utils import AlgorithmSnapshot
import time
import parsimony.utils.start_vectors as start_vectors
import parsimony.utils as utils

OUTPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/synthetic_data"
OUTPUT_MASK_DIR = "/neurospin/brainomics/2014_pca_struct/synthetic_data/masks"
OUTPUT_DATA_DIR_FORMAT = os.path.join(OUTPUT_BASE_DIR,"data_{s[0]}_{s[1]}")
OUTPUT_DATASET_FILE = "data.npy"
OUTPUT_STD_DATASET_FILE = "data.std.npy"
OUTPUT_BETA_FILE = "beta3d.npy"
OUTPUT_OBJECT_MASK_FILE_FORMAT = "mask_{i}.npy"
OUTPUT_MASK_FILE = "mask.npy"
OUTPUT_L1MASK_FILE = "l1_max.txt"


if not os.path.exists(OUTPUT_MASK_DIR):
    os.makedirs(OUTPUT_MASK_DIR)



###############################################################################
# All objects
SEED = 42
SHAPE = (25, 25, 1)
N_SAMPLES = 100
objects = dice5.dice_five_with_union_of_pairs(SHAPE)
# We only use union12, d3, union45
_, _, d3, _, _, union12, union45, _ = objects
sub_objects = [union12, union45, d3]
full_mask = np.zeros(SHAPE, dtype=bool)
for i, o in enumerate(sub_objects):
    mask = o.get_mask()
    full_mask += mask
    filename = OUTPUT_OBJECT_MASK_FILE_FORMAT.format(i=i)
    full_filename = os.path.join(OUTPUT_MASK_DIR, filename)
    np.save(full_filename, mask)
full_filename = os.path.join(OUTPUT_MASK_DIR, OUTPUT_MASK_FILE)
np.save(full_filename, full_mask)



# Generate data 
snr=0.5
model = dice5_data.create_model(snr)
#I deleted the seed
X3d, y, beta3d = dice5.load(n_samples=N_SAMPLES,shape=SHAPE,model=model)
                            
# Save data and scaled data
output_dir = OUTPUT_DATA_DIR_FORMAT.format(s=SHAPE)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
X = X3d.reshape(N_SAMPLES, np.prod(SHAPE))
full_filename = os.path.join(output_dir, OUTPUT_DATASET_FILE)
np.save(full_filename, X)
scaler = StandardScaler(with_mean=True, with_std=False)
X_std = scaler.fit_transform(X)
full_filename = os.path.join(output_dir, OUTPUT_STD_DATASET_FILE)
np.save(full_filename, X_std)
# Save beta
full_filename = os.path.join(output_dir, OUTPUT_BETA_FILE)
np.save(full_filename, beta3d)

# Compute l1_max for this dataset
l1_max = pca_tv.PCA_L1_L2_TV.l1_max(X_std)
full_filename = os.path.join(output_dir, OUTPUT_L1MASK_FILE)
with open(full_filename, "w") as f:
    print (>> f)
    print (l1_max)

###############################################################################

X= np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_100_100/data.npy')
SHAPE = (100, 100, 1)

# a, l1, l2, tv penalties
global_pen = 0.01
tv_ratio = 0.5#1e-05
l1_ratio = 0.5

ltv = global_pen * tv_ratio
ll1 = l1_ratio * global_pen * (1 - tv_ratio)
ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
assert(np.allclose(ll1 + ll2 + ltv, global_pen))


Atv = nesterov_tv.A_from_shape(SHAPE)
start_vector=start_vectors.RandomStartVector(seed=42)



##############################################################################
snapshot = AlgorithmSnapshot('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_100_100_bis/').save_nipals
t0 = utils.time_cpu()
mod = pca_tv.PCA_L1_L2_TV(n_components=3,
                                l1=ll1, l2=ll2, ltv=ltv,
                                Atv=Atv,
                                criterion="frobenius",
                                eps=1e-4,
                                max_iter=100,
                                inner_max_iter=int(1e4),
                                output=True,start_vector=start_vector,callback=snapshot)  

mod.fit(X)                                
time  = utils.time_cpu() - t0
print time
#############################################################################




np.save('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_200_200/beta_star.npy',mod.V)
np.save('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_200_200/beta_star_modif.npy',mod.V)




beta_star = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_100_100/beta_star.npy')
plot_map2d(beta_star[:,0].reshape(100,100))
plot_map2d(beta_star[:,1].reshape(100,100))
plot_map2d(beta_star[:,2].reshape(100,100))

#############################################################################
comp1 = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_200_200/component:1.npz')
comp2 = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_200_200/component:2.npz')
comp3 = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_200_200/component:3.npz')

comp1_modif = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_200_200/modifcomponent:1.npz')
comp2_modif = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_200_200/modifcomponent:2.npz')
comp3_modif = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_200_200/modifcomponent:3.npz')

beta_star = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_200_200/beta_star.npy')
beta_star_modif = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_200_200/beta_star_modif.npy')

plot_map2d(beta_star[:,0].reshape(200,200))
plot_map2d(beta_star_modif[:,0].reshape(200,200))
#############################################################################






#############################################################################
comp1 = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_100_100_bis/component:1.npz')
comp2 = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_100_100_bis/component:2.npz')
comp3 = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_100_100_bis/component:3.npz')


comp1_V = (comp1['v'] - comp1['v'][-1,:])
eps_comp1 = np.zeros(comp1_V.shape[0]-1)
time_comp1 = np.zeros(comp1_V.shape[0]-1)
for i in range(0,comp1_V.shape[0]-1):
    eps_comp1[i] = np.linalg.norm(comp1_V[i,:])
    time_comp1[i] = np.linalg.norm(comp1_V[i,:])
    
comp2_V = (comp2['v'] - comp2['v'][-1,:])
eps_comp2 = np.zeros(comp2_V.shape[0]-1)
time_comp2 = np.zeros(comp2_V.shape[0]-1)
for i in range(0,comp2_V.shape[0]-1):
    eps_comp2[i] = np.linalg.norm(comp2_V[i,:])
    time_comp2[i] = np.linalg.norm(comp2_V[i,:])


comp3_V = (comp3['v'] - comp3['v'][-1,:])
eps_comp3 = np.zeros(comp3_V.shape[0]-1)
time_comp3 = np.zeros(comp3_V.shape[0]-1)
for i in range(0,comp3_V.shape[0]-1):
    eps_comp3[i] = np.linalg.norm(comp3_V[i,:])
    time_comp3[i] = np.linalg.norm(comp3_V[i,:])
    
plt.plot(eps_comp1,'o')
plt.plot(eps_comp2,'o')
plt.plot(eps_comp3,'o')
plt.yscale('log')
plt.xlabel('Time in second')
plt.ylabel('precision')
plt.title('synthetic dataset -Sparse PCA')        





import matplotlib.pyplot as plt
plt.plot(comp1['func'],'o')
plt.plot(comp1['v_func'],'o')
plt.plot(comp1['loss'],'o')
plt.plot(comp1['penalties'],'o')
plt.plot(comp1['l1'],'o')
plt.plot(comp1['l2'],'o')
plt.plot(comp1['tv'],'o')


plt.plot((comp1['func']) + comp1['penalties']  ,'o')




plt.plot(comp1['loss'] - comp1['l2']  ,'o')


plt.plot(comp2['func'],'o')
plt.plot(comp2['v_func'],'o')
plt.plot(comp2['loss'],'o')
plt.plot(comp2['penalties'],'o')
plt.plot(comp2['l1'],'o')
plt.plot(comp2['l2'],'o')
plt.plot(comp2['tv'],'o')

plt.plot((comp1['func'])/250. + comp1['penalties']  ,'o')


plt.plot(comp3['func'],'o')
plt.plot(comp3['v_func'],'o')
plt.plot(comp3['loss'],'o')
plt.plot(comp3['penalties'],'o')
plt.plot(comp3['l1'],'o')
plt.plot(comp3['l2'],'o')
plt.plot(comp3['tv'],'o')

plt.plot(comp1['v_func'] - comp1['l2']  ,'o')

#sparse PCA
##################################################################################


X= np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/data_100_100/data.npy')
SHAPE = (100, 100, 1)

# a, l1, l2, tv penalties
global_pen = 0.01
tv_ratio = 0.5#1e-05
l1_ratio = 0.5

ltv = global_pen * tv_ratio
ll1 = l1_ratio * global_pen * (1 - tv_ratio)
ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
assert(np.allclose(ll1 + ll2 + ltv, global_pen))



snapshot =  AlgorithmSnapshot(output_prefix='/neurospin/brainomics/2014_pca_struct/synthetic_data/sparse/').save
#mod = sklearn.decomposition.MiniBatchSparsePCA(n_components=3,verbose=10,alpha=1,n_iter=10000,callback=snapshot)   
cd "git/scripts/2014_pca_struct/"

import sklearn_modified
from sklearn_modified import decomposition


beta_start = start_vectors.RandomStartVector().get_vector(X.shape[1])


mod = sklearn_modified.decomposition.SparsePCA(n_components=3,alpha=1,tol=1e-20,verbose=100,callback=snapshot) 
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
        snapshot = dict(dict_ = locals["dictionary"],time_iter = locals['dt_iter'],code = locals['code'],time=locals["dt"],iter=locals["ii"],error= locals['errors'],current_cost=locals['current_cost'])       
        cpt_str = str(self.cpt)
        output_filename = self.output_prefix + 'ite:%s.npz' % (cpt_str)
        #print "AlgorithmSnapshot.save_conesta: save in ", output_filename
        np.savez_compressed(output_filename, **snapshot)

##############################################################################


cpt=0
time = np.zeros((18))
errors= np.zeros((18))
for i in range (1,19,1):
    if i!= 0:
        a = np.load('/neurospin/brainomics/2014_pca_struct/synthetic_data/sparse/ite:%s.npz' %(i))
        time[cpt] = a['time_iter']
        errors[cpt] = a["current_cost"]
        cpt = cpt+1
        
plt.plot(np.cumsum(time),errors,'o')  
     
precision = errors - errors[-1]  

plt.plot(np.cumsum(time),precision,'o')  
plt.xlabel('Time in second')
plt.ylabel('precision')
plt.yscale('log')
plt.title('synthetic dataset -Sparse PCA')        


import scipy
import scipy.interpolate
from scipy.interpolate import Rbf
    
eps = precision
ius = Rbf(np.abs(eps),np.cumsum(time),function='linear',smooth=0.01) 
x=np.linspace(1e-4,1e-10)
y = ius(x)
plt.plot(np.abs(eps),np.cumsum(time),'o',x,y)
plt.xscale('log')




eps = precision
ius = Rbf(np.cumsum(time),np.abs(eps),function='cubic'),smooth=1) 
x=np.linspace(0,31)
y = ius(x)
plt.plot(np.cumsum(time),np.abs(eps),'o',x,y)
plt.yscale('log')


y=np.interp(x,np.cumsum(time),np.abs(eps))



plt.ylabel('precision')
plt.xlabel('Time in second')
plt.title('fMRI dataset -Sparse PCA')        


print "Total time necessary in order to achieve a precision of 10;", ius(10) 
print "Total time necessary in order to achieve a precision of 1;", ius(1) 
print "Total time necessary in order to achieve a precision of 1e-1;", ius(1e-01) 
print "Total time necessary in order to achieve a precision of 1e-2;", ius(1e-02) 
print "Total time necessary in order to achieve a precision of 1e-3;", ius(1e-03) 
     