# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:53:33 2016

@author: ad247405
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from brainomics import plot_utilities
import parsimony.utils.check_arrays as check_arrays

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds/"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"results")
INPUT_DATASET = os.path.join(INPUT_BASE_DIR,"T.npy")
INPUT_MASK = os.path.join(INPUT_BASE_DIR, "mask.nii.gz")                        
INPUT_RESULTS_FILE=os.path.join(INPUT_BASE_DIR,"results.csv")                          
INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR,"config_5folds.json")
OUTPUT_DIR = "/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds/components_selected"
OUTPUT_COMPONENTS = os.path.join(OUTPUT_DIR,"components.csv")

##############
# Parameters #
##############

N_COMP = 3
EXAMPLE_FOLD = 0

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



# Load components and store them as nifti images
####################################################################
data = pd.read_csv(INPUT_RESULTS_FILE)

params=np.array(('pca', '0.0', '0.0', '0.0')) 
params=np.array(('sparse_pca', '0.0', '0.0', '10.0')) 
params=np.array(('sparse_pca', '0.1', '1e-06', '0.5')) 
params=np.array(('struct_pca', '0.1', '0.5', '0.5')) 

components  =np.zeros((63966, 3))
fold=0
key = '_'.join([str(param)for param in params])
print("process", key)
name=params[0]

components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=fold,key=key) 

      
components = np.load(components_filename)['arr_0']
projections = np.load(projections_filename)['arr_0']
assert projections.shape[1] == components.shape[1]

#####################################################################

 
#Correlation of components with clinic (Adjusting for subject) 
#####################################################################


BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"
y=np.load(os.path.join(BASE_PATH,'toward_on','svm','y.npy'))
subject=np.load(os.path.join(BASE_PATH,'toward_on','svm','subject.npy'))


table_patients= np.zeros((165,5))
table_patients[:,0]=subject
table_patients[:,1]=y
table_patients[:,2:]=(projections)
pop = pd.DataFrame(table_patients,columns=["subject","state","PC_0","PC_1","PC_2"])

import statsmodels.api as sm
mlm_full = sm.MixedLM.from_formula('state ~ PC_1 + PC_2 + PC_0', data=pop, groups=pop["subject"]).fit()
print((mlm_full.summary()))
#####################################################################
from sklearn import cluster
mod = cluster.KMeans(n_clusters=2)
predict = mod.fit_predict(projections[y==1,:])


plt.plot(projections[y==1,0][predict==0],projections[y==1,2][predict==0],'o')
plt.plot(projections[y==1,0][predict==1],projections[y==1,2][predict==1],'o')
plt.xlabel('component 0')
plt.ylabel('component 2')
plt.legend()

plt.plot(projections[y==1,0][predict==0],projections[y==1,1][predict==0],'o')
plt.plot(projections[y==1,0][predict==1],projections[y==1,1][predict==1],'o')
plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()

plt.plot(projections[y==1,1][predict==0],projections[y==1,2][predict==0],'o')
plt.plot(projections[y==1,1][predict==1],projections[y==1,2][predict==1],'o')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()




# Plot of components according to clinical status
#####################################################################


plt.plot(projections[y==0,0],projections[y==0,1],'o',label = " No hallucinations")
plt.plot(projections[y==1,0],projections[y==1,1],'o',label = " Preceeding hallucinations")
plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()
plt.savefig('/neurospin/brainomics/2014_pca_struct/fmri/components_analysis/comp0_1.png',format='png')


plt.plot(projections[y==0,0],projections[y==0,2],'o',label = " No hallucinations")
plt.plot(projections[y==1,0],projections[y==1,2],'o',label = " Preceeding hallucinations")
plt.xlabel('component 0')
plt.ylabel('component 2')
plt.legend()
plt.savefig('/neurospin/brainomics/2014_pca_struct/fmri/components_analysis/comp0_2.png',format='png')


plt.plot(projections[y==0,1],projections[y==0,2],'o',label = " No hallucinations")
plt.plot(projections[y==1,1],projections[y==1,2],'o',label = " Preceeding hallucinations")
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()
plt.savefig('/neurospin/brainomics/2014_pca_struct/fmri/components_analysis/comp1_2.png',format='png')


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(projections[y==0,0],projections[y==0,1],projections[y==0,2],c='b',label = " No hallucinations")
ax.scatter(projections[y==1,0],projections[y==1,1],projections[y==1,2],c='g',label = " Preceeding hallucinations")
ax.set_xlabel('component 0')
ax.set_ylabel('component 1')
ax.set_zlabel('component 2')
plt.legend()
plt.savefig('/neurospin/brainomics/2014_pca_struct/fmri/components_analysis/comp1_2_3.png',format='png')
#####################################################################



# Correlate clinicla iunformation on modality of hallu with scores
#####################################################################
#####################################################################
proj_on = projections[y==1].reshape(83,3)
subj_on= subject[y==1]
acv.reshape(23)

acv_block = np.zeros(83)
for i in range(83):
    acv_block[i] = acv[int(subj_on[i])-1]

    
    
data = [proj_on[acv_block==0,0],proj_on[acv_block==1,0]]
plt.boxplot(data)
plt.xticks([1,2],["MMH","Ac-V"])
plt.ylabel("score on component 1")
plt.plot(np.random.normal(1, 0.04, size=proj_on[acv_block==0,0].shape[0]), proj_on[acv_block==0,0],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=proj_on[acv_block==1,0].shape[0]), proj_on[acv_block==1,0],'o',color='b',alpha = 0.3,markersize=5)
   
   
data = [proj_on[acv_block==0,1],proj_on[acv_block==1,1]]
plt.boxplot(data)
plt.xticks([1,2],["MMH","Ac-V"])
plt.ylabel("score on component 2")
plt.plot(np.random.normal(1, 0.04, size=proj_on[acv_block==0,1].shape[0]), proj_on[acv_block==0,1],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=proj_on[acv_block==1,1].shape[0]),proj_on[acv_block==1,1],'o',color='b',alpha = 0.3,markersize=5)


data = [proj_on[acv_block==0,2],proj_on[acv_block==1,2]]
plt.boxplot(data)
plt.xticks([1,2],["MMH","Ac-V"])
plt.ylabel("score on component 2")
plt.plot(np.random.normal(1, 0.04, size=proj_on[acv_block==0,2].shape[0]), proj_on[acv_block==0,2],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=proj_on[acv_block==1,2].shape[0]),proj_on[acv_block==1,2],'o',color='b',alpha = 0.3,markersize=5)


scipy.stats.ttest_ind(proj_on[acv_block==0,0],proj_on[acv_block==1,0])
scipy.stats.ttest_ind(proj_on[acv_block==0,1],proj_on[acv_block==1,1])
scipy.stats.ttest_ind(proj_on[acv_block==0,2],proj_on[acv_block==1,2])

#####################################################################
#####################################################################
#####################################################################

for i in range(1,24):
    plt.plot(proj_on[subj_on==i,1],'o')
    
    
proj_on = projections[y==1].reshape(83,3)
subj_on= subject[y==1]
acv_b.reshape(23)

acv_block = np.zeros(83)
for i in range(83):
    acv_block[i] = acv_b[int(subj_on[i])-1]

    
    
data = [proj_on[acv_block==1,0],proj_on[acv_block==2,0],proj_on[acv_block==3,0],proj_on[acv_block==4,0],proj_on[acv_block==5,0],proj_on[acv_block==6,0],proj_on[acv_block==7,0]]
plt.boxplot(data)
plt.xticks([1,2,3,4,5,6,7],["Ac-V","son","Cen","Ac-V+Cen","Olf","Ac-V+Vision","else"])
plt.ylabel("score on component 1")
plt.plot(np.random.normal(1, 0.04, size=proj_on[acv_block==1,0].shape[0]), proj_on[acv_block==1,0],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=proj_on[acv_block==2,0].shape[0]),proj_on[acv_block==2,0],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(3, 0.04, size=proj_on[acv_block==3,0].shape[0]),proj_on[acv_block==3,0],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(4, 0.04, size=proj_on[acv_block==4,0].shape[0]),proj_on[acv_block==4,0],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(5, 0.04, size=proj_on[acv_block==5,0].shape[0]),proj_on[acv_block==5,0],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(6, 0.04, size=proj_on[acv_block==6,0].shape[0]),proj_on[acv_block==6,0],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(7, 0.04, size=proj_on[acv_block==7,0].shape[0]),proj_on[acv_block==7,0],'o',color='b',alpha = 0.3,markersize=5)
   
   
data = [proj_on[acv_block==1,1],proj_on[acv_block==2,1],proj_on[acv_block==3,1],proj_on[acv_block==4,1],proj_on[acv_block==5,1],proj_on[acv_block==6,1],proj_on[acv_block==7,1]]
plt.boxplot(data)
plt.xticks([1,2,3,4,5,6,7],["Ac-V","son","Cen","Ac-V+Cen","Olf","Ac-V+Vision","else"])
plt.ylabel("score on component 2")
plt.plot(np.random.normal(1, 0.04, size=proj_on[acv_block==1,1].shape[0]), proj_on[acv_block==1,1],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=proj_on[acv_block==2,1].shape[0]),proj_on[acv_block==2,1],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(3, 0.04, size=proj_on[acv_block==3,1].shape[0]),proj_on[acv_block==3,1],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(4, 0.04, size=proj_on[acv_block==4,1].shape[0]),proj_on[acv_block==4,1],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(5, 0.04, size=proj_on[acv_block==5,1].shape[0]),proj_on[acv_block==5,1],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(6, 0.04, size=proj_on[acv_block==6,1].shape[0]),proj_on[acv_block==6,1],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(7, 0.04, size=proj_on[acv_block==7,1].shape[0]),proj_on[acv_block==7,1],'o',color='b',alpha = 0.3,markersize=5)
 

   
data = [proj_on[acv_block==1,2],proj_on[acv_block==2,2],proj_on[acv_block==3,2],proj_on[acv_block==4,2],proj_on[acv_block==5,2],proj_on[acv_block==6,2],proj_on[acv_block==7,2]]
plt.boxplot(data)
plt.xticks([1,2,3,4,5,6,7],["Ac-V","son","Cen","Ac-V+Cen","Olf","Ac-V+Vision","else"])
plt.ylabel("score on component 2")
plt.plot(np.random.normal(1, 0.04, size=proj_on[acv_block==1,2].shape[0]), proj_on[acv_block==1,2],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(2, 0.04, size=proj_on[acv_block==2,2].shape[0]),proj_on[acv_block==2,2],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(3, 0.04, size=proj_on[acv_block==3,2].shape[0]),proj_on[acv_block==3,2],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(4, 0.04, size=proj_on[acv_block==4,2].shape[0]),proj_on[acv_block==4,2],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(5, 0.04, size=proj_on[acv_block==5,2].shape[0]),proj_on[acv_block==5,2],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(6, 0.04, size=proj_on[acv_block==6,2].shape[0]),proj_on[acv_block==6,2],'o',color='b',alpha = 0.3,markersize=5)
plt.plot(np.random.normal(7, 0.04, size=proj_on[acv_block==7,2].shape[0]),proj_on[acv_block==7,2],'o',color='b',alpha = 0.3,markersize=5)
 

