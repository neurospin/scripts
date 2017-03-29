# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:59:39 2017

@author: ad247405
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:29:19 2016

@author: ad247405
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import array_utils
  # Compute the Explained variance for each folds, and average it for the graph
 ############################################################################# 

BASE_PATH = "/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds"
OUTPUT_METRIC_PATH = os.path.join(BASE_PATH,"metrics")

config_filenane = os.path.join(BASE_PATH,"config_dCV.json")
config = json.load(open(config_filenane))

evr_pca=np.zeros((5,10))
evr_sparse=np.zeros((5,10))
evr_enet=np.zeros((5,10))
evr_tv=np.zeros((5,10))

frobenius_reconstruction_error_test_pca = np.zeros((5,11))
frobenius_reconstruction_error_test_sparse =  np.zeros((5,11))
frobenius_reconstruction_error_test_enet =  np.zeros((5,11))
frobenius_reconstruction_error_test_tv =  np.zeros((5,11))

#Cumulativ explained variance ratio
cev_test_pca = np.zeros((5,11))
cev_train_pca = np.zeros((5,11))
cev_test_sparse = np.zeros((5,11))
cev_train_sparse = np.zeros((5,11))
cev_test_enet = np.zeros((5,11))
cev_train_enet = np.zeros((5,11))
cev_test_tv = np.zeros((5,11))
cev_train_tv = np.zeros((5,11))


frobenius_reconstruction_error_train_pca = np.zeros((5,11))
frobenius_reconstruction_error_train_sparse =  np.zeros((5,11))
frobenius_reconstruction_error_train_enet =  np.zeros((5,11))
frobenius_reconstruction_error_train_tv =  np.zeros((5,11))


for cv in range(0,5):
    fold = "cv0%r" %(cv)
    X = np.load("/neurospin/brainomics/2016_pca_struct/adni/data/X.npy")
    fold = fold+'/all'
    test_samples =  config['resample'][fold][1]
    train_samples =  config['resample'][fold][0]

    X_test = X[test_samples,:]
    X_train = X[train_samples,:]
    
    comp_pca = np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/pca_0.0_0.0_0.0/components.npz')['arr_0']
    X_transform_test_pca =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/pca_0.0_0.0_0.0/X_test_transform.npz')['arr_0']
    X_transform_train_pca =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/pca_0.0_0.0_0.0/X_train_transform.npz')['arr_0']


    comp_sparse = np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/sparse_pca_0.0_0.0_1.0/components.npz')['arr_0']
    X_transform_test_sparse =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/sparse_pca_0.0_0.0_1.0/X_test_transform.npz')['arr_0']
    X_transform_train_sparse =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/sparse_pca_0.0_0.0_1.0/X_train_transform.npz')['arr_0']


    comp_enet = np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/struct_pca_0.01_0.0001_0.8/components.npz')['arr_0']
    X_transform_test_enet =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/struct_pca_0.01_0.0001_0.8/X_test_transform.npz')['arr_0']
    X_transform_train_enet =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/struct_pca_0.01_0.0001_0.8/X_train_transform.npz')['arr_0']


    comp_tv = np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/struct_pca_0.1_0.5_0.1/components.npz')['arr_0']
    X_transform_test_tv =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/struct_pca_0.1_0.5_0.1/X_test_transform.npz')['arr_0']
    X_transform_train_tv =np.load('/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/model_selectionCV/'+fold+'/struct_pca_0.1_0.5_0.1/X_train_transform.npz')['arr_0']
      
    pca = np.zeros((1,11))
    sparse = np.zeros((1,11))
    enet = np.zeros((1,11))
    tv = np.zeros((1,11))

    for j in range(0,11):
            X_predict_train_pca = np.dot(X_transform_train_pca[:,:j], comp_pca.T[:j,:])
            X_predict_train_sparse = np.dot(X_transform_train_sparse[:,:j], comp_sparse.T[:j,:])
            X_predict_train_enet = predict(X_train,comp_enet[:,:j])
            X_predict_train_tv = predict(X_train,comp_tv[:,:j])
            X_predict_test_pca = np.dot(X_transform_test_pca[:,:j], comp_pca.T[:j,:])
            X_predict_test_sparse = np.dot(X_transform_test_sparse[:,:j], comp_sparse.T[:j,:])
            X_predict_test_enet = predict(X_test,comp_enet[:,:j])
            X_predict_test_tv = predict(X_test,comp_tv[:,:j])
        
            
            frobenius_reconstruction_error_test_pca[cv,j] = (np.linalg.norm(X_test - X_predict_test_pca, 'fro'))
            frobenius_reconstruction_error_test_sparse[cv,j] =  (np.linalg.norm(X_test - X_predict_test_sparse, 'fro'))
            frobenius_reconstruction_error_test_enet[cv,j] =  (np.linalg.norm(X_test - X_predict_test_enet, 'fro'))
            frobenius_reconstruction_error_test_tv[cv,j] =  (np.linalg.norm(X_test - X_predict_test_tv, 'fro'))
            
            frobenius_reconstruction_error_train_pca[cv,j] = (np.linalg.norm(X_train - X_predict_train_pca, 'fro'))
            frobenius_reconstruction_error_train_sparse[cv,j] =  (np.linalg.norm(X_train - X_predict_train_sparse, 'fro'))
            frobenius_reconstruction_error_train_enet[cv,j] =  (np.linalg.norm(X_train - X_predict_train_enet, 'fro'))
            frobenius_reconstruction_error_train_tv[cv,j] =  (np.linalg.norm(X_train - X_predict_train_tv, 'fro'))
            

            cev_train_pca[cv,j] = 1 - ( (np.square(np.linalg.norm(X_train - X_predict_train_pca, 'fro'))))  / np.square((np.linalg.norm(X_train, 'fro'))) 
            cev_train_sparse[cv,j] = 1 - ( (np.square(np.linalg.norm(X_train - X_predict_train_sparse, 'fro'))))  / np.square((np.linalg.norm(X_train, 'fro')))
            cev_train_enet[cv,j] = 1 - ( (np.square(np.linalg.norm(X_train - X_predict_train_enet, 'fro'))))  / np.square((np.linalg.norm(X_train, 'fro')))
            cev_train_tv[cv,j] = 1 - ( (np.square(np.linalg.norm(X_train - X_predict_train_tv, 'fro'))))  / np.square((np.linalg.norm(X_train, 'fro')))
            cev_test_pca[cv,j] = 1 - ( (np.square(np.linalg.norm(X_test - X_predict_test_pca, 'fro'))))  / np.square((np.linalg.norm(X_test, 'fro')))
            cev_test_sparse[cv,j] = 1 - ( (np.square(np.linalg.norm(X_test - X_predict_test_sparse, 'fro'))))  / np.square((np.linalg.norm(X_test, 'fro')))
            cev_test_enet[cv,j] = 1 - ( (np.square(np.linalg.norm(X_test - X_predict_test_enet, 'fro'))))  / np.square((np.linalg.norm(X_test, 'fro')))
            cev_test_tv[cv,j] = 1 - ( (np.square(np.linalg.norm(X_test - X_predict_test_tv, 'fro'))))  / np.square((np.linalg.norm(X_test, 'fro')))

        
    print(cv)



np.save(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_test_pca"),frobenius_reconstruction_error_test_pca)
np.save(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_train_pca"),frobenius_reconstruction_error_train_pca)
np.save(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_test_sparse"),frobenius_reconstruction_error_test_sparse)
np.save(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_train_sparse"),frobenius_reconstruction_error_train_sparse)
np.save(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_test_enet"),frobenius_reconstruction_error_test_enet)
np.save(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_train_enet"),frobenius_reconstruction_error_train_enet)
np.save(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_test_tv"),frobenius_reconstruction_error_test_tv)
np.save(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_train_tv"),frobenius_reconstruction_error_train_tv)

np.save(os.path.join(OUTPUT_METRIC_PATH,"cev_train_pca"),cev_train_pca)
np.save(os.path.join(OUTPUT_METRIC_PATH,"cev_train_sparse"),cev_train_sparse)
np.save(os.path.join(OUTPUT_METRIC_PATH,"cev_train_enet"),cev_train_enet)
np.save(os.path.join(OUTPUT_METRIC_PATH,"cev_train_tv"),cev_train_tv)
np.save(os.path.join(OUTPUT_METRIC_PATH,"cev_test_pca"),cev_test_pca)
np.save(os.path.join(OUTPUT_METRIC_PATH,"cev_test_sparse"),cev_test_sparse)
np.save(os.path.join(OUTPUT_METRIC_PATH,"cev_test_enet"),cev_test_enet)
np.save(os.path.join(OUTPUT_METRIC_PATH,"cev_test_tv"),cev_test_tv)



frobenius_reconstruction_error_test_pca= np.load(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_test_pca.npy"))
frobenius_reconstruction_error_train_pca = np.load(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_train_pca.npy"))
frobenius_reconstruction_error_test_sparse = np.load(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_test_sparse.npy"))
frobenius_reconstruction_error_train_sparse = np.load(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_train_sparse.npy"))
frobenius_reconstruction_error_test_enet = np.load(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_test_enet.npy"))
frobenius_reconstruction_error_train_enet = np.load(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_train_enet.npy"))
frobenius_reconstruction_error_test_tv = np.load(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_test_tv.npy"))
frobenius_reconstruction_error_train_tv = np.load(os.path.join(OUTPUT_METRIC_PATH,"frobenius_reconstruction_error_train_tv.npy"))

cev_train_pca = np.load(os.path.join(OUTPUT_METRIC_PATH,"cev_train_pca.npy"))
cev_train_sparse = np.load(os.path.join(OUTPUT_METRIC_PATH,"cev_train_sparse.npy"))
cev_train_enet = np.load(os.path.join(OUTPUT_METRIC_PATH,"cev_train_enet.npy"))
cev_train_tv = np.load(os.path.join(OUTPUT_METRIC_PATH,"cev_train_tv.npy"))
cev_test_pca = np.load(os.path.join(OUTPUT_METRIC_PATH,"cev_test_pca.npy"))
cev_test_sparse = np.load(os.path.join(OUTPUT_METRIC_PATH,"cev_test_sparse.npy"))
cev_test_enet = np.load(os.path.join(OUTPUT_METRIC_PATH,"cev_test_enet.npy"))
cev_test_tv = np.load(os.path.join(OUTPUT_METRIC_PATH,"cev_test_tv.npy"))


#Dice index graph
##############################################################################

INPUT_BASE_DIR = "/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"model_selectionCV")         
INPUT_MASK_PATH = "/neurospin/brainomics/2016_pca_struct/adni/data/mask.npy"              
INPUT_RESULTS_FILE=os.path.join(INPUT_BASE_DIR,"results_dCV_5folds.xlsx")                          
INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR,"config_dCV.json")

N_COMP = 10
N_OUTER_FOLDS = 5

number_features = int(np.load(INPUT_MASK_PATH).sum())
scores_cv_sparse = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 7)
scores_cv_enet = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 6)
scores_cv_enettv = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 5)

best_sparse_param = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 4)["param_key"]
best_enet_param = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 3)["param_key"]
best_enettv_param= pd.read_excel(INPUT_RESULTS_FILE,sheetname = 2)["param_key"]


components_sparse = np.zeros((number_features,N_COMP,N_OUTER_FOLDS))
components_enet = np.zeros((number_features,N_COMP,N_OUTER_FOLDS))
components_tv = np.zeros((number_features,N_COMP,N_OUTER_FOLDS))
for i in range(5):
    components_sparse[:,:,i] = np.load(os.path.join(INPUT_DIR,"cv0%s/all" %(i),best_sparse_param[i],"components.npz"))['arr_0']
    components_enet[:,:,i] = np.load(os.path.join(INPUT_DIR,"cv0%s/all" %(i),best_enet_param[i],"components.npz"))['arr_0']
    components_tv[:,:,i] = np.load(os.path.join(INPUT_DIR,"cv0%s/all" %(i),best_enettv_param[i],"components.npz"))['arr_0']
    

for i in range(10):
    for j in range(5):
        components_sparse[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_sparse[:,i,j], .99)
        components_enet[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_enet[:,i,j], .99)
        components_tv[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_tv[:,i,j], .99)

components_tv = identify_comp(components_tv)
components_sparse = identify_comp(components_sparse)
components_enet = identify_comp(components_enet)
  



dice_sparse = list()
all_pairwise_dice_sparse = list()
dice_enet = list()
all_pairwise_dice_enet = list()
dice_enettv = list() 
all_pairwise_dice_enettv = list()
for i in range(N_COMP):
    dice_sparse.append(dice_bar(components_sparse[:,i,:])[0]) #mean of all 10 pairwise dice
    all_pairwise_dice_sparse.append(dice_bar(components_sparse[:,i,:])[1])
    dice_enet.append(dice_bar(components_enet[:,i,:])[0])
    all_pairwise_dice_enet.append(dice_bar(components_enet[:,i,:])[1])
    dice_enettv.append(dice_bar(components_tv[:,i,:])[0])
    all_pairwise_dice_enettv.append(dice_bar(components_tv[:,i,:])[1])
print (np.mean(dice_sparse)) 
print (np.mean(dice_enet))                           
print (np.mean(dice_enettv))

sparse_plot= plt.plot(np.arange(1,11),dice_sparse,'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(1,11),dice_enet,'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(1,11),dice_enettv,'r-s',markersize=3,label = "PCA-TV")
    


#Figure of 2x3 plots (reconstrcution error/cumulative explained variance and Dice index)
##############################################################################################
plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplot(231)
plt.title("Train Set",fontsize=8)
#pca_plot= plt.plot(np.arange(1,11),frobenius_reconstruction_error_train_pca.mean(axis=0),'y-d',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_train_sparse.mean(axis=0),'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_train_enet.mean(axis=0),'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_train_tv.mean(axis=0),'r-s',markersize=3,label = "SPCA-TV")
plt.ylabel("Reconstruction Error",fontsize=8)

plt.subplot(232)
plt.title("Test Set",fontsize=8)
#pca_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_test_pca.mean(axis=0),'y-d',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_test_sparse.mean(axis=0),'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_test_enet.mean(axis=0),'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_test_tv.mean(axis=0),'r-s',markersize=3,label = "SPCA-TV")
plt.ylabel("Reconstruction Error",fontsize=8)

plt.subplot(234)
plt.title("Train Set",fontsize=8)
#pca_plot= plt.plot(np.arange(0,11),cev_train_pca.mean(axis=0)*100,'y-d',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(0,11),cev_train_sparse.mean(axis=0)*100,'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(0,11),cev_train_enet.mean(axis=0)*100,'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(0,11),cev_train_tv.mean(axis=0)*100,'r-s',markersize=3,label = "SPCA-TV")
plt.ylabel("Cumulative \n Explained Variance [\%]",fontsize=8)
plt.xlabel("Number of components",fontsize=8)

plt.subplot(235)
plt.title("Test Set",fontsize=8)
#pca_plot= plt.plot(np.arange(0,11),cev_test_pca.mean(axis=0)*100,'y-d',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(0,11),cev_test_sparse.mean(axis=0)*100,'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(0,11),cev_test_enet.mean(axis=0)*100,'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(0,11),cev_test_tv.mean(axis=0)*100,'r-s',markersize=3,label = "SPCA-TV")
plt.xlabel("Number of components",fontsize=8)
plt.ylabel("Cumulative \n Explained Variance [\%]",fontsize=8)

plt.subplot(133)
plt.title("Similarity \n measurements of \n weight maps across the 5CV.",fontsize=8)
sparse_plot= plt.plot(np.arange(1,11),dice_sparse,'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(1,11),dice_enet,'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(1,11),dice_enettv,'r-s',markersize=3,label = "SPCA-TV")

plt.xlabel("Number of components",fontsize=8)
plt.ylabel("Dice index",fontsize=8)
plt.legend(loc= 'upper right')
plt.tight_layout() 
plt.legend(bbox_to_anchor=(0.15, -0.15),loc= 1,fancybox=True,ncol=3,fontsize = 8)
plt.savefig("/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds/metrics/adni_metrics_plot+diceindex.pdf", bbox_inches='tight')
plt.savefig("/neurospin/brainomics/2016_pca_struct/submission/adni_metrics_plot+diceindex.pdf", bbox_inches='tight')

##############################################################################################



##############################################################################


#explained variance ratio
evr_test_pca = np.zeros((5,10))
evr_train_pca = np.zeros((5,10))
evr_test_sparse = np.zeros((5,10))
evr_train_sparse = np.zeros((5,10))
evr_test_enet = np.zeros((5,10))
evr_train_enet = np.zeros((5,10))
evr_test_tv =np.zeros((5,10))
evr_train_tv = np.zeros((5,10))

evr_train_pca[:,0] = cev_train_pca[:,0]
evr_test_pca[:,0] = cev_test_pca[:,0]
evr_test_sparse[:,0]  = cev_test_sparse[:,0]
evr_train_sparse[:,0]  = cev_train_sparse[:,0]
evr_test_enet[:,0]  = cev_test_enet[:,0]
evr_train_enet[:,0]  = cev_train_enet[:,0]
evr_test_tv[:,0]  = cev_test_tv[:,0]
evr_train_tv[:,0]  = cev_train_tv[:,0]

    
for i in range(1,10):
    evr_train_pca[:,i] = cev_train_pca[:,i] - cev_train_pca[:,i-1]
    evr_test_pca[:,i] = cev_test_pca[:,i] - cev_test_pca[:,i-1]
    evr_train_sparse[:,i] = cev_train_sparse[:,i] - cev_train_sparse[:,i-1]
    evr_test_sparse[:,i] = cev_test_sparse[:,i] - cev_test_sparse[:,i-1]
    evr_train_enet[:,i] = cev_train_enet[:,i] - cev_train_enet[:,i-1]
    evr_test_enet[:,i] = cev_test_enet[:,i] - cev_test_enet[:,i-1]
    evr_train_tv[:,i] = cev_train_tv[:,i] - cev_train_tv[:,i-1]
    evr_test_tv[:,i] = cev_test_tv[:,i] - cev_test_tv[:,i-1]

    
    
pca_plot= plt.plot(np.arange(1,11),evr_train_pca.mean(axis=0)*100,'y-d',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(1,11),evr_train_sparse.mean(axis=0)*100,'b-o',markersize=5,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(1,11),evr_train_enet.mean(axis=0)*100,'g-^',markersize=5,label = "ElasticNet")
tv_plot= plt.plot(np.arange(1,11),evr_train_tv.mean(axis=0)*100,'r-s',markersize=5,label = "PCA-TV")
plt.xlabel("Component Number")
plt.ylabel("Train Data Explained Variance Ratio (%)")
plt.legend(loc= 'upper right')

pca_plot= plt.plot(np.arange(1,11),evr_test_pca.mean(axis=0)*100,'y-d',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(1,11),evr_test_sparse.mean(axis=0)*100,'b-o',markersize=5,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(1,11),evr_test_enet.mean(axis=0)*100,'g-^',markersize=5,label = "ElasticNet")
tv_plot= plt.plot(np.arange(1,11),evr_test_tv.mean(axis=0)*100,'r-s',markersize=5,label = "PCA-TV")
plt.xlabel("Component Number")
plt.ylabel("Test Data Explained Variance Ratio(%)")
plt.legend(loc= 'upper right')
    
    
###############################################################################
      
def identify_comp(comp):
    for i in range(1,N_COMP):
        corr = np.zeros((10,5))
        for j in range(1,N_COMP):
            for k in range(1,N_OUTER_FOLDS):
                #corr[j,k] = np.abs(np.corrcoef(comp[:,i,0],comp[:,j,k]))[0,1]
                map = np.vstack((comp[:,i,0],comp[:,j,k]))
                corr[j,k] = dice_bar(map.T)[0]

        for k in range(1,N_OUTER_FOLDS):
            comp[:,i,k] = comp[:,np.argmax(corr,axis=0)[k],k]
    return comp          


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
        