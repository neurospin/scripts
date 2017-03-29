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
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import array_utils

  # Compute the Explained variance for each folds, and average it for the graph
 ############################################################################# 

BASE_PATH = "/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_0"
OUTPUT_METRIC_PATH = os.path.join(BASE_PATH,"metrics")

config_filenane = os.path.join(BASE_PATH,"config_dCV.json")
config = json.load(open(config_filenane))

evr_pca=np.zeros((50,10))
evr_sparse=np.zeros((50,10))
evr_enet=np.zeros((50,10))
evr_tv=np.zeros((50,10))

frobenius_reconstruction_error_test_pca = np.zeros((50,11))
frobenius_reconstruction_error_test_sparse =  np.zeros((50,11))
frobenius_reconstruction_error_test_enet =  np.zeros((50,11))
frobenius_reconstruction_error_test_tv =  np.zeros((50,11))

#Cumulativ explained variance ratio
cev_test_pca = np.zeros((50,11))
cev_train_pca = np.zeros((50,11))
cev_test_sparse = np.zeros((50,11))
cev_train_sparse = np.zeros((50,11))
cev_test_enet = np.zeros((50,11))
cev_train_enet = np.zeros((50,11))
cev_test_tv = np.zeros((50,11))
cev_train_tv = np.zeros((50,11))


frobenius_reconstruction_error_train_pca = np.zeros((50,11))
frobenius_reconstruction_error_train_sparse =  np.zeros((50,11))
frobenius_reconstruction_error_train_enet =  np.zeros((50,11))
frobenius_reconstruction_error_train_tv =  np.zeros((50,11))


for set in range(0,50):
    X = np.load("/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/data.std.npy" %(set))

    test_samples =  config['resample']["cv01/all"][1]
    train_samples =  config['resample']["cv01/all"][0]

    X_test = X[test_samples,:]
    X_train = X[train_samples,:]
    
    comp_pca = np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/pca_0.0_0.0_0.0/components.npz'%(set))['arr_0']
    X_transform_test_pca =np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/pca_0.0_0.0_0.0/X_test_transform.npz'%(set))['arr_0']
    X_transform_train_pca =np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/pca_0.0_0.0_0.0/X_train_transform.npz'%(set))['arr_0']


    comp_sparse = np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/sparse_pca_0.0_0.0_1.0/components.npz'%(set))['arr_0']
    X_transform_test_sparse =np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/sparse_pca_0.0_0.0_1.0/X_test_transform.npz'%(set))['arr_0']
    X_transform_train_sparse =np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/sparse_pca_0.0_0.0_1.0/X_train_transform.npz'%(set))['arr_0']


    comp_enet = np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/struct_pca_0.01_0.0001_0.8/components.npz'%(set))['arr_0']
    X_transform_test_enet =np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/struct_pca_0.01_0.0001_0.8/X_test_transform.npz'%(set))['arr_0']
    X_transform_train_enet =np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/struct_pca_0.01_0.0001_0.8/X_train_transform.npz'%(set))['arr_0']


    comp_tv = np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/struct_pca_0.01_0.5_0.5/components.npz'%(set))['arr_0']
    X_transform_test_tv =np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/struct_pca_0.01_0.5_0.5/X_test_transform.npz'%(set))['arr_0']
    X_transform_train_tv =np.load('/neurospin/brainomics/2016_pca_struct/dice/results_10comp/data_100_100_%s/results/cv01/all/struct_pca_0.01_0.5_0.5/X_train_transform.npz'%(set))['arr_0']
      
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
        
            
            frobenius_reconstruction_error_test_pca[set,j] = (np.linalg.norm(X_test - X_predict_test_pca, 'fro'))
            frobenius_reconstruction_error_test_sparse[set,j] =  (np.linalg.norm(X_test - X_predict_test_sparse, 'fro'))
            frobenius_reconstruction_error_test_enet[set,j] =  (np.linalg.norm(X_test - X_predict_test_enet, 'fro'))
            frobenius_reconstruction_error_test_tv[set,j] =  (np.linalg.norm(X_test - X_predict_test_tv, 'fro'))
            
            frobenius_reconstruction_error_train_pca[set,j] = (np.linalg.norm(X_train - X_predict_train_pca, 'fro'))
            frobenius_reconstruction_error_train_sparse[set,j] =  (np.linalg.norm(X_train - X_predict_train_sparse, 'fro'))
            frobenius_reconstruction_error_train_enet[set,j] =  (np.linalg.norm(X_train - X_predict_train_enet, 'fro'))
            frobenius_reconstruction_error_train_tv[set,j] =  (np.linalg.norm(X_train - X_predict_train_tv, 'fro'))
            

            cev_train_pca[set,j] = 1 - ( (np.square(np.linalg.norm(X_train - X_predict_train_pca, 'fro'))))  / np.square((np.linalg.norm(X_train, 'fro'))) 
            cev_train_sparse[set,j] = 1 - ( (np.square(np.linalg.norm(X_train - X_predict_train_sparse, 'fro'))))  / np.square((np.linalg.norm(X_train, 'fro')))
            cev_train_enet[set,j] = 1 - ( (np.square(np.linalg.norm(X_train - X_predict_train_enet, 'fro'))))  / np.square((np.linalg.norm(X_train, 'fro')))
            cev_train_tv[set,j] = 1 - ( (np.square(np.linalg.norm(X_train - X_predict_train_tv, 'fro'))))  / np.square((np.linalg.norm(X_train, 'fro')))
            cev_test_pca[set,j] = 1 - ( (np.square(np.linalg.norm(X_test - X_predict_test_pca, 'fro'))))  / np.square((np.linalg.norm(X_test, 'fro')))
            cev_test_sparse[set,j] = 1 - ( (np.square(np.linalg.norm(X_test - X_predict_test_sparse, 'fro'))))  / np.square((np.linalg.norm(X_test, 'fro')))
            cev_test_enet[set,j] = 1 - ( (np.square(np.linalg.norm(X_test - X_predict_test_enet, 'fro'))))  / np.square((np.linalg.norm(X_test, 'fro')))
            cev_test_tv[set,j] = 1 - ( (np.square(np.linalg.norm(X_test - X_predict_test_tv, 'fro'))))  / np.square((np.linalg.norm(X_test, 'fro')))
            
    print (set)      
        




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





from matplotlib import rc
#Figure of 2x2 plots (reconstrcution error and cumulative explained variance on train and test sets)
##############################################################################################
plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplot(221)
plt.title("Reconstruction Error",fontsize=9)
#pca_plot= plt.plot(np.arange(1,11),frobenius_reconstruction_error_train_pca.mean(axis=0),'y-d',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_train_sparse.mean(axis=0),'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_train_enet.mean(axis=0),'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_train_tv.mean(axis=0),'r-s',markersize=3,label = "SPCA-TV")

plt.ylabel("Train Data",fontsize=9)

plt.subplot(222)

plt.title("Reconstruction Error",fontsize=9)
#pca_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_test_pca.mean(axis=0),'y-d',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_test_sparse.mean(axis=0),'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_test_enet.mean(axis=0),'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(0,11),frobenius_reconstruction_error_test_tv.mean(axis=0),'r-s',markersize=3,label = "SPCA-TV")

plt.ylabel("Test Data",fontsize=9)
plt.subplot(223)
plt.title("Cumulative Explained Variance [\%]",fontsize=9)
#pca_plot= plt.plot(np.arange(0,11),cev_train_pca.mean(axis=0)*100,'y-d',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(0,11),cev_train_sparse.mean(axis=0)*100,'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(0,11),cev_train_enet.mean(axis=0)*100,'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(0,11),cev_train_tv.mean(axis=0)*100,'r-s',markersize=3,label = "SPCA-TV")
plt.ylabel("Train Data",fontsize=9)
plt.xlabel("Number of components",fontsize=9)
plt.subplot(224)
plt.title("Cumulative Explained Variance [\%]",fontsize=9)
#pca_plot= plt.plot(np.arange(0,11),cev_test_pca.mean(axis=0)*100,'y-d',markersize=5,label = "Regular PCA")
sparse_plot= plt.plot(np.arange(0,11),cev_test_sparse.mean(axis=0)*100,'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(0,11),cev_test_enet.mean(axis=0)*100,'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(0,11),cev_test_tv.mean(axis=0)*100,'r-s',markersize=3,label = "SPCA-TV")
plt.xlabel("Number of components",fontsize=9)

plt.ylabel("Test Data",fontsize=9)
plt.tight_layout() 
plt.legend(bbox_to_anchor=(0.6, -0.3),loc= 1,fancybox=True,ncol=3,fontsize = 9)
plt.savefig("/neurospin/brainomics/2016_pca_struct/dice/figures_paper/dice_metrics_plot.pdf", bbox_inches='tight')
##############################################################################################


#Dice index
##############################################################################################

BASE_DIR = "/neurospin/brainomics/2016_pca_struct/dice"
N_COMP = 10
#Load components of all 50 datasets 
##############################################################################
MSE_results = np.zeros((50,4))
frob_test= np.zeros((50,4))
frob_train= np.zeros((50,4))
dice= np.zeros((50,4))
components_pca= np.zeros((10000,10,50))
components_sparse= np.zeros((10000,10,50))
components_enet= np.zeros((10000,10,50))
components_tv= np.zeros((10000,10, 50))

for i in range(50):
       
    INPUT_RESULTS_DIR= os.path.join(BASE_DIR,"results_10comp/data_100_100_%r") % (i) 
    INPUT_DATA_DIR= os.path.join(BASE_DIR,"data_0.1/data_100_100_%r") % (i) 
    INPUT_RESULTS_FILE = os.path.join(INPUT_RESULTS_DIR, "results_dCV_5folds.xlsx")
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
    
    scores_argmax_enet = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 3)
    best_enet_param = scores_argmax_enet.param_key[0]
    scores_argmax_enettv = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 2)
    best_enettv_param = scores_argmax_enettv.param_key[0]
    scores_argmax_sparse = pd.read_excel(INPUT_RESULTS_FILE,sheetname = 4)
    best_sparse_param = scores_argmax_sparse.param_key[0]
  
    #extract components 
    pca_param_path=os.path.join(INPUT_RESULTS_DIR,"results/cv01/all/%s/components.npz") % ("pca_0.0_0.0_0.0")
    components_pca[:,:,i] = np.load(pca_param_path)['arr_0']
    
    sparse_param_path=os.path.join(INPUT_RESULTS_DIR,"results/cv01/all/%s/components.npz") % (best_sparse_param)
    components_sparse[:,:,i] = np.load(sparse_param_path)['arr_0']

    enet_param_path=os.path.join(INPUT_RESULTS_DIR,"results/cv01/all/%s/components.npz") % (best_enet_param)
    components_enet[:,:,i] = np.load(enet_param_path)['arr_0']

    enettv_param_path=os.path.join(INPUT_RESULTS_DIR,"results/cv01/all/%s/components.npz") % (best_enettv_param)
    components_tv[:,:,i] = np.load(enettv_param_path)['arr_0']
    
    print (i)


for i in range(10):
    for j in range(50):
        components_sparse[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_sparse[:,i,j], .99)
        components_enet[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_enet[:,i,j], .99)
        components_tv[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_tv[:,i,j], .99)

##################################################################################

#Pairing of components              
  
components_tv = identify_comp(components_tv)
components_sparse = identify_comp(components_sparse)
components_pca = identify_comp(components_pca)
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


        
components_tv = identify_comp(components_tv)
components_sparse = identify_comp(components_sparse)
components_enet = identify_comp(components_enet)

    
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
plt.title("Similarity \n measurements of weight \n maps across the 50 datasets",fontsize=8)
sparse_plot= plt.plot(np.arange(1,11),dice_sparse,'b-o',markersize=3,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(1,11),dice_enet,'g-^',markersize=3,label = "ElasticNet")
tv_plot= plt.plot(np.arange(1,11),dice_enettv,'r-s',markersize=3,label = "SPCA-TV")
#plt.bar(np.arange(1,11),dice_enettv,color ='red',width = 0.3)
#plt.bar(np.arange(1,11)+0.3,dice_enet,color='green',width = 0.3)
#plt.bar(np.arange(1,11)+0.6,dice_sparse, color='blue',width = 0.3)

plt.xlabel("Number of components",fontsize=8)
plt.ylabel("Dice index",fontsize=8)
plt.legend(loc= 'upper right')
plt.tight_layout() 
plt.legend(bbox_to_anchor=(0.15, -0.15),loc= 1,fancybox=True,ncol=3,fontsize = 8)
plt.savefig("/neurospin/brainomics/2016_pca_struct/dice/figures_paper/dice_metrics+dice_index_plot.pdf", bbox_inches='tight')
plt.savefig("/neurospin/brainomics/2016_pca_struct/submission/dice_metrics+dice_index_plot.pdf", bbox_inches='tight')

##############################################################################################




################################################################################
#explained variance ratio
evr_test_pca = np.zeros((50,10))
evr_train_pca = np.zeros((50,10))
evr_test_sparse = np.zeros((50,10))
evr_train_sparse = np.zeros((50,10))
evr_test_enet = np.zeros((50,10))
evr_train_enet = np.zeros((50,10))
evr_test_tv =np.zeros((50,10))
evr_train_tv = np.zeros((50,10))

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
################################################################################    
    
###############################################################################

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
    