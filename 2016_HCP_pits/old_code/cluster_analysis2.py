"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re, json, time
import nibabel.gifti.giftiio as gio
from sklearn.cluster import KMeans
import time

## INPUTS ###
path0 = '/media/yl247234/SAMSUNG/hcp/databaseBV/'
path = path0+'hcp/'
temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
sides = ['R', 'L']
with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)
s_ids  = list(json.loads(data))
## CONSTANTS ##
FILTERING = False

THRESHOLD_DPF = 1.2
if FILTERING:
    OUTPUT = '/neurospin/brainomics/2016_HCP/pheno_pits_closest_filtered'+str(THRESHOLD_DPF)+'/'
else:
    OUTPUT = '/neurospin/brainomics/2016_HCP/pheno_pits_closest_filtered/'


for side in sides:
    t0 = time.time()
    file_parcels_on_atlas = path0 +'pits_density/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'
    array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
    parcels_org =  np.unique(array_parcels)
    NB_PARCELS = len(parcels_org)
    DATA_DPF = np.zeros((len(s_ids), NB_PARCELS))*np.nan
    filename_average = '/neurospin/imagen/workspace/cati/BVdatabase/average_'+side+'mesh_BL.gii'
    mesh_average = gio.read(filename_average)
    vertices_origin = mesh_average.darrays[0].data
    triangles_origin = mesh_average.darrays[1].data

    for k,parcel in enumerate(parcels_org):
        t = time.time()
        X = np.array([])
        for j, s_id in enumerate(s_ids):
            file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                              "default_analysis", "segmentation", "mesh",
                                              "surface_analysis", ""+s_id+"_"+side+"white_pits_on_atlas.gii")
            file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                             "default_analysis", "segmentation", "mesh",
                                             "surface_analysis", ""+s_id+"_"+side+"white_DPF_on_atlas.gii")
            if os.path.isfile(file_pits_on_atlas) and os.path.isfile(file_DPF_on_atlas):
                array_pits = gio.read(file_pits_on_atlas).darrays[0].data
                array_DPF = gio.read(file_DPF_on_atlas).darrays[0].data
                if FILTERING:
                    array_pits = array_pits*array_DPF>THRESHOLD_DPF
                    array_pits = array_pits.astype(int)
        
                index_pits = np.nonzero(array_pits)[0]
                parcels = array_parcels[index_pits]
                ind = np.where(parcel == parcels)[0]
                # If the subject has pit in this parcel we consider add their position
                if ind.size:
                    temp = np.zeros((len(ind),3))
                    temp += vertices_origin[index_pits[ind]]
                    if X.size == 0:
                        X = temp
                    else:
                        X = np.concatenate((X,temp))

        estimator = KMeans(n_clusters=1, init='random', n_init=100, max_iter=1000, tol=0.0001, n_jobs = -2, verbose=0)
        estimator.fit(X)
        for j, s_id in enumerate(s_ids):
            file_pits_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                              "default_analysis", "segmentation", "mesh",
                                              "surface_analysis", ""+s_id+"_"+side+"white_pits_on_atlas.gii")
            file_DPF_on_atlas = os.path.join(path, s_id, "t1mri", "BL",
                                             "default_analysis", "segmentation", "mesh",
                                             "surface_analysis", ""+s_id+"_"+side+"white_DPF_on_atlas.gii")
            if os.path.isfile(file_pits_on_atlas) and os.path.isfile(file_DPF_on_atlas):
                array_pits = gio.read(file_pits_on_atlas).darrays[0].data
                array_DPF = gio.read(file_DPF_on_atlas).darrays[0].data
                if FILTERING:
                    array_pits = array_pits*array_DPF>THRESHOLD_DPF
                    array_pits = array_pits.astype(int)

                index_pits = np.nonzero(array_pits)[0]
                parcels = array_parcels[index_pits]
                ind = np.where(parcel == parcels)[0]
                if ind.size:
                    min_dist = np.inf
                    for i in range(ind.size):
                        dist = np.sum(np.square(estimator.cluster_centers_[0]- vertices_origin[index_pits[i]]))
                        if dist < min_dist:
                            pit = i
                            min_dist = dist
                    # Multiply by 20 because SOLAR advised so in order to have larger variance so their model is not troubled
                    DATA_DPF[j,k] = 20*array_DPF[index_pits[i]]
        print "Elapsed time for parcel " +str(parcel)+ " : "+ str(time.time()-t)
    # We will not consider subject with exactly 0 for now
    # Else use find zeros of numpy and replace them with almost 0
    DATA_DPF = np.nan_to_num(DATA_DPF)
    index_columns_kept = []
    for j in range(DATA_DPF.shape[1]):
        print np.count_nonzero(DATA_DPF[:,j])
        if np.count_nonzero(DATA_DPF[:,j]) > DATA_DPF.shape[0]*0.5:
            index_columns_kept.append(j)

    print index_columns_kept
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    for index in index_columns_kept:
        df3 = pd.DataFrame()
        df3['IID'] = np.asarray(s_ids)[np.nonzero(DATA_DPF[:,index])].tolist()
        df3['Parcel_'+str(int(parcels_org[index]))] = DATA_DPF[:,index][np.nonzero(DATA_DPF[:,index])]
        if not os.path.exists(OUTPUT):
            os.makedirs(OUTPUT)
        output = OUTPUT+'DPF_pit'+str(int(parcels_org[index]))+"side"+side
        df3.to_csv(output+'.csv',  header=True, index=False)

    df = pd.DataFrame()
    for j in range(DATA_DPF.shape[1]):
        df['Parcel_'+str(int(parcels_org[j]))] = DATA_DPF[:,j]
    df[df != 0] = 2
    df[df == 0] = 1
    df['IID'] = np.asarray(s_ids)
    OUTPUT2 = OUTPUT + 'case_control/'
    if not os.path.exists(OUTPUT2):
        os.makedirs(OUTPUT2)
    output = OUTPUT2+'all_pits_side'+side
    df.to_csv(output+'.csv', sep= ',',  header=True, index=False)
    print "Elapsed time for side " +str(side)+ " : "+ str(time.time()-t0)
