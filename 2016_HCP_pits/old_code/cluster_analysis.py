"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re, json, time
import nibabel.gifti.giftiio as gio

## INPUTS ###
path0 = '/media/yl247234/SAMSUNG/hcp/databaseBV/'
path0 = '/media/yl247234/SAMSUNG/hcp/Freesurfer_mesh_database/'
path = path0+'hcp/'
temp_file_s_ids='/home/yl247234/s_ids_lists/s_ids.json'
sides = ['R', 'L']
with open(temp_file_s_ids, 'r') as f:
    data = json.load(f)
s_ids  = list(json.loads(data))
## CONSTANTS ##
FILTERING = False
FILTERING_PARCEL = False
THRESHOLD_DPF = 1.2
if FILTERING:
    OUTPUT = '/neurospin/brainomics/2016_HCP/pheno_pits_filtered'+str(THRESHOLD_DPF)+'/'
elif FILTERING_PARCEL:
    OUTPUT = '/neurospin/brainomics/2016_HCP/pheno_pits_deepest_filtered/'
else:
    OUTPUT = '/neurospin/brainomics/2016_HCP/pheno_pits_Freesurfer/'

for side in sides:
    file_parcels_on_atlas = path0 +'pits_density/clusters_'+side+'_average_pits_smoothed0.7_60_dist15.0_area100.0_ridge2.0.gii'
    array_parcels = gio.read(file_parcels_on_atlas).darrays[0].data
    parcels_org =  np.unique(array_parcels)
    NB_PARCELS = len(parcels_org)
    DATA_DPF = np.zeros((len(s_ids), NB_PARCELS))*np.nan
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
            for k,parcel in enumerate(parcels_org):
                ind = np.where(parcel == parcels)
                # If the subject has pit in this parcel we consider the deepest
                if ind[0].size:
                    if FILTERING_PARCEL:
                        temp = np.zeros(ind[0].size)
                        array_pits[index_pits[ind[0]]] =temp
                    
                        deepest_pit_val = max(array_DPF[index_pits[ind[0]]])
                        ind2 = np.where(parcel == array_parcels)
                        # Check if this pit corresponds to the deepest DPF value in the parcel #ie hopefully distinguishing superficial pits
                        if deepest_pit_val == max(array_DPF[ind2[0]]):
                            # Multiply by 20 because SOLAR advised so in order to have larger variance so their model is not troubled
                            DATA_DPF[j,k] = 20*deepest_pit_val
                            index = np.argmax(array_DPF[index_pits[ind[0]]])
                            array_pits[index_pits[ind[0]][index]] = 1
                    else:
                        DATA_DPF[j,k] = 20*max(array_DPF[index_pits[ind[0]]])
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
