"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re

from sklearn.cluster import KMeans

## INPUTS ###
#right 'R' or left 'L'
side = 'L'
sides = {'R': 'Right',
         'L': 'Left'}
INPUT = "/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/"

## OUTPUT ##
OUTPUT = "/neurospin/brainomics/2016_sulcal_pits/pheno_pits/test2/"+sides[side]+"/"
OUTPUT2 = OUTPUT+"pits_numeros/"

df = pd.read_csv(INPUT+sides[side]+'_statistic_pits.csv', sep= '\t')
s_ids = df.keys().tolist()

#parcels = {}  # To keep th cluster center coordinates
keys_parcels = np.loadtxt('/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/'+sides[side]+'_parcels_kept.txt')
nb_pits = []
for parcel in keys_parcels:
    nb_pits.append(np.round(np.mean(df.loc[np.int(parcel)])))

DATA_DPF = np.zeros((len(s_ids), sum(nb_pits), 3))

import time
for j, parcel in enumerate(keys_parcels):
    X = np.array([])
    data = np.array([])
    t = time.time()
    for k,s_id in enumerate(s_ids):
        parcels_numbers = np.loadtxt(INPUT+sides[side]+'/kept/parcels/'+side+s_id+'_parcels_numbers_kept.txt')
        lat = np.loadtxt(INPUT+sides[side]+'/kept/lat/'+side+s_id+'_lat_pits_kept.txt')
        lon = np.loadtxt(INPUT+sides[side]+'/kept/lon/'+side+s_id+'_lon_pits_kept.txt')
        DPF = np.loadtxt(INPUT+sides[side]+'/kept/DPF/'+side+s_id+'_DPF_pits_kept.txt')
        pits = np.loadtxt(INPUT+sides[side]+'/kept/pits/'+side+s_id+'_pits_kept.txt')
        parcels_index = np.where(parcels_numbers == parcel)[0]
        temp = np.zeros((len(parcels_index),2))
        temp[:,0]= lat[parcels_index]
        temp[:,1]= lon[parcels_index]
        data_temp = np.zeros((len(parcels_index),3))
        data_temp[:,0] = np.repeat(int(s_id),len(parcels_index))
        data_temp[:,1] = DPF[parcels_index]
        data_temp[:,2] = pits[parcels_index]
        if X.size == 0:
            X = temp
            data = data_temp
        else:
            X = np.concatenate((X,temp))
            data = np.vstack((data,data_temp))
    estimator = KMeans(n_clusters= np.int(np.round(np.mean(df.loc[np.int(parcel)]))), init='random', n_init=100, max_iter=1000, tol=0.0001, n_jobs = -2, verbose=0)
    estimator.fit(X)
    for i in range(X.shape[0]):
        index_id = s_ids.index('%012d' % int(data[i,0]))
        index_start_parcel = sum(nb_pits[:np.where(keys_parcels == parcel)[0]])
        pit_numero = estimator.labels_[i]
        dist = np.linalg.norm(estimator.cluster_centers_[pit_numero]-X[i,:])
        if DATA_DPF[index_id, index_start_parcel+pit_numero,0] != 0: 
            if DATA_DPF[index_id, index_start_parcel+pit_numero,1] > dist:
                DATA_DPF[index_id, index_start_parcel+pit_numero,:] = [data[i,1], dist, data[i,2]]
            else:
                pass
        else:
            DATA_DPF[index_id, index_start_parcel+pit_numero,:] = [data[i,1], dist, data[i,2]]

    print "Elapsed time for parcel " +str(parcel)+ " : "+ str(time.time()-t)


index_columns_kept = []
for j in range(DATA_DPF.shape[1]):
    print np.count_nonzero(DATA_DPF[:,j,0]) 
    if np.count_nonzero(DATA_DPF[:,j,0]) > DATA_DPF.shape[0]-400:
        index_columns_kept.append(j)


for index in index_columns_kept:
    df = pd.DataFrame()
    df['IID'] = np.asarray(s_ids)[np.nonzero(DATA_DPF[:,index,0])].tolist()
    df['FID'] = df['IID']
    df.index = df['IID']
    df2 = df
    df['Pit_'+str(index)] = DATA_DPF[:,index,0][np.nonzero(DATA_DPF[:,index,0])]
    df.to_csv(OUTPUT+sides[side]+'_Pit_'+str(index)+'.phe', sep= '\t',  header=True, index=False)
    df2['Pit_numero'] = DATA_DPF[:,index,2][np.nonzero(DATA_DPF[:,index,0])]
    df2.to_csv(OUTPUT2+sides[side]+'_Pit'+str(index)+'_positions.csv', sep= '\t',  header=True, index=False)

df = pd.DataFrame()
df2 = pd.DataFrame()
for j in range(DATA_DPF.shape[1]):
    df['Pit_'+str(j)] = DATA_DPF[:,j,0]
    df2['Pit_numero'+str(j)] = DATA_DPF[:,j,2]
df[df != 0] = 2
df[df == 0] = 1

df['IID'] = np.asarray(s_ids)
df['FID'] = df['IID']
df.index = df['IID']
columns = df.columns-['IID','FID']
columns_f =['IID','FID']+columns.tolist()
df = df[columns_f]
df2['IID'] = np.asarray(s_ids)
df2['FID'] = df2['IID']
df2.index = df2['IID']
columns = df2.columns-['IID','FID']
columns_f =['IID','FID']+columns.tolist()
df2 = df2[columns_f]


df.to_csv(OUTPUT+sides[side]+'_Pits_binary.phe', sep= '\t',  header=True, index=False)
df2.to_csv(OUTPUT2+sides[side]+'_Pits_binary_positions.csv', sep= '\t',  header=True, index=False)
