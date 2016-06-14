"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""
import pandas as pd
import numpy as np
import os, glob, re

## INPUTS ###
#right 'R' or left 'L'
side = 'L'
sides = {'R': 'Right',
         'L': 'Left'}
INPUT = "/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/"

## OUTPUT ##
OUTPUT = "/neurospin/brainomics/2016_sulcal_pits/pheno_pits/test0/"

path = INPUT+sides[side]+'/kept/pits/'
s_ids = []
for filename in glob.glob(os.path.join(path,'*.txt')):
    m = re.search(path+side+'(.+?)_pits_kept.txt', filename)
    if m:
        label = m.group(1)
        if '000' in label:
            s_ids.append(label)

parcels = {}
parcels_max = {}
parcels_init = {}
keys_parcels = np.loadtxt('/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/'+sides[side]+'_parcels_kept.txt')
s_ids_max = np.loadtxt('/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/'+sides[side]+'_s_ids_max.txt')
s_ids_max  = ['%012d' % int(i) for i in s_ids_max]
max_pits = np.loadtxt('/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/'+sides[side]+'_max_pits.txt')
s_ids_init = np.loadtxt('/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/'+sides[side]+'_s_ids_init.txt')
s_ids_init  = ['%012d' % int(i) for i in s_ids_init]
init_pits = np.loadtxt('/neurospin/brainomics/2016_sulcal_pits/extracting_pits_V0/'+sides[side]+'_init_pits.txt')
# Building our tables for comparison
for j, parcel in enumerate(keys_parcels):
    s_id = s_ids_init[j]
    lat_pits = np.loadtxt(INPUT+sides[side]+'/kept/lat/'+side+s_id+'_lat_pits_kept.txt')
    lon_pits = np.loadtxt(INPUT+sides[side]+'/kept/lon/'+side+s_id+'_lon_pits_kept.txt')
    parcels_numbers = np.loadtxt(INPUT+sides[side]+'/kept/parcels/'+side+s_id+'_parcels_numbers_kept.txt')
    index = np.where(parcels_numbers==parcel)[0].tolist()
    parcels_max[str(parcel)] = max_pits[j]
    parcels_init[str(parcel)] = init_pits[j]
    for i, ind in enumerate(index):
        parcels["Parcel"+str(parcel)+"_pit"+str(i)] = [lat_pits[ind], lon_pits[ind]] 


DATA_DPF = np.zeros((len(s_ids), sum(init_pits),2))
for k,s_id in enumerate(s_ids):
    parcel_previous = np.nan
    #pits_index = np.loadtxt(INPUT+sides[side]+'/kept/pits/'+side+s_id+'_pits_kept.txt')
    parcels_number = np.loadtxt(INPUT+sides[side]+'/kept/parcels/'+side+s_id+'_parcels_numbers_kept.txt')
    DPF = np.loadtxt(INPUT+sides[side]+'/kept/DPF/'+side+s_id+'_DPF_pits_kept.txt')
    lat = np.loadtxt(INPUT+sides[side]+'/kept/lat/'+side+s_id+'_lat_pits_kept.txt')
    lon = np.loadtxt(INPUT+sides[side]+'/kept/lon/'+side+s_id+'_lon_pits_kept.txt')
    print s_id
    for j, parcel in enumerate(parcels_number):
        index_start_parcel = sum(init_pits[:np.where(keys_parcels == parcel)[0]])
        min_dist_pit = np.inf
        pit = np.nan
        for i in range(np.int(parcels_init[str(parcel)])):
            dist = np.square(lat[j]-parcels["Parcel"+str(parcel)+"_pit"+str(i)][0])+np.square(lon[j]-parcels["Parcel"+str(parcel)+"_pit"+str(i)][1])
            if dist < min_dist_pit:
                min_dist_pit = dist
                pit = i
        # case when DPF is really 0 is ignored for th moment
        if not (DATA_DPF[k,index_start_parcel+pit,0] == 0):
            # Here should had a comparison with the previously calculated distance
            #print "Warning you are about to replace an existing DPF value ! " + str(DATA_DPF[k,index_start_parcel+pit])+ " by " + str([DPF[j], dist])
            if DATA_DPF[k,index_start_parcel+pit,1] > dist:
                DATA_DPF[k,index_start_parcel+pit,:] = [DPF[j], dist]
            else:
                #print "Value not replaced."
                pass
        else:
            DATA_DPF[k,index_start_parcel+pit,:] = [DPF[j], dist]
        
        
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
    df['Pit_'+str(index)] = DATA_DPF[:,index,0][np.nonzero(DATA_DPF[:,index,0])]
    df.to_csv(OUTPUT+sides[side]+'_Pit_'+str(index)+'.phe', sep= '\t',  header=True, index=False)
